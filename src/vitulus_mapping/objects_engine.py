#!/usr/bin/env python3
"""objects_engine.py -- the logic of map_objects_server, without ROS.

The node owns the topics and feeds this class from ONE thread: handle() per
/map_objects/op, tick() at 1 Hz. Nothing here runs concurrently, so the watcher
never sees a forwarded op half done and a forwarded write is journaled once: the
op that made it records it and updates the baseline before the watcher looks.

`owners` is what talks to navi_man, node_planner and mapping_manager; tests hand
in a fake that writes the files itself. Interface:
    site()                         the served site, or None
    not_ready(family, site)        reason the owner cannot take a write, or None
    save(family, obj)              obj in the map frame, full object
    remove(family, name)
    rename(family, old, new)
    failure(family, name, since)   owner's refusal since that time, or None
    running_program()              name of the program being mowed, or None
    mission_busy()                 a mission is on (robot may be out, blades too)
    save_program(name, zone_names, added)  the planner's program save, zones only
A method may raise Refuse. Owners confirm through the files: navi_man has no
result channel, and its log says nothing about whether the export ran.

Baseline: the last state of each family this engine has accounted for, kept in
history/snapshot.json so a change made while the node was down is still seen and
a rename does not lose its id. A family whose file is missing keeps its baseline:
no information, nothing is deleted.
"""

import json
import os
import time

from vitulus_mapping import bundle, objects

try:
    from vitulus_mapping import suggestions
except ImportError:
    suggestions = None

UNDO_REACH = 200
HISTORY_ENTRIES = 100
ACTORS = ("owner", "agent", "robot", "unknown")
EDITABLE = ("type", "name", "points", "yaw", "aliases", "note", "tags",
            "valid_from", "valid_until", "props")
META_ONLY = ("aliases", "note", "tags")
NO_SITE_HOLD_S = 30.0
PROGRAMS_KEY = "_programs"     # in the baseline: program -> its zones
_UNSET = object()


class Refuse(Exception):
    """An op that cannot be done; args[0] is the reason for /map_objects/result."""


# --------------------------------------------------------------------------
# journal
# --------------------------------------------------------------------------
def read_journal(path):
    """Entries of a journal file. A line torn by a crash is skipped."""
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def append_journal(path, entries):
    """One write of whole lines. If the file ends in a torn line, end it first so
    the crash loses that entry only."""
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    try:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                text = "\n" + text
    except OSError:
        pass
    with open(path, "a") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def split_steps(entries):
    """Consecutive entries of one group are one undo step."""
    steps = []
    for e in entries:
        if steps and e.get("group") and steps[-1][-1].get("group") == e.get("group"):
            steps[-1].append(e)
        else:
            steps.append([e])
    return steps


def _undoable(e):
    """The dock flow rewrites the dock points by itself. Those stay in the journal
    but are no step of anyone's undo, or undo would move the dock under the robot."""
    return not (e.get("actor") == "unknown"
                and objects.is_protected(e.get("family"), e.get("name")))


def compute_stacks(entries, reach=UNDO_REACH):
    """(undo, redo) lists of steps, derived from the journal alone: the history
    is never rewritten, an undo is an entry like any other."""
    undo, redo = [], []
    for step in split_steps(entries):
        op = step[0].get("op")
        if op == "undo":
            if undo:
                redo.append(undo.pop())
        elif op == "redo":
            if redo:
                undo.append(redo.pop())
        else:
            kept = [e for e in step if _undoable(e)]
            if kept:
                undo.append(kept)
                redo = []
    return undo[-reach:], redo


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
class Engine:
    def __init__(self, owners, emit, clock=time.time, sleep=time.sleep,
                 log=lambda text: None, confirm_s=None, ready_s=5.0):
        self.owners = owners
        self.emit = emit
        self.clock = clock
        self.sleep = sleep
        self.log = log
        # How long an owner gets to show the change in its file. A zone is
        # regenerated first (coverage paths), that takes a while.
        self.confirm_s = confirm_s or {"waypoint": 6.0, "path": 6.0, "edit": 8.0,
                                       "zone": 40.0, "program": 8.0}
        self.ready_s = ready_s
        self.site = _UNSET
        self.datum = None
        self.reason = None
        self.baseline = {}
        self.meta = {}
        self.entries = []
        self.undo, self.redo = [], []
        self.late = {}             # (family, name) -> {actor, until}: op gave up
        self._mtimes = None
        self._next_rules = 0.0
        self._last_objects = None
        self._last_rules = None
        self._last_history = None
        self._last_suggestions = None
        # Survives a restart without going backwards for a seq-comparing reader.
        self._pub_seq = int(clock())
        self._warnings = []
        self._actor = "unknown"
        self._applied_before = None
        self._no_site_since = None
        self._site_rules = (None, [], [])

    # ---- paths ------------------------------------------------------------
    def _journal_path(self):
        return os.path.join(objects.history_dir(self.site), "journal.jsonl")

    def _snapshot_path(self):
        return os.path.join(objects.history_dir(self.site), "snapshot.json")

    def _watched(self):
        files = [bundle.zones_path(self.site), bundle.waypoints_path(self.site),
                 bundle.paths_path(self.site), bundle.edits_path(self.site),
                 bundle.programs_path(self.site), objects.places_path(self.site),
                 os.path.join(bundle.site_dir(self.site), "datum.yaml")]
        return tuple(os.path.getmtime(p) if os.path.exists(p) else None for p in files)

    # ---- site -------------------------------------------------------------
    def switch(self, site):
        """Load everything of a site; the old state goes with the old site."""
        self.site = site
        self.datum, self.reason = None, None
        self.baseline, self.meta, self.entries = {}, {}, []
        self.undo, self.redo = [], []
        self.late = {}
        self._mtimes = None
        self._last_history = None
        self._last_suggestions = None
        if site is None or not bundle.site_exists(site):
            self.reason = "no_site"
        else:
            try:
                self._load_site()
            except Exception as e:
                self.log("loading %s failed: %r" % (site, e))
        self.publish_all()

    def _load_site(self):
        # Before the sidecar is read: the migration moves its keys, ids survive.
        try:
            bundle.migrate_dock_names(self.site)
        except Exception as e:
            self.log("dock name migration failed on %s: %s" % (self.site, e))
        self.datum = objects.load_datum(self.site)
        self.reason = None if self.datum else "no_datum"
        meta = objects.load_meta(self.site)
        if meta is None:
            # Unreadable, not absent: keep the bytes, ids fall back to default_id.
            p = objects.meta_path(self.site)
            os.replace(p, "%s.corrupt.%d" % (p, int(self.clock())))
            self.log("unreadable %s moved aside" % p)
            meta = {}
        self.meta = meta
        self.entries = read_journal(self._journal_path())
        self.undo, self.redo = compute_stacks(self.entries)
        fams = objects.load_families(self.site, self.clock(), self.meta)
        snapshot = self._read_snapshot()
        if snapshot is None:
            # First contact with this site: what is there just gets ids.
            self.baseline = fams
            self._note_programs()
            self._sync_meta()
            self._write_snapshot()
        else:
            self.baseline = snapshot
            if PROGRAMS_KEY not in self.baseline:
                self._note_programs()
            self._absorb(fams)
        self._mtimes = self._watched()

    def _read_snapshot(self):
        try:
            with open(self._snapshot_path()) as f:
                snap = json.load(f)
            return snap if isinstance(snap, dict) else None
        except (OSError, ValueError):
            return None

    def _write_snapshot(self):
        bundle._write_json(self._snapshot_path(), self.baseline)

    # ---- watcher ----------------------------------------------------------
    def tick(self):
        now = self.clock()
        site = self.owners.site()
        if self._no_site_is_transient(site, now):
            if self.site is _UNSET:
                return
            site = self.site
        if site != self.site:
            self.switch(site)
            self._next_rules = now + 5.0
            return
        try:
            self.scan()
        except Exception as e:           # the rules below still go out
            self.log("scan failed: %r" % (e,))
        # Temporary rules start and end with no file changing.
        if now >= self._next_rules:
            self._next_rules = now + 5.0
            self.publish_all()

    def _no_site_is_transient(self, site, now):
        """mapping_manager keeps `serving` in memory: its respawn says "nothing
        served" until it has served again, and this node's own start knows no site
        yet. Taking that at its word would lift every rule under a running mission,
        so nothing changes until no site has lasted, and never during a mission. A
        new site is no doubt and switches at once."""
        if site is not None or self.site is None:
            self._no_site_since = None
            return False
        if self._no_site_since is None:
            self._no_site_since = now
        return now - self._no_site_since < NO_SITE_HOLD_S or self.owners.mission_busy()

    def scan(self):
        """Account for anything that changed without us. Nothing to do while the
        watched files stand still, so this is one stat per file at 1 Hz."""
        if self.site in (None, _UNSET) or not bundle.site_exists(self.site):
            return
        stamp = self._watched()
        if stamp == self._mtimes:
            return
        self._mtimes = stamp
        self.datum = objects.load_datum(self.site)
        self.reason = None if self.datum else "no_datum"
        self._absorb(objects.load_families(self.site, self.clock(), self.meta))
        self.publish_all()

    def _absorb(self, fams):
        entries = []
        for fam in objects.FAMILIES:
            items = fams[fam]
            if items is None:
                continue             # no information
            if fam == "place":
                bad = []
                objects.load_places(self.site, bad)
                if bad:
                    continue         # a feature that went bad is no deletion
            changes = objects.diff_family(self.baseline.get(fam) or [], items)
            group = self._new_group() if len(changes) > 1 else None
            for before, after in changes:
                entries.append(self._observed(fam, before, after, group))
            self.baseline[fam] = items
        self._note_programs()     # after: a deleted zone needs the old table
        dirty = self._sync_meta()
        if entries:
            self._commit(entries)
        elif dirty:
            self._write_snapshot()

    def _note_programs(self):
        table = objects.program_zone_names(self.site)
        if table is not None:      # no information keeps what we knew
            self.baseline[PROGRAMS_KEY] = table

    def _with_memberships(self, zone, table):
        """The zone carrying the programs that use it, for its undo."""
        used = [{"name": p, "zone_names": z} for p, z in (table or {}).items()
                if zone["name"] in z]
        out = dict(zone, props=dict(zone.get("props") or {}))
        if used:
            out["props"]["programs"] = used
        return out

    def _observed(self, family, before, after, group):
        """One change found in the files. The geometry is the file's, what a
        person attached to the object travels with it across a rename."""
        now = self.clock()
        name = (after or before)["name"]
        late = self.late.pop((family, name), None)
        actor = late["actor"] if late and now < late["until"] else "unknown"
        if after is not None:
            if family != "place":
                if before is not None:
                    for k in objects.META_FIELDS:
                        after[k] = before.get(k, after.get(k))
                else:
                    known = self.meta.get(objects.meta_key(family, name))
                    if not known:
                        after["created"] = now
                        if actor in objects.SOURCES:
                            after["source"] = actor
                after["updated"] = now
                self._set_meta(after, before)
        else:
            self._drop_meta(before)
            if family == "zone":
                # Outside delete: the planner has already dropped it from the
                # programs, so the table from before that is the only record.
                before = self._with_memberships(before, self.baseline.get(PROGRAMS_KEY))
        op = "save" if before is None else "delete" if after is None else "update"
        return self._entry(actor, group, op, before, after)

    # ---- meta -------------------------------------------------------------
    def _set_meta(self, obj, before=None):
        if obj["family"] == "place":
            return
        if before is not None and before["name"] != obj["name"]:
            self.meta.pop(objects.meta_key(obj["family"], before["name"]), None)
        self.meta[objects.meta_key(obj["family"], obj["name"])] = {
            k: obj.get(k) for k in objects.META_FIELDS}

    def _drop_meta(self, obj):
        if obj["family"] != "place":
            self.meta.pop(objects.meta_key(obj["family"], obj["name"]), None)

    def _sync_meta(self):
        """Every object has its id in the sidecar; save only if something moved.
        Keys of a family with no information stay: they are not deletions."""
        dirty = False
        for fam in objects.FAMILIES:
            if fam == "place":
                continue
            for o in self.baseline.get(fam) or []:
                key = objects.meta_key(fam, o["name"])
                if key not in self.meta:
                    self.meta[key] = {k: o.get(k) for k in objects.META_FIELDS}
                    dirty = True
        if dirty or not os.path.exists(objects.meta_path(self.site)):
            objects.save_meta(self.site, self.meta)
        return dirty

    # ---- journal ----------------------------------------------------------
    def _new_group(self):
        return "g_" + objects.new_id()[2:]

    def _entry(self, actor, group, op, before, after):
        some = after or before
        return {"seq": None, "ts": self.clock(), "actor": actor, "site": self.site,
                "group": group, "op": op, "family": some["family"], "id": some["id"],
                "name": some["name"], "before": _plain(before), "after": _plain(after)}

    def _commit(self, entries):
        seq = self.entries[-1]["seq"] if self.entries else 0
        for e in entries:
            seq += 1
            e["seq"] = seq
        append_journal(self._journal_path(), entries)
        self.entries.extend(entries)
        self.undo, self.redo = compute_stacks(self.entries)
        objects.save_meta(self.site, self.meta)
        self._write_snapshot()

    # ---- ops --------------------------------------------------------------
    def handle(self, msg):
        result = {"req": msg.get("req") if isinstance(msg, dict) else None, "ok": False,
                  "reason": None, "id": None, "warnings": []}
        self._warnings = []
        try:
            if not isinstance(msg, dict):
                raise Refuse("bad_request")
            self.scan()
            result["id"] = self._dispatch(msg)
            result["ok"] = True
        except Refuse as e:
            result["reason"] = str(e.args[0])
        except Exception as e:           # a bad op must not end the node
            self.log("op failed: %r" % (e,))
            result["reason"] = "internal_error"
        result["warnings"] = self._warnings + [
            "bad_rule:%s:%s:%s" % (p["id"], p["name"], p["reason"])
            for p in self._site_rules[2]]
        # What the op left in the files; ours is accounted for already.
        if self.site not in (None, _UNSET):
            self._mtimes = None
            self.scan()
        self.emit("result", result)
        return result

    def _dispatch(self, msg):
        actor = msg.get("actor", "unknown")
        if actor not in ACTORS:
            raise Refuse("bad_actor")
        self._actor = actor
        if self.site in (None, _UNSET):
            raise Refuse("no_site")
        if self.owners.site() != self.site or msg.get("site", self.site) != self.site:
            raise Refuse("site_changed")
        if self.reason:
            raise Refuse(self.reason)
        op = msg.get("op")
        if op in ("undo", "redo"):
            return self._undo_redo(op, actor)
        if op == "group":
            return self._group(msg, actor)
        if op == "suggestion":
            return self._suggestion(msg, actor)
        if op in ("save", "update", "delete"):
            done = []
            self._one(msg, actor, None, done)
            if not done:
                return msg.get("id") or (msg.get("object") or {}).get("id")
            self._commit([e for e, _, _ in done])
            return done[0][0]["id"]
        raise Refuse("bad_op")

    def _one(self, msg, actor, group, done):
        before, after = self._plan(msg, actor)
        if before is not None and after is not None and _unchanged(before, after):
            return
        self._guard(actor, before, after)
        actual = self._apply(before, after)
        before = self._applied_before or before
        op = "save" if before is None else "delete" if after is None else "update"
        done.append((self._entry(actor, group, op, before, actual), before, actual))

    def _guard(self, actor, before, after):
        """Owner's rule: an agent cannot take a nogo away. Checked where a change
        is decided, not in _apply: a rollback only puts things back."""
        if actor == "agent":
            reason = objects.owner_nogo_problem(before, after, self.clock())
            if reason:
                raise Refuse(reason)

    def _group(self, msg, actor):
        ops = msg.get("ops")
        if not isinstance(ops, list) or not ops:
            raise Refuse("bad_request")
        group = self._new_group()
        done = []
        try:
            for sub in ops:
                if not isinstance(sub, dict) or sub.get("op") not in ("save", "update", "delete"):
                    raise Refuse("bad_op")
                self._one(sub, actor, group, done)
        except Refuse:
            self._rollback(done)
            raise
        if not done:
            return None
        self._commit([e for e, _, _ in done])
        return done[0][0]["id"]

    def _rollback(self, done):
        for _, before, actual in reversed(done):
            try:
                self._apply(actual, before)
            except Refuse as e:
                self.log("rollback of %s failed: %s"
                         % ((before or actual)["name"], e.args[0]))
                self._warnings.append("rollback_failed:" + (before or actual)["name"])

    def _undo_redo(self, op, actor):
        stack = self.undo if op == "undo" else self.redo
        if not stack:
            raise Refuse("nothing_to_" + op)
        step = stack[-1]
        group = self._new_group() if len(step) > 1 else None
        done = []
        current = self._current()
        order = list(reversed(step) if op == "undo" else step)
        # All of it or nothing: refuse before the first entry is applied.
        for e in order:
            self._guard(actor, objects.find_by_id(current, e["id"]),
                        e["before"] if op == "undo" else e["after"])
        try:
            for e in order:
                target = e["before"] if op == "undo" else e["after"]
                now_obj = objects.find_by_id(current, e["id"])
                if target is None and now_obj is None:
                    continue
                if target is not None:
                    target = dict(target)
                    target["updated"] = self.clock()
                    self._check_restore(target)
                actual = self._apply(now_obj, target)
                done.append((self._entry(actor, group, op, now_obj, actual), now_obj, actual))
                current = self._current()
        except Refuse:
            self._rollback(done)
            raise
        if done:
            self._commit([d[0] for d in done])
        return step[0]["id"]

    def _check_restore(self, target):
        """A rule comes back only if it is still valid today: the dock may have
        moved under it since. Restoring is no reason to check anything else."""
        if target["family"] != "place":
            return
        v = objects.validate(objects.to_frame(target, self.datum, "map"),
                             objects=self._current_map())
        if not v["ok"]:
            raise Refuse(v["reason"])
        self._warnings.extend(v["warnings"])

    def _suggestion(self, msg, actor):
        if suggestions is None:
            raise Refuse("not_implemented")
        sid, action = msg.get("id"), msg.get("action")
        if action == "ignore":
            ok, reason = suggestions.mark(self.site, sid, "ignored")
            if not ok:
                raise Refuse(reason or "failed")
            self._last_suggestions = None
            return sid
        if action != "accept":
            raise Refuse("bad_request")
        obj, reason = suggestions.accept_object(self.site, sid, msg.get("as") or {}, self.datum)
        if obj is None:
            raise Refuse(reason or "failed")
        done = []
        self._one({"op": "save", "object": obj}, actor, None, done)
        if not done:
            raise Refuse("failed")
        self._commit([e for e, _, _ in done])
        suggestions.mark(self.site, sid, "accepted")
        self._last_suggestions = None
        return done[0][0]["id"]

    # ---- planning ---------------------------------------------------------
    def _current(self):
        fams = objects.load_families(self.site, self.clock(), self.meta)
        return [o for f in objects.FAMILIES for o in (fams[f] or [])]

    def _current_map(self):
        return [objects.to_frame(o, self.datum, "map") for o in self._current()]

    def _plan(self, msg, actor):
        """(before, after), both UTM; None = does not exist before/after."""
        op = msg.get("op")
        current = self._current()
        if op == "delete":
            before = objects.find_by_id(current, msg.get("id"))
            if before is None:
                raise Refuse("not_found")
            if before["protected"]:
                raise Refuse("protected")
            return before, None
        if op == "update":
            before = objects.find_by_id(current, msg.get("id"))
            if before is None:
                raise Refuse("not_found")
            fields = msg.get("fields") if isinstance(msg.get("fields"), dict) else msg
            if not any(k in fields for k in EDITABLE if k not in META_ONLY):
                # Nothing the validator or an owner has a say in, and an old
                # shape it would refuse today must not block a note.
                after = dict(before)
                for k in META_ONLY:
                    if k in fields:
                        after[k] = fields[k]
                after["updated"] = self.clock()
                return before, after
            raw = objects.to_frame(before, self.datum, "map")
            for k in EDITABLE:
                if k in fields and k != "props":
                    raw[k] = fields[k]
            props = fields.get("props") if isinstance(fields.get("props"), dict) else {}
            raw["props"].update(props)
            if "points" in fields and "yaws" not in props:
                raw["props"].pop("yaws", None)   # new shape, owner derives them
        else:
            raw = msg.get("object") if isinstance(msg.get("object"), dict) else msg
            before = objects.find_by_id(current, raw.get("id")) if raw.get("id") else None
        after = self._build(raw, actor, before, current)
        return before, objects.to_frame(after, self.datum, "utm")

    def _build(self, raw, actor, before, current):
        """A complete object in the map frame from what the op carried."""
        t = raw.get("type") or (before or {}).get("type")
        fam = objects.family_of_type(t)
        if fam is None:
            raise Refuse("bad_type")
        if before is not None and before["family"] != fam:
            raise Refuse("bad_family")
        name = str(raw.get("name") or "").strip() or (before or {}).get("name") \
            or _auto_name(t, current)
        o = objects._blank(fam, t, name, raw.get("points") or [])
        props = dict(raw.get("props") or {})
        props.pop("effective", None)
        props.pop("programs", None)
        if t == "zone":
            for k, v in objects.ZONE_DEFAULTS.items():
                props.setdefault(k, v)
            try:
                for k in objects.ZONE_INT_PROPS:
                    props[k] = int(props[k])
            except (TypeError, ValueError):
                raise Refuse("bad_props")
        elif t == "slow":
            props.setdefault("speed_mps", objects.DEFAULT_SLOW_MPS)
        elif t == "corridor":
            props.setdefault("width_m", objects.DEFAULT_CORRIDOR_M)
            props.setdefault("connects", [])
        elif t == "wall":
            props.setdefault("width_m", 0.10)
        elif t == "path":
            yaws = props.get("yaws")
            if not isinstance(yaws, list) or len(yaws) != len(o["points"]):
                props.pop("yaws", None)
        o["props"] = props
        try:
            if t == "waypoint":
                o["yaw"] = float(raw.get("yaw") or 0.0)
            elif raw.get("yaw") is not None:
                o["yaw"] = float(raw["yaw"])
        except (TypeError, ValueError):
            raise Refuse("bad_points")
        for k in ("aliases", "tags"):
            o[k] = list(raw.get(k) or [])
        o["note"] = str(raw.get("note") or "")
        o["valid_from"], o["valid_until"] = raw.get("valid_from"), raw.get("valid_until")
        now = self.clock()
        o["id"] = raw.get("id") or (before or {}).get("id") or objects.new_id()
        o["created"] = before["created"] if before else now
        o["updated"] = now
        # An agent cannot make its object look like the owner's.
        src = (before or {}).get("source") or ("agent" if actor == "agent" else None) \
            or raw.get("source") or actor
        o["source"] = src if src in objects.SOURCES else "owner"
        o["protected"] = objects.is_protected(fam, name)
        o["active"] = objects.is_active(o, now)
        v = objects.validate(o, objects=[objects.to_frame(c, self.datum, "map")
                                         for c in current])
        if not v["ok"]:
            raise Refuse(v["reason"])
        self._warnings.extend(v["warnings"])
        return o

    # ---- applying ---------------------------------------------------------
    def _apply(self, before, after):
        """Make the files say `after`; the object as it now is, or None. Either
        may be None; used by ops, undo, redo and rollback alike."""
        family = (after or before)["family"]
        self._applied_before = None
        if family == "place":
            actual = self._apply_place(before, after)
        else:
            actual = self._apply_owner(family, before, after)
            self._set_meta(actual, before) if actual else self._drop_meta(before)
        items = [o for o in (self.baseline.get(family) or [])
                 if before is None or o["name"] != before["name"]]
        self.baseline[family] = items + ([actual] if actual else [])
        return actual

    def _apply_place(self, before, after):
        unread = []
        if os.path.exists(objects.places_path(self.site)):
            items = objects.load_places(self.site, unread)
            if items is None:
                raise Refuse("places_unreadable")   # never overwrite what we cannot read
        else:
            items = []
        gone = (before or after)["id"]
        kept = [o for o in items if o.get("id") != gone]
        if after is not None:
            at = next((i for i, o in enumerate(items) if o.get("id") == gone), len(kept))
            kept.insert(at, after)
        objects.save_places(self.site, kept, self.datum.get("utm_zone"),
                            keep=[b["feature"] for b in unread])
        return after

    def _apply_owner(self, family, before, after):
        if after is None:
            self._await_ready(family)
            if family == "zone":
                before = self._applied_before = self._before_zone_delete(before)
            since = self.clock()
            self.owners.remove(family, before["name"])
            self._wait(family, before["name"], since,
                       lambda: self._read(family, before["name"]) is None)
            return None
        if before is not None and before["name"] != after["name"]:
            self._await_ready(family)
            since = self.clock()
            self.owners.rename(family, before["name"], after["name"])
            self._wait(family, before["name"], since,
                       lambda: self._read(family, after["name"]) is not None
                       and self._read(family, before["name"]) is None,
                       expect=after["name"])
        if before is None or not objects.same_content(before, after):
            self._await_ready(family)
            used = (after.get("props") or {}).get("programs") or []
            if before is None:
                self._check_programs_free(used)
            since = self.clock()
            self.owners.save(family, _owner_view(objects.to_frame(after, self.datum, "map")))

            def landed():
                found = self._read(family, after["name"])
                return found is not None and objects.same_content(found, after)
            self._wait(family, after["name"], since, landed)
            if before is None and used:
                self._rejoin_programs(after["name"], used)
        actual = self._read(family, after["name"])
        for k in objects.META_FIELDS:
            actual[k] = after.get(k)
        return actual

    def _read(self, family, name):
        items = objects.load_family(self.site, family, self.clock(), self.meta) or []
        return next((o for o in items if o["name"] == name), None)

    def _before_zone_delete(self, zone):
        """The planner drops a deleted zone from its programs, so what they held
        goes into the journal with the zone. A running program keeps its zone."""
        table = objects.program_zone_names(self.site)
        out = self._with_memberships(zone, table)
        running = self.owners.running_program()
        if running and running in (table or {}) and zone["name"] in table[running]:
            raise Refuse("zone_in_running_program:" + running)
        return out

    def _check_programs_free(self, used):
        running = self.owners.running_program()
        if running and any(m["name"] == running for m in used):
            raise Refuse("program_running:" + running)

    def _rejoin_programs(self, zone, used):
        """After undo of a delete: the zone back in its programs, in its place.
        Failing that it goes again, half an undo is not an undo."""
        try:
            for m in used:
                table = objects.program_zone_names(self.site)
                if table is None:
                    raise Refuse("programs_unreadable")
                if m["name"] not in table:
                    self._warnings.append("program_gone:" + m["name"])
                    continue
                if zone in table[m["name"]]:
                    continue
                want = objects.zone_back_in_order(m["zone_names"], table[m["name"]], zone)
                since = self.clock()
                self.owners.save_program(m["name"], want, zone)
                self._wait("program", m["name"], since,
                           lambda: (objects.program_zone_names(self.site) or {})
                           .get(m["name"]) == want)
        except Refuse:
            self.owners.remove("zone", zone)
            raise

    def _await_ready(self, family):
        """Right after a site switch an owner still holds the old site, and a save
        then lands in the wrong place, so wait for it to say it has the new one."""
        t0 = self.clock()
        while True:
            reason = self.owners.not_ready(family, self.site)
            if not reason:
                return
            if self.clock() - t0 >= self.ready_s:
                raise Refuse(reason)
            self.sleep(0.1)

    def _wait(self, family, name, since, done, expect=None):
        t0 = self.clock()
        limit = self.confirm_s.get(family, 8.0)
        while True:
            if done():
                return
            reason = self.owners.failure(family, name, since)
            if reason:
                raise Refuse(reason)
            if self.owners.site() != self.site:
                raise Refuse("site_changed")
            if self.clock() - t0 >= limit:
                # It may still land. Then the watcher knows whose it was.
                self.late[(family, expect or name)] = {"actor": self._actor,
                                                       "until": self.clock() + 60.0}
                raise Refuse("owner_timeout")
            self.sleep(0.05)

    # ---- output -----------------------------------------------------------
    def publish_all(self):
        """Each topic on its own: rules first, and a failure in one does not stop
        the rest, a publisher that says nothing looks like a robot without rules."""
        site = None if self.site is _UNSET else self.site
        now = self.clock()
        for step in (self._publish_rules, self._publish_objects, self._publish_history,
                     self._publish_suggestions):
            try:
                step(site, now)
            except Exception as e:
                self.log("%s failed: %r" % (step.__name__, e))

    def _publish_objects(self, site, now):
        usable = site is not None and self.datum is not None
        objs = self._current_map() if usable else []
        for o in objs:
            o["active"] = objects.is_active(o, now)
        key = (site, json.dumps(objs, sort_keys=True))
        if key != self._last_objects:
            self._last_objects = key
            self._pub_seq += 1
            self.emit("objects", {"site": site, "seq": self._pub_seq, "objects": objs})

    def _publish_rules(self, site, now):
        rules, problems = self._rules_now(site, now)
        key = (site, objects.rules_signature(rules), json.dumps(problems, sort_keys=True))
        if key != self._last_rules:
            self._last_rules = key
            self._pub_seq += 1
            for p in problems:
                self.log("bad rule %s '%s': %s" % (p["id"], p["name"], p["reason"]))
            self.emit("rules", {"site": site, "seq": self._pub_seq, "stamp": now,
                                "rules": rules, "problems": problems})

    def _publish_history(self, site, now):
        last = self.entries[-1]["seq"] if self.entries else 0
        key = (site, last, bool(self.undo), bool(self.redo), self.datum is None)
        if key != self._last_history:
            self._last_history = key
            self.emit("history", {"seq": last, "can_undo": bool(self.undo),
                                  "can_redo": bool(self.redo),
                                  "entries": [self._entry_map(e) for e
                                              in self.entries[-HISTORY_ENTRIES:]]
                                  if self.datum else []})

    def _rules_now(self, site, now):
        """No information (file or datum gone for a moment) keeps the site's last
        rules: it is no reason to let the robot into a nogo."""
        problems = []
        rules = objects.load_rules(site, now, problems) if site is not None else []
        if rules is None:
            kept = self._site_rules[0] == site
            rules, problems = (self._site_rules[1:] if kept else ([], []))
        self._site_rules = (site, rules, problems)
        return rules, problems

    def _publish_suggestions(self, site, now):
        if suggestions is None:
            return
        mtime = suggestions.mtime(site) if site else 0.0
        key = (site, mtime, self.datum is None)
        if key != self._last_suggestions:
            self._last_suggestions = key
            if site:
                self.emit("suggestions", suggestions.public(site, self.datum))
            else:
                self.emit("suggestions", {"site": None, "seq": 0, "stamp": self.clock(),
                                          "suggestions": []})

    def _entry_map(self, e):
        out = dict(e)
        for k in ("before", "after"):
            if out[k] is not None:
                out[k] = objects.to_frame(out[k], self.datum, "map")
        return out


def _plain(obj):
    """An object as the journal keeps it: what `active` was then means nothing."""
    if obj is None:
        return None
    o = dict(obj)
    o.pop("active", None)
    return o


def _unchanged(before, after):
    if before["name"] != after["name"] or not objects.same_content(before, after):
        return False
    return all(before.get(k) == after.get(k) for k in ("aliases", "note", "tags", "source"))


def _owner_view(obj):
    """What goes to an owner: its props without the computed ones."""
    o = dict(obj)
    o["props"] = {k: v for k, v in obj["props"].items()
                  if k not in objects.COMPUTED_PROPS.get(obj["family"], ())}
    return o


def _auto_name(obj_type, current):
    used = {str(o.get("name")).casefold() for o in current}
    n = 1
    while ("%s_%d" % (obj_type, n)).casefold() in used:
        n += 1
    return "%s_%d" % (obj_type, n)
