"""Names of map objects (edits, waypoints, paths, zones).

`name` is the only key of a map object, so a rename must be refused rather than
collide. The owner nodes (mapping_manager, navi_man, node_planner) share these
rules; results go out on RENAME_RESULT_TOPIC as
{"kind", "from", "to", "ok", "code", "message"}.
"""

import json

RENAME_RESULT_TOPIC = '/map_objects/rename_result'

MAX_LEN = 60
# Fixed meaning elsewhere: dock flows, the dock seed.
RESERVED = ('DOCK', 'UNDOCK', 'docked')
_RESERVED_FOLDED = {n.casefold() for n in RESERVED}


def parse_request(data):
    """'{"from":..,"to":..}' -> (from, to), stripped. ValueError if malformed."""
    req = json.loads(data)
    if not isinstance(req, dict):
        raise ValueError('expected a JSON object')
    return str(req.get('from', '')).strip(), str(req.get('to', '')).strip()


def check_name(name, existing, current=None):
    """Why `name` cannot be used, as a code ('empty', 'too_long', 'bad_chars',
    'reserved', 'exists'), or None if it can. `existing` = names of the same kind;
    `current` = the object's present name, so only its case may change."""
    if not name:
        return 'empty'
    if len(name) > MAX_LEN:
        return 'too_long'
    # '*' = "all" in remove_edit, '|||' joins the legacy lists.
    if name == '*' or '|||' in name or any(ord(c) < 32 or ord(c) == 127 for c in name):
        return 'bad_chars'
    folded = name.casefold()
    if folded in _RESERVED_FOLDED:
        return 'reserved'
    skip = current.casefold() if current is not None else None
    if any(n.casefold() == folded and n.casefold() != skip for n in existing):
        return 'exists'
    return None


def check_rename(old, new, existing):
    """Code why `old` cannot become `new`, or None. 'same' = nothing to do."""
    if old not in existing:
        return 'not_found'
    if old.casefold() in _RESERVED_FOLDED:
        return 'reserved'
    if old == new:
        return 'same'
    return check_name(new, existing, current=old)


def message_for(code, old, new):
    return {
        'ok': "'{}' renamed to '{}'.".format(old, new),
        'same': "'{}' keeps its name.".format(old),
        'bad_request': 'Bad rename request.',
        'not_found': "'{}' does not exist.".format(old),
        'empty': 'The name is empty.',
        'too_long': 'The name is longer than {} characters.'.format(MAX_LEN),
        'bad_chars': "The name '{}' contains characters that cannot be used.".format(new),
        'reserved': "DOCK, UNDOCK and docked are reserved names.",
        'exists': "The name '{}' is already used.".format(new),
        'busy': "'{}' is in use right now — stop it first.".format(old),
        'no_site': 'No map is active.',
        'failed': "'{}' could not be renamed.".format(old),
    }.get(code, code)


def result_json(kind, old, new, code):
    """The rename result payload. 'same' counts as success."""
    return json.dumps({'kind': kind, 'from': old, 'to': new,
                       'ok': code in ('ok', 'same'), 'code': code,
                       'message': message_for(code, old, new)})
