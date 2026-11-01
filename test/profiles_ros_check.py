#!/usr/bin/env python3
"""Integration check on a PRIVATE roscore: real mapping_manager + real
depth_obstacles + real direct_raster (started by the manager), a synthetic depth
camera. Never against the robot's master: run through profiles_ros_check.sh.

Scene (base_link, camera at the mount): a near box at 2.0 m, a far box at 3.5 m,
both 0.3..0.7 m high, seen in a synthetic depth image.
"""
import json
import os
import sys
import time

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf.transformations import quaternion_from_matrix

assert os.environ['ROS_MASTER_URI'].endswith(':11890')
HOME = os.environ['HOME']
assert 'jobs' in HOME
BASE = os.path.join(HOME, '.vitulus/mapping_v3')
W, H, F = 640, 480, 500.0
results = []


def check(name, ok, detail=''):
    results.append(bool(ok))
    print('%s %s %s' % ('PASS' if ok else 'FAIL', name, detail), flush=True)


class T:
    state = None
    cfg = None
    grid = None
    direct = None


def on_state(m): T.state = json.loads(m.data)
def on_cfg(m): T.cfg = json.loads(m.data)
def on_direct(m): T.direct = json.loads(m.data)
def on_grid(m): T.grid = m


def wide():
    return [p for p in T.cfg['profiles'] if p['name'] == 'wide'][0]


def nav():
    return [p for p in T.cfg['profiles'] if p['name'] == 'nav'][0]


def wait(fn, t=12.0, what=''):
    end = time.time() + t
    while time.time() < end and not rospy.is_shutdown():
        try:
            if fn():
                return True
        except (TypeError, KeyError, IndexError, AttributeError):
            pass
        time.sleep(0.05)
    print('  timeout: %s' % what, flush=True)
    return False


def depth_image():
    img = np.zeros((H, W), np.uint16)
    cam_h, cam_x = 0.29, 0.21
    for dist, u0, u1 in ((2.0, 200, 300), (3.5, 360, 420)):
        z = dist - cam_x
        v_top = int(H / 2 - (0.7 - cam_h) * F / z)
        v_bot = int(H / 2 + (cam_h - 0.3) * F / z)
        img[max(v_top, 0):min(v_bot, H), u0:u1] = int(z * 1000)
    for v in range(H // 2 + 20, H):         # the ground, so a plane exists
        z = cam_h * F / (v - H / 2)
        if z < 3.9:
            row = img[v]
            row[row == 0] = int(z * 1000)
    return img


def occupied_at(x_lo, x_hi):
    g = T.grid
    if g is None:
        return 0
    d = np.array(g.data, np.int8).reshape(g.info.height, g.info.width)
    ys, xs = np.nonzero(d == 100)
    wx = g.info.origin.position.x + (xs + 0.5) * g.info.resolution
    return int(((wx >= x_lo) & (wx <= x_hi)).sum())


def cmd(pub, **kw):
    kw['id'] = 't%f' % time.time()
    pub.publish(String(json.dumps(kw)))
    return wait(lambda: T.state['last']['id'] == kw['id'], what=kw['cmd'])


def main():
    rospy.init_node('profiles_check')
    rospy.Subscriber('/mapping_manager/profile_state', String, on_state)
    rospy.Subscriber('/depth_obstacles/config', String, on_cfg)
    rospy.Subscriber('/mapping/direct_status', String, on_direct)
    rospy.Subscriber('/mapping/direct_map', OccupancyGrid, on_grid, queue_size=1)
    pc = rospy.Publisher('/mapping_manager/profile_cmd', String, queue_size=4)
    ps = rospy.Publisher('/mapping_manager/serve_site', String, queue_size=2)
    pst = rospy.Publisher('/mapping_manager/start', String, queue_size=2)
    pstop = rospy.Publisher('/mapping_manager/stop', String, queue_size=2)
    pd = rospy.Publisher('/depth_obstacles/set', String, queue_size=4)
    pg = rospy.Publisher('/mapping/gate_mode', String, queue_size=1, latch=True)
    pdir = rospy.Publisher('/mapping/set_direct', String, queue_size=2)
    pi = rospy.Publisher('/cam/depth', Image, queue_size=1)
    pk = rospy.Publisher('/cam/info', CameraInfo, queue_size=1)

    br = tf2_ros.StaticTransformBroadcaster()

    def tf(parent, child, xyz, R=np.eye(3)):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id, t.child_frame_id = parent, child
        tr = t.transform.translation
        tr.x, tr.y, tr.z = xyz
        M = np.eye(4)
        M[:3, :3] = R
        q = quaternion_from_matrix(M)
        r = t.transform.rotation
        r.x, r.y, r.z, r.w = q
        return t
    opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    br.sendTransform([tf('map', 'odom', (0, 0, 0)), tf('odom', 'base_link', (0, 0, 0)),
                      tf('base_link', 'cam_opt', (0.21, 0.0, 0.29), opt)])

    img = depth_image()
    im = Image(height=H, width=W, encoding='16UC1', is_bigendian=0, step=W * 2, data=img.tobytes())
    im.header.frame_id = 'cam_opt'
    ci = CameraInfo(height=H, width=W, distortion_model='plumb_bob', D=[0.0] * 5)
    ci.K = [F, 0, W / 2, 0, F, H / 2, 0, 0, 1]
    ci.P = [F, 0, W / 2, 0, 0, F, H / 2, 0, 0, 0, 1, 0]
    ci.header.frame_id = 'cam_opt'

    def tick(_e):
        im.header.stamp = ci.header.stamp = rospy.Time.now()
        pi.publish(im)
        pk.publish(ci)
    rospy.Timer(rospy.Duration(0.15), tick)

    check('state published, cfg seen', wait(lambda: T.state and T.cfg, 25, 'state+cfg'))
    print('  first state:', {k: T.state[k] for k in ('site', 'active', 'modified', 'camera_sync')}, flush=True)

    # ---- migration + site switch: the camera chain follows the site
    ps.publish(String('siteA'))
    check('A served, profile migrated', wait(lambda: T.state['site'] == 'siteA' and T.state['active'] == 'Lawn tall', 12, 'site A'))
    check('A keeps its effective lidar', T.state['values']['mapper']['lidar_min_intensity'] == 0.0)
    check('wide has A values', wait(lambda: wide()['values']['min_obstacle_height'] == 0.25 and wide()['values']['range_max'] == 2.8, 8, 'wide A'),
          str({k: wide()['values'][k] for k in ('range_max', 'min_obstacle_height', 'max_obstacle_height')}))
    check('camera sync ok', wait(lambda: T.state['camera_sync']['state'] == 'ok'))
    nav0 = json.dumps(nav()['values'], sort_keys=True)
    ps.publish(String('siteB'))
    check('B served, its profile', wait(lambda: T.state['site'] == 'siteB' and T.state['active'] == 'Paved yard', 12, 'site B'))
    check('wide has B values', wait(lambda: wide()['values']['min_obstacle_height'] == -1.0 and wide()['values']['max_obstacle_height'] == 1.5, 8, 'wide B'))
    check('min_cluster back to default on switch', wide()['values']['min_cluster_size'] == 2)

    # ---- Camera card edit (the UI's message on the node's own topic) joins the site
    pd.publish(String(json.dumps({'cmd': 'set', 'id': 'ui-1', 'profile': 'wide', 'values': {'range_max': 3.2}})))
    check('card edit adopted', wait(lambda: T.state['overrides']['camera'].get('range_max') == 3.2 and T.state['modified'] == 1, 6, 'adopt'))
    ps.publish(String('siteA'))
    check('back on A: its value', wait(lambda: T.state['site'] == 'siteA' and wide()['values']['range_max'] == 2.8, 12, 'A again'))
    check('A not modified by B', T.state['overrides']['camera'] == {})
    ps.publish(String('siteB'))
    check('B again: edit restored', wait(lambda: T.state['site'] == 'siteB' and wide()['values']['range_max'] == 3.2, 12, 'B again'))
    check('nav untouched by all of this', json.dumps(nav()['values'], sort_keys=True) == nav0)
    check('store keeps nav', json.load(open(HOME + '/.vitulus/depth_obstacles/settings.json'))['profiles'].get('nav', {}).get('max_depth') == 3.05)

    # ---- commands without a session
    check('set without session', cmd(pc, cmd='set', mapper={'lidar_max_range_m': 6.5}) and T.state['last']['ok'])
    check('kept in the site file', json.load(open(BASE + '/siteB/mapping_profile.json'))['mapper'].get('lidar_max_range_m') == 6.5)
    check('direct_overrides written', json.load(open(BASE + '/siteB/direct_overrides.json'))['lidar_max_range_m'] == 6.5)
    cmd(pc, cmd='set', mapper={'min_hits': 0})
    check('refused with a reason', T.state['last']['ok'] is False and 'min_hits' in T.state['last']['error'], T.state['last']['error'])
    cmd(pc, cmd='save', name='Zkouska')
    check('save as', T.state['active'] == 'Zkouska' and T.state['modified'] == 0)
    check('wide stays', wide()['values']['range_max'] == 3.2)
    cmd(pc, cmd='select', name='Lawn short')
    check('select switches camera', wait(lambda: wide()['values']['max_obstacle_height'] == 0.6 and wide()['values']['range_max'] == 2.8, 6, 'select'))
    cmd(pc, cmd='delete', name='Zkouska')
    check('delete', all(p['name'] != 'Zkouska' for p in T.state['profiles']))
    pd.publish(String(json.dumps({'cmd': 'set', 'id': 'ui-bad', 'profile': 'wide', 'values': {'range_max': 99}})))
    time.sleep(0.5)
    check('a refused card value is not adopted', 'range_max' not in T.state['overrides']['camera'])

    # ---- what is in the card is what lands in the map
    pst.publish(String('siteC'))
    check('recording, new site copies the state', wait(lambda: T.state['site'] == 'siteC' and T.state['recording'] and T.state['active'] == 'Lawn short', 15, 'start'))
    check('direct_raster up', wait(lambda: T.direct is not None and 'lidar_max_range_m' in T.direct['settings'], 30, 'direct'))
    check('no camera filter in status', 'cam_obstacle_min_h' not in T.direct['settings'] and 'presets' not in T.direct)
    pg.publish(String('force_on'))
    check('near box mapped at reach 2.8', wait(lambda: occupied_at(1.7, 2.3) > 0, 40, 'near'), 'near cells=%d' % occupied_at(1.7, 2.3))
    time.sleep(3)
    check('far box NOT mapped at reach 2.8', occupied_at(3.2, 3.8) == 0, 'far=%d' % occupied_at(3.2, 3.8))
    pd.publish(String(json.dumps({'cmd': 'set', 'id': 'ui-2', 'profile': 'wide', 'values': {'range_max': 3.9}})))
    check('card 3.9 m: far box appears, nothing filters it again', wait(lambda: occupied_at(3.2, 3.8) > 0, 25, 'far'), 'far=%d' % occupied_at(3.2, 3.8))
    check("edit is the site's", T.state['overrides']['camera'].get('range_max') == 3.9)
    pc.publish(String(json.dumps({'id': 'x', 'cmd': 'set', 'mapper': {'min_hits': 6}})))
    check('live forward to direct_raster', wait(lambda: T.direct['settings']['min_hits'] == 6, 6, 'forward'))
    pdir.publish(String(json.dumps({'min_hits': 7})))
    check('raw set_direct joins the site', wait(lambda: T.state['overrides']['mapper'].get('min_hits') == 7, 6, 'raw'))
    pstop.publish(String('stop'))
    check('stopped', wait(lambda: not T.state['recording'], 40, 'stop'))
    print('RESULT %d/%d' % (sum(results), len(results)), flush=True)
    sys.exit(0 if all(results) else 1)


main()
