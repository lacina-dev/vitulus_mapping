"""DemGrid -> grid_map_msgs/GridMap (visualize with the grid_map rviz plugin).

grid_map conventions (verified against grid_map_ros GridMapRosConverter +
GridMapMsgHelpers): info.pose = grid CENTER; matrix rows follow x, cols follow
y, index (0,0) is the corner with MAX x, MAX y; data serialized column-major
with dim[0] = 'column_index'.
"""

import numpy as np
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from .demgrid import SRC_NONE


def _layer_msg(mat):
    """mat: grid_map-oriented matrix (rows = x desc, cols = y desc)."""
    m = Float32MultiArray()
    rows, cols = mat.shape
    m.layout.dim = [
        MultiArrayDimension(label='column_index', size=cols, stride=rows * cols),
        MultiArrayDimension(label='row_index', size=rows, stride=rows),
    ]
    m.data = mat.flatten(order='F').tolist()
    return m


def dem_to_gridmap_msg(dem, frame_id, stamp, crop_margin_cells=4):
    """Publishes only the valid bounding box (the full grown array can be
    huge). Layers: ground_elevation, confidence, last_seen_age, source."""
    bbox = dem.valid_bbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - crop_margin_cells)
    y0 = max(0, y0 - crop_margin_cells)
    x1 = min(dem.shape[0] - 1, x1 + crop_margin_cells)
    y1 = min(dem.shape[1] - 1, y1 + crop_margin_cells)
    sl = (slice(x0, x1 + 1), slice(y0, y1 + 1))
    nx = x1 - x0 + 1
    ny = y1 - y0 + 1

    elev = dem.elev[sl].astype(np.float32)
    conf = np.where(dem.source[sl] == SRC_NONE, np.nan,
                    np.minimum(dem.weight[sl], 50.0) / 50.0).astype(np.float32)
    age = np.where(dem.last_seen[sl] > 0,
                   stamp.to_sec() - dem.last_seen[sl], np.nan).astype(np.float32)
    src = np.where(dem.source[sl] == SRC_NONE, np.nan,
                   dem.source[sl]).astype(np.float32)

    msg = GridMap()
    msg.info.header.frame_id = frame_id
    msg.info.header.stamp = stamp
    msg.info.resolution = dem.res
    msg.info.length_x = nx * dem.res
    msg.info.length_y = ny * dem.res
    msg.info.pose.position.x = dem.ox + (x0 + nx / 2.0) * dem.res
    msg.info.pose.position.y = dem.oy + (y0 + ny / 2.0) * dem.res
    msg.info.pose.orientation.w = 1.0
    msg.layers = ['ground_elevation', 'confidence', 'last_seen_age', 'source']
    msg.basic_layers = ['ground_elevation']
    for a in (elev, conf, age, src):
        msg.data.append(_layer_msg(a[::-1, ::-1]))   # natural -> grid_map axes
    msg.outer_start_index = 0
    msg.inner_start_index = 0
    return msg
