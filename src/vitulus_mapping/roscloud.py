"""PointCloud2 -> numpy without ros_numpy (not installed on the robot)."""

import numpy as np


def pointcloud2_to_xyz(msg, stride=1):
    """Returns Nx3 float32 (finite points only). stride > 1 subsamples."""
    if msg.is_bigendian:
        raise ValueError('big-endian PointCloud2 not supported')
    offsets = {}
    for f in msg.fields:
        if f.name in ('x', 'y', 'z'):
            if f.datatype != 7:  # sensor_msgs/PointField.FLOAT32
                raise ValueError('field %s is not FLOAT32' % f.name)
            offsets[f.name] = f.offset
    if len(offsets) != 3:
        raise ValueError('cloud lacks x/y/z fields')

    dt = np.dtype({'names': ['x', 'y', 'z'],
                   'formats': ['<f4', '<f4', '<f4'],
                   'offsets': [offsets['x'], offsets['y'], offsets['z']],
                   'itemsize': msg.point_step})
    n = msg.width * msg.height
    arr = np.frombuffer(msg.data, dtype=dt, count=n)
    if stride > 1:
        arr = arr[::stride]
    xyz = np.column_stack((arr['x'], arr['y'], arr['z']))
    return xyz[np.isfinite(xyz).all(axis=1)]
