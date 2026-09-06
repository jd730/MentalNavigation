import numpy as np
from fragmentedmaps.sensor import BaseSensor

def discretize_path(path, start, end, grid_size):
    """
    Discretize an arbitrary path such that each step is in
    one of the four cardinal directions

    Inputs:

    """
    path_locations = path.x[start:end, ]
    path_locations = np.round(path_locations/grid_size)

    path_locations_discretized = [path_locations[0]]
    for i in range(1, len(path_locations)):
        x1, y1 = path_locations[i-1]
        x2, y2 = path_locations[i]
        if x1 != x2:
            intermediate_x = np.sign(x2-x1) * np.arange(np.abs(x2-x1)+1)[1:] + x1
            path_locations_discretized.extend([(x, y1) for x in intermediate_x])
        if y1 != y2:
            intermediate_y = np.sign(y2-y1) * np.arange(np.abs(y2-y1)+1)[1:] + y1
            path_locations_discretized.extend([(x2, y) for y in intermediate_y])
    path_locations_discretized = np.array(path_locations_discretized)
    return path_locations_discretized


# all angles should be in degrees, in [-180, 180)
# input can be int/float or np.array
def standardize_angle(angle):
    angle = angle % 360
    if isinstance(angle, np.ndarray):
        angle[angle>=180] = angle[angle>=180] - 360
    else:
        if angle >= 180:
            angle -= 360
    return angle

def compute_hd(path_locations):
    displacement = path_locations[1:] - path_locations[:-1]
    hd = np.arctan2(displacement[:, 1], displacement[:, 0])
    return standardize_angle(hd / np.pi * 180)

# path_locations input: dim (Npatts, 2)
# sbook output: dim (Ns, Npatts)

# hds: list of hd values
def get_sbook_egocentric(env, path_locations, hds, num_rays, fov, r):
    n = len(path_locations)
    if fov == 360:
        num_rays += 1
    sbook = np.zeros((num_rays, n))
    sensor = BaseSensor(r=r, fov=fov, num_rays=num_rays)
    for i in range(n):
        loc = path_locations[i]
        sbook[:, i] = sensor(loc, hd=hds[i] / 180 * np.pi, segs=env['segs'])[0]
    if np.sum(np.isinf(sbook)) > 0:
        print('Warning! Inf values found')
    if fov == 360:
        sbook = sbook[:-1]
    return sbook

# hds: list of hd values
# num_rays is total number of possible rays, not the number of rays observed at any given moment
def select_angle_in_range(hd, fov, ray_angles):
    start_angle = standardize_angle(hd - fov/2)
    end_angle = standardize_angle(hd + fov/2)
    start = np.where(ray_angles >= start_angle)[0][0]
    end = np.where(ray_angles <= end_angle)[0][-1]
    return start, end


def get_sbook_allocentric(env, path_locations, hds, num_rays, fov, r,
    precomputed_sbook=None):
    sensor = BaseSensor(r=r, fov=360, num_rays=num_rays+1)
    ray_angles = np.linspace(-180, 180, num_rays, endpoint=False)
    sbook = np.zeros((num_rays, len(path_locations)))
    for i, loc in enumerate(path_locations):
        if precomputed_sbook is None:
            s_original = sensor(loc, hd=0, segs=env['segs'])[0][:-1]
        else:
            s_original = precomputed_sbook[:, i]
        start, end = select_angle_in_range(hds[i], fov, ray_angles)
        s = np.empty(num_rays)
        s[:] = np.nan
        if start <= end:
            s[start:end+1] = s_original[start:end+1]
        else:
            s[start:] = s_original[start:]
            s[:end+1] = s_original[:end+1]
        sbook[:, i] = s
    if np.sum(np.isinf(sbook)) > 0:
        print('Warning! Inf values found')
    return sbook

# hd: int/float
def get_sbook_fixed_hd(env, path_locations, hd, num_rays, fov, r):
    hd = [hd for i in range(len(path_locations))]
    return get_sbook_egocentric(env, path_locations, hd, num_rays, fov, r)