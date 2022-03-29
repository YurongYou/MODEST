import types

import numpy as np
import sklearn
import torch
from sklearn.linear_model import RANSACRegressor

from utils.iou3d_nms import iou3d_nms_utils
from utils import kitti_util

def cart2hom(pts_3d):
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1), dtype=np.float32)))
    return pts_3d_hom


def transform_points(pts_3d_ref, Tr):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(Tr)).reshape(-1, 4)[:, 0:3]


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def load_plane(plane_filename):
    with open(plane_filename, 'r') as f:
        lines = f.readlines()
    lines = [float(i) for i in lines[3].split()]

    plane = np.asarray(lines)

    # Ensure normal is always facing up, this is in the rectified camera coordinate
    if plane[1] > 0:
        plane = -plane

    norm = np.linalg.norm(plane[0:3])
    plane = plane / norm
    return plane


def estimate_plane(origin_ptc, max_hs=-1.5, it=1, ptc_range=((-20, 70), (-20, 20))):
    mask = (origin_ptc[:, 2] < max_hs) & \
        (origin_ptc[:, 0] > ptc_range[0][0]) & \
        (origin_ptc[:, 0] < ptc_range[0][1]) & \
        (origin_ptc[:, 1] > ptc_range[1][0]) & \
        (origin_ptc[:, 1] < ptc_range[1][1])
    for _ in range(it):
        ptc = origin_ptc[mask]
        reg = RANSACRegressor().fit(ptc[:, [0, 1]], ptc[:, 2])
        w = np.zeros(3)
        w[0] = reg.estimator_.coef_[0]
        w[1] = reg.estimator_.coef_[1]
        w[2] = -1.0
        h = reg.estimator_.intercept_
        norm = np.linalg.norm(w)
        w /= norm
        h = h / norm
        result = np.array((w[0], w[1], w[2], h))
        result *= -1
        mask = np.logical_not(above_plane(
            origin_ptc[:, :3], result, offset=0.2))
    return result


def above_plane(ptc, plane, offset=0.05, only_range=((-30, 30), (-30, 30))):
    mask = distance_to_plane(ptc, plane, directional=True) < offset
    if only_range is not None:
        range_mask = (ptc[:, 0] < only_range[0][1]) * (ptc[:, 0] > only_range[0][0]) * \
            (ptc[:, 1] < only_range[1][1]) * (ptc[:, 1] > only_range[1][0])
        mask *= range_mask
    return np.logical_not(mask)

def distance_to_plane(ptc, plane, directional=False):
    d = ptc @ plane[:3] + plane[3]
    if not directional:
        d = np.abs(d)
    d /= np.sqrt((plane[:3]**2).sum())
    return d


import numpy as np
from scipy.spatial import ConvexHull


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval, angles[best_idx], areas[best_idx]

def PCA_rectangle(cluster_ptc):
    components = sklearn.decomposition.PCA(
        n_components=2).fit(cluster_ptc).components_
    on_component_ptc = cluster_ptc @ components.T
    min_x, max_x = on_component_ptc[:, 0].min(), on_component_ptc[:, 0].max()
    min_y, max_y = on_component_ptc[:, 1].min(), on_component_ptc[:, 1].max()
    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    angle = np.arctan2(components[0, 1], components[0, 0])
    return rval, angle, area

def closeness_rectangle(cluster_ptc, delta=0.1, d0=1e-2):
    max_beta = -float('inf')
    choose_angle = None
    for angle in np.arange(0, 90+delta, delta):
        angle = angle / 180. * np.pi
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:,0].min(), projection[:,0].max()
        min_y, max_y = projection[:,1].min(), projection[:,1].max()
        Dx = np.vstack((projection[:, 0] - min_x, max_x - projection[:, 0])).min(axis=0)
        Dy = np.vstack((projection[:, 1] - min_y, max_y - projection[:, 1])).min(axis=0)
        beta = np.vstack((Dx, Dy)).min(axis=0)
        beta = np.maximum(beta, d0)
        beta = 1 / beta
        beta = beta.sum()
        if beta > max_beta:
            max_beta = beta
            choose_angle = angle
    angle = choose_angle
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = cluster_ptc @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    if (max_x - min_x) < (max_y - min_y):
        angle = choose_angle + np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    return rval, angle, area


def variance_rectangle(cluster_ptc, delta=0.1):
    max_var = -float('inf')
    choose_angle = None
    for angle in np.arange(0, 90+delta, delta):
        angle = angle / 180. * np.pi
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
        Dx = np.vstack((projection[:, 0] - min_x,
                       max_x - projection[:, 0])).min(axis=0)
        Dy = np.vstack((projection[:, 1] - min_y,
                       max_y - projection[:, 1])).min(axis=0)
        Ex = Dx[Dx < Dy]
        Ey = Dy[Dy < Dx]
        var = 0
        if (Dx < Dy).sum() > 0:
            var += -np.var(Ex)
        if (Dy < Dx).sum() > 0:
            var += -np.var(Ey)
        # print(angle, var)
        if var > max_var:
            max_var = var
            choose_angle = angle
    # print(choose_angle, max_var)
    angle = choose_angle
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = cluster_ptc @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    if (max_x - min_x) < (max_y - min_y):
        angle = choose_angle + np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    return rval, angle, area


def get_lowest_point_rect(ptc, xz_center, l, w, ry):
    ptc_xz = ptc[:, [0, 2]] - xz_center
    rot = np.array([
        [np.cos(ry), -np.sin(ry)],
        [np.sin(ry), np.cos(ry)]
    ])
    ptc_xz = ptc_xz @ rot.T
    mask = (ptc_xz[:, 0] > -l/2) & \
        (ptc_xz[:, 0] < l/2) & \
        (ptc_xz[:, 1] > -w/2) & \
        (ptc_xz[:, 1] < w/2)
    ys = ptc[mask, 1]
    return ys.max()

def get_obj(ptc, full_ptc, fit_method='min_zx_area_fit'):
    if fit_method == 'min_zx_area_fit':
        corners, ry, area = minimum_bounding_rectangle(ptc[:, [0, 2]])
    elif fit_method == 'PCA':
        corners, ry, area = PCA_rectangle(ptc[:, [0, 2]])
    elif fit_method == 'variance_to_edge':
        corners, ry, area = variance_rectangle(ptc[:, [0, 2]])
    elif fit_method == 'closeness_to_edge':
        corners, ry, area = closeness_rectangle(ptc[:, [0, 2]])
    else:
        raise NotImplementedError(fit_method)
    ry *= -1
    l = np.linalg.norm(corners[0] - corners[1])
    w = np.linalg.norm(corners[0] - corners[-1])
    c = (corners[0] + corners[2]) / 2
    # bottom = ptc[:, 1].max()
    bottom = get_lowest_point_rect(full_ptc, c, l, w, ry)
    h = bottom - ptc[:, 1].min()
    obj = types.SimpleNamespace()
    obj.t = np.array([c[0], bottom, c[1]])
    obj.l = l
    obj.w = w
    obj.h = h
    obj.ry = ry
    obj.volume = area * h
    return obj


def objs_nms(objs, use_score_rank=False, nms_threshold=0.1):
    # generate box array
    boxes = np.array(
        [[obj.t[0], obj.t[2], 0, obj.l, obj.w, obj.h, -obj.ry] for obj in objs])
    boxes = torch.from_numpy(boxes).float().cuda()

    overlaps_bev = iou3d_nms_utils.boxes_iou_bev(
        boxes.contiguous(), boxes.contiguous())

    overlaps_bev = overlaps_bev.cpu().numpy()
    mask = np.ones(overlaps_bev.shape[0], dtype=bool)
    if use_score_rank:
        scores = [obj.score for obj in objs]
        order = np.argsort(scores)[::-1]
    else:
        bbox_area = np.diag(overlaps_bev)
        order = bbox_area.argsort()[::-1]
    for idx in order:
        if not mask[idx]:
            continue
        mask[overlaps_bev[idx] > nms_threshold] = False
        mask[idx] = True

    objs_nmsed = [objs[i] for i in range(len(objs)) if mask[i]]
    return objs_nmsed


def objs2label(objs, calib, obj_type="Dynamic", with_score=False):
    label_strings = []
    for obj in objs:
        alpha = -np.arctan2(obj.t[0], obj.t[2]) + obj.ry
        corners_2d = kitti_util.compute_box_3d(obj, calib.P)[0]
        min_uv = np.min(corners_2d, axis=0)
        max_uv = np.max(corners_2d, axis=0)
        boxes2d_image = np.concatenate([min_uv, max_uv], axis=0)
        score = -1
        if hasattr(obj, 'score'):
            score = obj.score
        if with_score:
            label_strings.append(
                f"{obj_type} -1 -1 {alpha:.4f} "
                f"{boxes2d_image[0]:.4f} {boxes2d_image[1]:.4f} {boxes2d_image[2]:.4f} {boxes2d_image[3]:.4f} "
                f"{obj.h:.4f} {obj.w:.4f} {obj.l:.4f} "
                f"{obj.t[0]:.4f} {obj.t[1]:.4f} {obj.t[2]:.4f} {obj.ry:.4f} {score:.4f}")
        else:
            label_strings.append(
                f"{obj_type} -1 -1 {alpha:.4f} "
                f"{boxes2d_image[0]:.4f} {boxes2d_image[1]:.4f} {boxes2d_image[2]:.4f} {boxes2d_image[3]:.4f} "
                f"{obj.h:.4f} {obj.w:.4f} {obj.l:.4f} "
                f"{obj.t[0]:.4f} {obj.t[1]:.4f} {obj.t[2]:.4f} {obj.ry:.4f}")
    return "\n".join(label_strings)


def is_within_fov(obj, calib, image_shape):
    center = obj.t.copy()
    center[1] -= obj.h / 2
    uv = calib.project_rect_to_image(center.reshape(1, -1)).squeeze()
    return uv[0] < image_shape[1] and uv[0] >= 0 and \
        uv[1] < image_shape[0] and uv[1] >= 0 and \
        center[2] > 0
