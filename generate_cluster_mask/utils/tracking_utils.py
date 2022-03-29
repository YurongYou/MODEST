import types
import os.path as osp

from pandas.core import frame

import numpy as np
import sklearn
from sklearn import cluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R

from utils import kitti_util
from pre_compute_pp_score import get_relative_pose
from utils.pointcloud_utils import load_velo_scan, get_obj, transform_points
import open3d as o3d
from pyquaternion import Quaternion

_KITTI2NU_lyft = Quaternion(axis=(0, 0, 1), angle=np.pi).transformation_matrix
ASSOCIATION_THRESHOLD = 3.5
MIN_POINTS_THRESH = 0


class Track(object):
    def __init__(self, bbox, start_frame):
        self.bboxes = [bbox]
        self.start_id = start_frame
        self.stale_count = 0
        self.terminated = False
        self.last_prev_to_curr_pose = None
    
    def get_interpolated_position(self, pose_prev_to_curr, calib):
        # return t, ry
        if len(self.bboxes) == 1:
            # compute last bbox in current pose
            new_t_in_prev = self.bboxes[0].t
            new_ry_in_prev = self.bboxes[0].ry
        else:
            # compute 2nd-to-last bbox in last bbox
            bbox_prev_in_last = transform_bbox_rect_with_velo(self.bboxes[-2], self.last_prev_to_curr_pose, calib)

            # compute the delta
            t_delta = self.bboxes[-1].t - bbox_prev_in_last.t
            ry_delta = self.bboxes[-1].ry - bbox_prev_in_last.ry

            new_t_in_prev, new_ry_in_prev = self.bboxes[-1].t + t_delta, self.bboxes[-1].ry + ry_delta 

        # transform into current frame
        new_t = translate_rect_with_velo(new_t_in_prev[np.newaxis, :], pose_prev_to_curr, calib).reshape(3,)
        new_ry = rotate_rect_with_velo(new_ry_in_prev, pose_prev_to_curr, calib)
        # return the translation, rotation
        return new_t, new_ry


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def get_trans_matrix(query_scan, lidars, global_reg=False):
    threshold = 0.5
    voxel_size = 0.5
    pcd_0 = o3d.geometry.PointCloud()
    pcd_0.points = o3d.utility.Vector3dVector(lidars)

    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(query_scan)
    # print(o3d.registration.evaluate_registration(
    #     pcd_0, pcd_1, threshold), np.eye(4))

    if global_reg:
        source_down, source_fpfh = preprocess_point_cloud(pcd_0, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(pcd_1, voxel_size)
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_0, pcd_1, threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    else:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_0, pcd_1, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation


def get_obj_ptc_mask(ptc_rect, obj):
    ry = obj.ry
    l = obj.l
    w = obj.w
    xz_center = obj.t[[0, 2]]
    ptc_xz = ptc_rect[:, [0, 2]] - xz_center
    rot = np.array([
        [np.cos(ry), -np.sin(ry)],
        [np.sin(ry), np.cos(ry)]
    ])
    ptc_xz = ptc_xz @ rot.T
    mask = (ptc_xz[:, 0] > -l/2) & \
        (ptc_xz[:, 0] < l/2) & \
        (ptc_xz[:, 1] > -w/2) & \
        (ptc_xz[:, 1] < w/2)
    y_mask = (ptc_rect[:, 1] > obj.t[1] - obj.h) * (ptc_rect[:, 1] <= obj.t[1])
    mask = mask * y_mask
    return mask


def transform_bbox_rect_with_velo(bbox, pose, calib):
    t_new = translate_rect_with_velo(bbox.t[np.newaxis, :], pose, calib).reshape(3,)
    ry_new = rotate_rect_with_velo(bbox.ry, pose, calib)
    obj = types.SimpleNamespace()
    obj.t = t_new
    obj.l = bbox.l
    obj.w = bbox.w
    obj.h = bbox.h
    obj.ry = ry_new
    obj.volume = bbox.volume
    return obj


def translate_rect_with_velo(rect_pnt, pose, calib):
    pnt_velo = calib.project_rect_to_velo(rect_pnt).reshape(-1, 3)
    pnt_velo = transform_points(pnt_velo, pose)
    pnt_rect = calib.project_velo_to_rect(pnt_velo)
    return pnt_rect


def rotate_rect_with_velo(ry, pose, calib):
    rot = kitti_util.roty(ry)
    rot_velo = calib.C2V[:3, :3] @ np.linalg.inv(calib.R0) @ rot
    rot_velo = pose[:3, :3] @ rot_velo 
    rot_camera = calib.R0 @ calib.V2C[:3, :3] @ rot_velo

    r = R.from_matrix(rot_camera)
    ry_new = r.as_euler('zxy')[-1]
    return ry_new


def process_unassociated_track(track, pose_prev_to_curr, calib, keep_stale_counter=True):
    # TODO: take last transformation in velo space
    # import ipdb; ipdb.set_trace()
    if keep_stale_counter:
        if track.stale_count > 3:
            track.terminated = True
        else:
            track.stale_count += 1
            t_new, ry_new = track.get_interpolated_position(pose_prev_to_curr, calib)
            last_bbox = track.bboxes[-1]
            # make new bbox
            obj = types.SimpleNamespace()
            obj.t = t_new
            obj.l = last_bbox.l
            obj.w = last_bbox.w
            obj.h = last_bbox.h
            obj.ry = ry_new
            obj.volume = last_bbox.volume
            track.bboxes.append(obj)
        track.last_prev_to_curr_pose = pose_prev_to_curr
    else:
        track.terminated = True


def associate_bbox_to_track(tracks,  # list of track objects
                            bboxes,  # new bboxes to add into tracks
                            pose_prev,  # previous pose
                            pose_curr,  # bbox pose
                            l2e_prev,  # bbox pose
                            l2e_curr,  # bbox pose
                            calib,  # kitti calibration tool
                            frame_id
):
    # Do association:
    # Compute poses of next frame in current frame
    _pose_prev_to_curr = get_relative_pose(
        fixed_l2e=l2e_curr, fixed_ego=pose_curr,
        query_l2e=l2e_prev, query_ego=pose_prev,
        KITTI2NU=_KITTI2NU_lyft)
    
    bboxes_prev = [track.bboxes[-1] for track in tracks]

    # Compute association based on distance to closest bbox in next frame
    trans_curr = np.array([obj.t for obj in bboxes]) 
    trans_prev = np.array([translate_rect_with_velo(bbox.t[np.newaxis, :], _pose_prev_to_curr, calib).reshape(3,) for bbox in bboxes_prev]) # TODO: debug

    # association matrix on distance
    if trans_curr.shape[0] > 0 and trans_prev.shape[0] > 0:
        dist_matrix = np.linalg.norm(trans_curr[:, None, :] - trans_prev[None, :, :], axis=-1)
        curr_ind, prev_ind = linear_sum_assignment(dist_matrix)
    else:
        curr_ind, prev_ind = [], []
    new_tracks = []

    for curr_i, prev_i in zip(curr_ind, prev_ind):
        dist_association = dist_matrix[curr_i, prev_i]

        if dist_association > ASSOCIATION_THRESHOLD:
            # Do not count this as an association
            process_unassociated_track(tracks[prev_i], _pose_prev_to_curr, calib)

            # make a new track
            new_track = Track(bboxes[curr_i], frame_id)
            new_tracks.append(new_track)

        else:
            tracks[prev_i].bboxes.append(bboxes[curr_i])
            tracks[prev_i].last_prev_to_curr_pose = _pose_prev_to_curr

    # set all tracks that were not associated to be terminated
    for prev_i, track in enumerate(tracks):
        if prev_i not in prev_ind:
            process_unassociated_track(track, _pose_prev_to_curr, calib)
    for curr_i, bbox in enumerate(bboxes):
        if curr_i not in curr_ind:
            # make a new track
            new_track = Track(bbox, frame_id)
            new_tracks.append(new_track)
    
    return new_tracks


def transpose_tracks_to_frame_sequence(tracks, num_frames):
    frames = [[] for _ in range(num_frames)]
    for track in tracks:
        start_id = track.start_id
        for i, bbox in enumerate(track.bboxes):
            frames[start_id + i].append(bbox)
    return frames


def compute_aligned_transform(ptc1, ptc2, pose_ptc2_to_ptc1):
    # print(ptc1.shape, ptc2.shape)
    # point cloud 2 into ptc1 coordinate frame
    ptc2_in_ptc1 = transform_points(ptc2, pose_ptc2_to_ptc1)
    # compute the transformation from ptc2 to ptc1 in ptc1's frame
    tranf_bbox2_to_bbox1 = get_trans_matrix(ptc1, ptc2_in_ptc1, global_reg=True)
    # compute combined points in ptc1 frame
    combined_box_ptc_new = np.concatenate(
        (ptc1, transform_points(ptc2_in_ptc1, tranf_bbox2_to_bbox1)))
    return combined_box_ptc_new, tranf_bbox2_to_bbox1


def tracking_reshape(bbox_sequence, # lists of list if bbox in each frame (in ego coordinate)
                     pose_sequence, # pose of ego each frame
                     l2e_sequence, # lidar to ego for each frame
                     ptc_file_sequence, # sequence of point clouds files
                     calibs_sequence, # sequence of calibrations
                     re_filter_aligned_tracks=False
):
    tracks = []
    frame_ids = list(range(len(bbox_sequence)))

    # associate bboxes in each frame to a track
    for frame_id in frame_ids:
        frame_bboxes = bbox_sequence[frame_id]

        # add first frame objects into track
        if frame_id == 0:
            for bbox in frame_bboxes:
                track = Track(bbox, frame_id)
                tracks.append(track)
        else: # update with new bboxes
            
            # pull info in previous frame from tracks
            active_tracks = [track for track in tracks if not track.terminated]
            pose_prev = pose_sequence[frame_id-1]
            l2e_prev = l2e_sequence[frame_id-1]

            # Pull information from current frame
            calib = calibs_sequence[frame_id]
            bboxes_curr = frame_bboxes
            pose_curr = pose_sequence[frame_id]
            l2e_curr = l2e_sequence[frame_id]

            # Do association:
            new_tracks = associate_bbox_to_track(active_tracks, bboxes_curr, pose_prev, pose_curr, l2e_prev, l2e_curr, calib, frame_id)
            tracks.extend(new_tracks)

    for track in tracks:
        filtered_bboxes = []
        for idx, bbox in enumerate(track.bboxes):
            calib = calibs_sequence[track.start_id + idx]
            ptc = load_velo_scan(ptc_file_sequence[track.start_id + idx])
            ptc_bbox_mask = get_obj_ptc_mask(calib.project_velo_to_rect(ptc[:, :3]), track.bboxes[idx])
            ptc_bbox = ptc[ptc_bbox_mask]
            if ptc_bbox.shape[0] <= MIN_POINTS_THRESH:
                break
            else:
                filtered_bboxes.append(bbox)
            # if ptc_bbox.shape[0] > 0:
            #     filtered_bboxes.append(bbox)
        track.bboxes = filtered_bboxes

    if re_filter_aligned_tracks:
        for track in tracks:
            # process tracks
            # find closest bbox, select alignment order
            if len(track.bboxes) == 1:
                continue
            
            order = np.argsort([np.linalg.norm(bbox.t) for bbox in track.bboxes])
            closest_bbox_idx = order[0]

            calib = calibs_sequence[track.start_id + closest_bbox_idx]
            ptc1 = load_velo_scan(ptc_file_sequence[track.start_id + closest_bbox_idx])
            ptc_bbox1_mask = get_obj_ptc_mask(calib.project_velo_to_rect(ptc1[:, :3]), track.bboxes[closest_bbox_idx])
            ptc_bbox1 = ptc1[ptc_bbox1_mask]

            # initialize the combined pointcloud with the closest bbox
            combined_box_ptc = ptc_bbox1[:, :3]
            pose_closest = pose_sequence[track.start_id + closest_bbox_idx]
            l2e_closest = l2e_sequence[track.start_id + closest_bbox_idx]


            # relative_trans = {}
            trans_to_closest = {}
            for _idx in order[1:]:
                calib = calibs_sequence[track.start_id + _idx]
                ptc2 = load_velo_scan(ptc_file_sequence[track.start_id + _idx])
                ptc_bbox2_mask = get_obj_ptc_mask(calib.project_velo_to_rect(ptc2[:, :3]), track.bboxes[_idx])
                ptc_bbox2 = ptc2[ptc_bbox2_mask, :3]
                # transform ptc_bbox2 to closest_bbox_idx coordinate first
                pose_curr = pose_sequence[track.start_id + _idx]
                l2e_curr = l2e_sequence[track.start_id + _idx]
                _trans_to_closest = get_relative_pose(
                    fixed_l2e=l2e_closest, fixed_ego=pose_closest,
                    query_l2e=l2e_curr, query_ego=pose_curr,
                    KITTI2NU=_KITTI2NU_lyft)
                combined_box_ptc, _trans_bbox_velo = compute_aligned_transform(combined_box_ptc, ptc_bbox2, _trans_to_closest)
                trans_to_closest[_idx] = _trans_bbox_velo @ _trans_to_closest
            
            calib = calibs_sequence[track.start_id + closest_bbox_idx]        
            tracked_bbox = get_obj(
                calib.project_velo_to_rect(combined_box_ptc),
                calib.project_velo_to_rect(ptc1[:, :3]), fit_method='closeness_to_edge')
            track.bboxes[closest_bbox_idx] = tracked_bbox
            
            for _ind in order[1:]:
                calib = calibs_sequence[track.start_id + _ind]
                rel_from_closest_velo = np.linalg.inv(trans_to_closest[_ind])
                bbox_ind = transform_bbox_rect_with_velo(tracked_bbox, rel_from_closest_velo, calib)
                
                track.bboxes[_ind] = bbox_ind
        
    # transpose per-track representation to per frame instance
    updated_bbox_sequence = transpose_tracks_to_frame_sequence(tracks, len(bbox_sequence))

    return updated_bbox_sequence