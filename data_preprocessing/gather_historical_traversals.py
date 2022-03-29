import argparse
import os
import os.path as osp
import pickle

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm


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


_KITTI2NU_lyft = Quaternion(axis=(0, 0, 1), angle=np.pi).transformation_matrix
_KITTI2NU_nusc = Quaternion(
    axis=(0, 0, 1), angle=np.pi/2).transformation_matrix


def get_relative_pose(fixed_l2e, fixed_ego, query_l2e, query_ego, KITTI2NU=_KITTI2NU_lyft):
    return np.linalg.solve(KITTI2NU, np.linalg.solve(fixed_l2e, np.linalg.solve(fixed_ego, query_ego @ query_l2e @ KITTI2NU))).astype(np.float32)


def remove_center(ptc, x_range=(-1.15, 1.75), y_range=(-0.65, 0.65)):
    mask = (ptc[:, 0] < x_range[1]) & (ptc[:, 0] >= x_range[0]) & (
        ptc[:, 1] < y_range[1]) & (ptc[:, 1] >= y_range[0])
    mask = np.logical_not(mask)
    return ptc[mask]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_path", type=str)
    parser.add_argument("--idx_info", type=str)
    parser.add_argument("--traversal_ptc_save_root", type=str, default=None)
    parser.add_argument("--trans_mat_save_root",
                        type=str, default=None)
    parser.add_argument("--idx_list",
                        type=str, default=None)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--nusc", action="store_true")

    return parser.parse_args()


def main(args):
    track_list = pickle.load(open(args.track_path, "rb"))
    valid_idx = pickle.load(open(args.idx_info, "rb"))
    oxts_path = osp.join(args.data_root, "training/oxts")
    # LiDAR to Ego transformation
    l2e_path = osp.join(args.data_root, "training/l2e")
    poses = []
    for seq in tqdm(track_list):
        poses.append([])
        for idx in seq:
            with open(osp.join(oxts_path, f"{idx:06d}.txt"), "r") as f:
                info = np.array([float(x) for x in f.readline().split()])
                trans = np.eye(4)
                trans[:3, 3] = info[:3]
                trans[:3, :3] = R.from_euler('xyz', info[3:]).as_matrix()
                poses[-1].append(trans.astype(np.float32))

    l2es = []
    for seq in tqdm(track_list):
        l2es.append([])
        for idx in seq:
            l2es[-1].append(np.load(osp.join(l2e_path, f"{idx:06d}.npy")))

    if args.idx_list is not None:
        idx_list = [int(x) for x in open(args.idx_list).readlines()]
    else:
        idx_list = [x for x in valid_idx]

    idx_list = np.array(idx_list)
    if args.trans_mat_save_root is not None:
        os.makedirs(args.trans_mat_save_root, exist_ok=True)
    if args.traversal_ptc_save_root is not None:
        os.makedirs(args.traversal_ptc_save_root, exist_ok=True)

    for origin_idx in tqdm(idx_list):
        N = len(valid_idx[origin_idx][2])
        assert N > 1, origin_idx
        seq_id, indices = valid_idx[origin_idx][2][0]
        first_seq_id = seq_id
        first_pose = poses[seq_id][indices[0]]
        first_l2e = l2es[seq_id][indices[0]]

        combined_lidar = {}
        for seq_id, indices in valid_idx[origin_idx][2]:
            _combined_ptcs_queue = []
            for frame in indices:
                _ptc = load_velo_scan(
                    osp.join(
                        args.data_root,
                        "training/velodyne",
                        f"{track_list[seq_id][frame]:06d}.bin"))[:, :3]
                if args.nusc:  # Remove center for nuscenes dataset
                    _ptc = remove_center(_ptc)
                _relative_pose = get_relative_pose(
                    fixed_l2e=first_l2e, fixed_ego=first_pose,
                    query_l2e=l2es[seq_id][frame], query_ego=poses[seq_id][frame],
                    KITTI2NU=_KITTI2NU_nusc if args.nusc else _KITTI2NU_lyft)
                _ptc = transform_points(_ptc, _relative_pose)
                _combined_ptcs_queue.append(_ptc)
            _combined_ptcs = np.concatenate(_combined_ptcs_queue)
            combined_lidar[seq_id] = _combined_ptcs.astype(np.float32)

        pickle.dump(combined_lidar,
                    open(osp.join(args.traversal_ptc_save_root,
                                    f"{origin_idx:06d}.pkl"), "wb"))

        origin_seq = valid_idx[origin_idx][0]
        origin_frame = valid_idx[origin_idx][1]
        origin_pose = poses[origin_seq][origin_frame]
        origin_l2e = l2es[origin_seq][origin_frame]
        trans_mat = get_relative_pose(
            fixed_l2e=first_l2e, fixed_ego=first_pose,
            query_l2e=origin_l2e, query_ego=origin_pose,
            KITTI2NU=_KITTI2NU_nusc if args.nusc else _KITTI2NU_lyft)
        np.save(osp.join(args.trans_mat_save_root,
                            f"{origin_idx:06d}.npy"),
                trans_mat)

if __name__ == "__main__":
    main(parse_args())
