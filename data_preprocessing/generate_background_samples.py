import argparse
import os
import os.path as osp
import pickle

import MinkowskiEngine as ME
import numpy as np
import scipy
import kitti_util
from pyquaternion import Quaternion
from scipy.spatial import Delaunay
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

def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def extend_width(obj, values=(1.0, 1.0, 1.0)):
    obj.w += values[0]
    obj.l += values[1]
    obj.h += values[2]
    return obj


def remove_objects(ptc, label_path, calib_path, idx, extend_values=(1.0, 1.0, 1.0)):
    labels = kitti_util.read_label(osp.join(label_path, f"{idx:06d}.txt"))
    labels = [extend_width(obj, extend_values) for obj in labels]
    calib = kitti_util.Calibration(osp.join(calib_path, f"{idx:06d}.txt"))
    mask = np.zeros(ptc.shape[0]).astype(bool)
    for label in labels:
        mask = np.logical_or(mask,
                             in_hull(ptc[:, :3],
                                     calib.project_rect_to_velo((kitti_util.compute_box_3d(label, calib.P)[1])))
                             )
    return ptc[~mask]


def remove_center(ptc, x_range=(-1.15, 1.75), y_range=(-0.65, 0.65)):
    mask = (ptc[:, 0] < x_range[1]) & (ptc[:, 0] >= x_range[0]) & (
        ptc[:, 1] < y_range[1]) & (ptc[:, 1] >= y_range[0])
    mask = np.logical_not(mask)
    return ptc[mask]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_path", type=str,
                        default="./lyft/meta_data/train_track_list.pkl")
    parser.add_argument("--idx_info", type=str,
                        default="./lyft/meta_data/valid_train_idx_info.pkl")
    parser.add_argument("--idx_list", type=str,
                        default="./lyft/meta_data/train_idx.txt")
    parser.add_argument("--mata_file_root", type=str,
                        default="./lyft/meta_data/")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--label_dir", type=str)
    parser.add_argument("--calib_dir", type=str)
    parser.add_argument("--trans_mat_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--use_n_hist", type=int, default=-1)
    parser.add_argument("--voxel_size", type=float, default=0.4)
    parser.add_argument("--nusc", action="store_true")
    parser.add_argument("--nusc_max_dist", type=float, default=30.)

    return parser.parse_args()

def main(args):
    train_seq = pickle.load(open(args.track_path, "rb"))
    valid_idx = pickle.load(open(args.idx_info, "rb"))
    oxts_path = osp.join(args.data_root, "training/oxts")
    l2e_path = osp.join(args.data_root, "training/l2e")

    poses = []
    for seq in tqdm(train_seq):
        poses.append([])
        for idx in seq:
            with open(osp.join(oxts_path, f"{idx:06d}.txt"), "r") as f:
                info = np.array([float(x) for x in f.readline().split()])
                trans = np.eye(4)
                trans[:3, 3] = info[:3]
                trans[:3, :3] = R.from_euler('xyz', info[3:]).as_matrix()
                poses[-1].append(trans.astype(np.float32))

    l2es = []
    for seq in tqdm(train_seq):
        l2es.append([])
        for idx in seq:
            l2es[-1].append(np.load(osp.join(l2e_path, f"{idx:06d}.npy")))

    if args.nusc:
        track_infos = pickle.load(open(
            osp.join(args.mata_file_root, "track_infos.pkl"), "rb"))

        token_seq_labeled = pickle.load(open(
            osp.join(args.mata_file_root, "labeled_tokens.pkl"), "rb"))

        sequence_mapping = pickle.load(open(
            osp.join(args.mata_file_root, "sequence_mapping.pkl"), "rb"))
        boston_scenes = pickle.load(open(
            osp.join(args.mata_file_root, "scenes_tokens.pkl"), "rb"))

        track_list = track_infos[0]
        token_seq = track_infos[1]

        token_ld2seq_frame = {}
        for seq_id, seq in enumerate(track_list):
            for frame_id, idx in enumerate(seq):
                token_ld2seq_frame[token_seq[idx]] = (seq_id, frame_id)

        labeledidx2seq_frame = {}
        for scene in boston_scenes:
            for tk in sequence_mapping[scene]:
                if token_seq_labeled[tk] in token_ld2seq_frame:
                    labeledidx2seq_frame[tk] = token_ld2seq_frame[token_seq_labeled[tk]]
                else:
                    raise ValueError(f"{tk} not found")
        trackidx2labeledidx = {}
        for labeledidx, (seq_id, frame) in labeledidx2seq_frame.items():
            trackidx2labeledidx[train_seq[seq_id][frame]] = labeledidx

    idx_list = [int(x) for x in open(args.idx_list).readlines()]

    valid_train_idx_list = idx_list

    os.makedirs(args.save_dir, exist_ok=True)
    for origin_idx in tqdm(valid_train_idx_list):
        N = len(valid_idx[origin_idx][2])
        assert N > 1, origin_idx
        if osp.exists(osp.join(args.save_dir, f"{origin_idx:06d}.npy")):
            continue
        seq_id, indices = valid_idx[origin_idx][2][0]
        first_seq_id = seq_id
        first_pose = poses[seq_id][indices[0]]
        first_l2e = l2es[seq_id][indices[0]]

        seq_indices = valid_idx[origin_idx][2]
        if args.use_n_hist > 0:
            seq_indices = valid_idx[origin_idx][2][:args.use_n_hist]

        combined_lidar = {}
        for seq_id, indices in seq_indices:
            _combined_ptcs_queue = []
            if args.nusc:
                forward = indices[1] - indices[0] > 0
                if forward:
                    indices = list(range(indices[0], len(train_seq[seq_id])))
                else:
                    indices = list(range(indices[0], -1, -1))
                indices = [_i for _i in indices
                           if train_seq[seq_id][_i] in trackidx2labeledidx]
            for frame in indices:
                distance = np.linalg.norm(
                    poses[seq_id][frame][:2, 3] - first_pose[:2, 3])
                if args.nusc and distance > args.nusc_max_dist:
                    break
                _ptc = load_velo_scan(
                    osp.join(
                        args.data_root,
                        "training/velodyne",
                        f"{train_seq[seq_id][frame]:06d}.bin"))[:, :3]
                if args.nusc:  # Remove center for nuscenes dataset
                    _ptc = remove_center(_ptc)
                    _ptc = remove_objects(_ptc, args.label_dir,
                                          args.calib_dir,
                                          trackidx2labeledidx[train_seq[seq_id][frame]])
                else:
                    _ptc = remove_objects(_ptc, args.label_dir,
                                        args.calib_dir, train_seq[seq_id][frame])
                _relative_pose = get_relative_pose(
                    fixed_l2e=first_l2e, fixed_ego=first_pose,
                    query_l2e=l2es[seq_id][frame], query_ego=poses[seq_id][frame],
                    KITTI2NU=_KITTI2NU_nusc if args.nusc else _KITTI2NU_lyft)
                _ptc = transform_points(_ptc, _relative_pose)
                _combined_ptcs_queue.append(_ptc)
            _combined_ptcs = np.concatenate(_combined_ptcs_queue)
            combined_lidar[seq_id] = _combined_ptcs.astype(np.float32)
        combined_lidar = [v for k, v in combined_lidar.items()]
        combined_lidar = np.concatenate(combined_lidar)
        trans_mat = np.load(
            osp.join(args.trans_mat_dir, f"{origin_idx:06d}.npy"))
        combined_lidar = transform_points(
            combined_lidar, np.linalg.inv(trans_mat))
        if args.voxel_size > 0:
            _, quantize_idx = ME.utils.sparse_quantize(
                combined_lidar / args.voxel_size, return_index=True)
            combined_lidar = combined_lidar[quantize_idx]
        np.save(osp.join(args.save_dir,
                f"{origin_idx:06d}.npy"), combined_lidar)


if __name__ == "__main__":
    main(parse_args())
