import os
import os.path as osp
import pickle
import sys

import hydra
import numpy as np
import scipy
from omegaconf import DictConfig, OmegaConf
from pyquaternion import Quaternion
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from utils.pointcloud_utils import load_velo_scan, transform_points


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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


def remove_center(ptc, x_range=(-1.15, 1.75), y_range=(-0.65, 0.65)):
    mask = (ptc[:, 0] < x_range[1]) & (ptc[:, 0] >= x_range[0]) & (
        ptc[:, 1] < y_range[1]) & (ptc[:, 1] >= y_range[0])
    mask = np.logical_not(mask)
    return ptc[mask]

def count_neighbors(ptc, trees, args):
    neighbor_count = {}
    for seq in trees.keys():
        neighbor_count[seq] = trees[seq].query_ball_point(
            ptc[:, :3], r=args.max_neighbor_dist,
            return_length=True)
    return np.stack(list(neighbor_count.values())).T


def shuffle_along(X):
    """Minimal in place independent-row shuffler."""
    [np.random.shuffle(x) for x in X]


def compute_ephe_score(count, args):
    N = count.shape[1]
    if args.ephe_type == "entropy":
        P = count / (np.expand_dims(count.sum(axis=1), -1) + 1e-8)
        H = (-P * np.log(P + 1e-8)).sum(axis=1) / np.log(N)
    else:
        raise NotImplementedError()
    return H

def display_args(args):
    eprint("========== ephemerality info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")

@hydra.main(config_path="configs/", config_name="pp_score.yaml")
def main(args: DictConfig):
    display_args(args)
    track_list = pickle.load(open(args.data_paths.track_path, "rb"))
    valid_idx = pickle.load(open(args.data_paths.idx_info, "rb"))
    os.makedirs(args.data_paths.pp_score_path, exist_ok=True)
    oxts_path = osp.join(args.data_root, "oxts")
    l2e_path = osp.join(args.data_root, "l2e")

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

    # process the whole training set
    if args.data_paths.idx_list is not None:
        idx_list = [int(x) for x in open(args.data_paths.idx_list).readlines()]
    else:
        idx_list = [x for x in valid_idx]
    valid_train_idx_list = np.array(idx_list)
    if args.total_part > 1:
        valid_train_idx_list = np.array_split(
            valid_train_idx_list, args.total_part)[args.part]
    if args.data_paths.load_save_precomputed_trans_mat is not None:
        os.makedirs(args.data_paths.load_save_precomputed_trans_mat, exist_ok=True)
    if args.data_paths.load_precomputed_lidars is not None:
        os.makedirs(args.data_paths.load_precomputed_lidars, exist_ok=True)

    for origin_idx in tqdm(valid_train_idx_list):
        if osp.exists(osp.join(args.data_paths.pp_score_path, f"{origin_idx:06d}")):
            continue
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
                        "velodyne",
                        f"{track_list[seq_id][frame]:06d}.bin"))[:, :3]
                if args.nusc:
                    _ptc = remove_center(_ptc)
                _relative_pose = get_relative_pose(
                    fixed_l2e=first_l2e, fixed_ego=first_pose,
                    query_l2e=l2es[seq_id][frame], query_ego=poses[seq_id][frame],
                    KITTI2NU=_KITTI2NU_nusc if args.nusc else _KITTI2NU_lyft)
                _ptc = transform_points(_ptc, _relative_pose)
                _combined_ptcs_queue.append(_ptc)
            _combined_ptcs = np.concatenate(_combined_ptcs_queue)
            combined_lidar[seq_id] = _combined_ptcs.astype(np.float32)

        if args.data_paths.load_precomputed_lidars is not None:
            pickle.dump(combined_lidar,
                        open(osp.join(args.data_paths.load_precomputed_lidars,
                                        f"{origin_idx:06d}.pkl"), "wb"))
        origin_seq = valid_idx[origin_idx][0]
        origin_frame = valid_idx[origin_idx][1]
        origin_pose = poses[origin_seq][origin_frame]
        origin_l2e = l2es[origin_seq][origin_frame]
        origin_ptc = load_velo_scan(
            osp.join(args.data_root, "velodyne",
                     f"{track_list[origin_seq][origin_frame]:06d}.bin"))[:, :3]

        trans_mat = get_relative_pose(
            fixed_l2e=first_l2e, fixed_ego=first_pose,
            query_l2e=origin_l2e, query_ego=origin_pose,
            KITTI2NU=_KITTI2NU_nusc if args.nusc else _KITTI2NU_lyft)
        if args.data_paths.load_save_precomputed_trans_mat is not None:
            np.save(osp.join(args.data_paths.load_save_precomputed_trans_mat,
                                f"{origin_idx:06d}.npy"),
                    trans_mat)
        if args.skip_ephe:
            continue
        origin_ptc = transform_points(origin_ptc[:, :3], trans_mat)
        if args.add_random_noise > 0:
            noise = np.random.randn(3)
            noise /= np.linalg.norm(noise)
            noise *= (args.add_random_noise * np.random.uniform())
            origin_ptc += noise.reshape(-1, 3)

        if args.limit_traversals > 1:
            temp = combined_lidar
            combined_lidar = {}
            traversals = [x[0] for x in valid_idx[origin_idx][2]]
            for k in traversals[:args.limit_traversals]:
                combined_lidar[k] = temp[k]

        trees = {}
        for seq, ptc in combined_lidar.items():
            trees[seq] = cKDTree(ptc)

        # count the neighbors of each points
        count = count_neighbors(origin_ptc, trees, args)
        H = compute_ephe_score(count, args)
        np.save(osp.join(args.data_paths.pp_score_path,
                         f"{origin_idx:06d}"), H.astype(np.float32))


if __name__ == "__main__":
    main()
