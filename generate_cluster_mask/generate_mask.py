import os
import os.path as osp
import sys
import warnings
import pickle

import hydra
import numpy as np
import sklearn
from omegaconf import DictConfig, OmegaConf
from sklearn import cluster
from tqdm.auto import tqdm

from utils.clustering_utils import filter_labels, precompute_affinity_matrix
from utils.pointcloud_utils import above_plane, estimate_plane, load_velo_scan, get_obj
from utils import kitti_util

warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.EfficiencyWarning)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def display_args(args):
    eprint("========== clustering info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=====================================")

@hydra.main(config_path="configs/", config_name="generate_mask.yaml")
def main(args: DictConfig):
    display_args(args)
    idx_list = [int(x) for x in open(args.data_paths.idx_list).readlines()]
    idx_list = np.array(idx_list)
    if args.total_part > 1:
        idx_list = np.array_split(
            idx_list, args.total_part)[args.part]
    os.makedirs(args.data_paths.seg_save_dst, exist_ok=True)
    if not osp.exists(osp.join(args.data_paths.seg_save_dst, "configs.yaml")):
        OmegaConf.save(config=args, f=osp.join(args.data_paths.seg_save_dst, "configs.yaml"))

    if args.data_paths.get("bbox_info_save_dst", "None") is not None:
        os.makedirs(args.data_paths.bbox_info_save_dst, exist_ok=True)
        if not osp.exists(osp.join(args.data_paths.bbox_info_save_dst, "configs.yaml")):
            OmegaConf.save(config=args, f=osp.join(
                args.data_paths.bbox_info_save_dst, "configs.yaml"))
    for idx in tqdm(idx_list):
        if osp.exists(osp.join(args.data_paths.seg_save_dst, f"{idx:06d}.npy")) and \
            (args.get("bbox_info_save_dst", "None") is None or
            osp.exists(osp.join(args.data_paths.bbox_info_save_dst, f"{idx:06d}.pkl"))):
            continue
        ptc = load_velo_scan(osp.join(args.ptc_path, f"{idx:06d}.bin"))
        pp_score = np.load(
            osp.join(args.data_paths.pp_score_path, f"{idx:06d}.npy"))
        plane = estimate_plane(
            ptc[:, :3], max_hs=args.plane_estimate.max_hs, ptc_range=args.plane_estimate.range)
        plane_mask = above_plane(
            ptc[:, :3], plane,
            offset=args.plane_estimate.offset,
            only_range=args.plane_estimate.range)
        range_mask = (ptc[:, 0] <= args.limit_range[0][1]) * \
            (ptc[:, 0] > args.limit_range[0][0]) * \
            (ptc[:, 1] <= args.limit_range[1][1]) * \
            (ptc[:, 1] > args.limit_range[1][0])
        final_mask = plane_mask * range_mask
        dist_knn_graph = precompute_affinity_matrix(
            ptc[final_mask],
            pp_score[final_mask],
            neighbor_type=args.graph.neighbor_type,
            affinity_type=args.graph.affinity_type,
            n_neighbors=args.graph.n_neighbors,
            radius=args.graph.radius,
        )

        if args.clustering.method == "DBSCAN":
            labels = np.zeros(ptc.shape[0], dtype=int) - 1
            labels[final_mask] = cluster.DBSCAN(
                metric='precomputed',
                eps=args.clustering.DBSCAN.eps,
                min_samples=args.clustering.DBSCAN.min_samples,
                n_jobs=-1).fit(dist_knn_graph).labels_
        else:
            raise NotImplementedError(args.clustering.method)
        labels_filtered = filter_labels(
            ptc, pp_score, labels,
            **args.filtering)

        calib = kitti_util.Calibration(
            osp.join(args.calib_path, f"{idx:06d}.txt"))
        ptc_in_rect = calib.project_velo_to_rect(ptc[:, :3])
        objs = []
        for i in range(1, labels_filtered.max()+1):
            obj = get_obj(ptc_in_rect[labels_filtered == i], ptc_in_rect,
                            fit_method=args.bbox_gen.fit_method)
            if obj.volume > args.filtering.min_volume and obj.volume < args.filtering.max_volume:
                objs.append(obj)
            else:
                labels_filtered[labels_filtered == i] = 0

        label_mapping = sorted(list(set(labels_filtered)))
        label_mapping = {x: i for i, x in enumerate(label_mapping)}
        for _i in label_mapping:
            labels_filtered[labels_filtered == _i] = label_mapping[_i]

        if args.data_paths.get("bbox_info_save_dst", "None") is not None:
            pickle.dump(objs, open(
                osp.join(args.data_paths.bbox_info_save_dst, f"{idx:06d}.pkl"), "wb"))
        np.save(osp.join(args.data_paths.seg_save_dst, f"{idx:06d}.npy"),
                labels_filtered)

if __name__ == "__main__":
    main()
