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

from utils import kitti_util
from utils.pointcloud_utils import objs_nms, objs2label, is_within_fov

warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.EfficiencyWarning)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def display_args(args):
    eprint("========== kitti_label gen info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("==========================================")

@hydra.main(config_path="configs/", config_name="generate_label_files.yaml")
def main(args: DictConfig):
    display_args(args)
    idx_list = [int(x) for x in open(args.data_paths.idx_list).readlines()]
    idx_list = np.array(idx_list)
    if args.total_part > 1:
        idx_list = np.array_split(
            idx_list, args.total_part)[args.part]
    os.makedirs(args.data_paths.label_file_save_dst, exist_ok=True)
    for idx in tqdm(idx_list):
        objs = pickle.load(
            open(osp.join(
                args.data_paths.bbox_info_save_dst, f"{idx:06d}.pkl"), "rb"))
        if args.nms.enable and len(objs) > 0:
            objs = objs_nms(objs, nms_threshold=args.nms.threshold)
        calib = kitti_util.Calibration(
            osp.join(args.calib_path, f"{idx:06d}.txt"))
        if args.fov_only:
            objs = [obj for obj in objs if is_within_fov(
                obj, calib, args.image_shape)]
        with open(osp.join(args.data_paths.label_file_save_dst, f"{idx:06d}.txt"), "w") as f:
            f.write(objs2label(objs, calib))


if __name__ == "__main__":
    main()
