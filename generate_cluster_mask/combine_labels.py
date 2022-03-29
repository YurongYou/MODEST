import os
import os.path as osp
import pickle
import sys
import warnings

import hydra
import numpy as np
import sklearn
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from utils import kitti_util
from utils.pointcloud_utils import (is_within_fov, load_velo_scan, objs2label,
                                    objs_nms)

warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.EfficiencyWarning)

from types import SimpleNamespace


def predicts2objs(preds):
    objs = []
    for i in range(preds['location'].shape[0]):
        obj = SimpleNamespace()
        obj.t = preds['location'][i]
        obj.l = preds['dimensions'][i][0]
        obj.h = preds['dimensions'][i][1]
        obj.w = preds['dimensions'][i][2]
        obj.ry = preds['rotation_y'][i]
        obj.score = preds['score'][i]
        objs.append(obj)
    return objs


def add_area_score(objs):
    for obj in objs:
        obj.score = -999 + obj.w * obj.l

def filter_by_ppscore(ptc_rect, pp_score, obj, percentile=50, threshold=0.5):
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
    if mask.sum() == 0 or np.percentile(pp_score[mask], percentile) > threshold:
        return False
    return True

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def display_args(args):
    eprint("========== combine_labels info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=========================================")


@hydra.main(config_path="configs/", config_name="combine_labels.yaml")
def main(args: DictConfig):
    display_args(args)
    # idx_list = [int(x) for x in open(args.data_paths.idx_list).readlines()]

    det_bboxes = pickle.load(open(args.det_result_path, "rb"))
    idx_list = [int(det_bbox['frame_id']) for det_bbox in det_bboxes]
    idx_list = np.array(idx_list)
    if args.total_part > 1:
        idx_list = np.array_split(
            idx_list, args.total_part)[args.part]
    os.makedirs(args.save_path, exist_ok=True)
    if args.data_paths.bbox_info_save_dst is None:
        eprint("Warning: not adding generated bboxes")
    for idx, det_bbox in zip(tqdm(idx_list), det_bboxes):
        if args.data_paths.bbox_info_save_dst is not None:
            gen_obj = pickle.load(
                open(osp.join(args.data_paths.bbox_info_save_dst, f'{idx:06d}.pkl'), "rb"))
        else:
            gen_obj = []
        assert idx == int(det_bbox['frame_id'])
        calib = kitti_util.Calibration(
            osp.join(args.calib_path, f"{idx:06d}.txt"))
        ptc = load_velo_scan(osp.join(args.ptc_path, f"{idx:06d}.bin"))
        ptc_in_rect = calib.project_velo_to_rect(ptc[:, :3])
        pp_score = np.load(
            osp.join(args.data_paths.pp_score_path, f"{idx:06d}.npy"))

        det_obj = list(filter(
            lambda obj: filter_by_ppscore(
                ptc_in_rect, pp_score, obj,
                percentile=args.det_filtering.pp_score_percentile,
                threshold=args.det_filtering.pp_score_threshold) & \
            (obj.score > args.det_filtering.score_filtering),
            predicts2objs(det_bbox)))
        add_area_score(gen_obj)
        objs = det_obj + gen_obj
        if len(objs) > 0:
            objs = objs_nms(objs,
                            nms_threshold=args.nms.threshold,
                            use_score_rank=True)

        calib = kitti_util.Calibration(
            osp.join(args.calib_path, f"{idx:06d}.txt"))
        if args.fov_only:
            objs = [obj for obj in objs if is_within_fov(
                obj, calib, args.image_shape)]
        with open(osp.join(args.save_path, f"{idx:06d}.txt"), "w") as f:
            f.write(objs2label(objs, calib, with_score=args.with_score))


if __name__ == "__main__":
    main()
