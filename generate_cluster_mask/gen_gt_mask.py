import argparse
from utils import kitti_util
import os
import os.path as osp
import numpy as np
import cv2
from tqdm.auto import tqdm
import pickle


from scipy.spatial import Delaunay
import scipy
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
        flag = np.zeros(p.shape[0], dtype=bool)

    return flag

def range_cutoff(ptc):
    return (ptc[:, 0] < 80)

def gen_gt_labels(origin_ptc, calib, labels, classes=('Dynamic',), within_image=True, img_shape=(1024, 1224)):
    if within_image:
        assert False
        pts_img = calib.project_velo_to_image(origin_ptc[:,:3])
        pts_rect_depth = calib.project_velo_to_rect(origin_ptc[:,:3])[:, 2]
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        pts_valid_flag = np.logical_and(pts_valid_flag, range_cutoff(origin_ptc))
    else:
        pts_valid_flag = np.ones(origin_ptc.shape[0], dtype=bool)
    gt_labels = np.zeros(origin_ptc.shape[0], dtype=np.int32) - 1
    gt_labels[pts_valid_flag] = 0
    count = 1
    for label in labels:
        if not label.cls_type in classes:
            continue
        mask = in_hull(origin_ptc[pts_valid_flag, :3], calib.project_rect_to_velo((kitti_util.compute_box_3d(label, calib.P)[1])))
        gt_labels[np.where(pts_valid_flag)[0][mask]] = count
        count += 1
    return gt_labels


calib_path = "/home/yy785/datasets/lyft_release_test/training/calib"
ptc_path = "/home/yy785/datasets/lyft_release_test/training/velodyne"
label_path = "/home/yy785/datasets/lyft_release_test/training/label_2_dynamic_obj_full_range/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--total_part", type=int, default=1)
    return parser.parse_args()

def main(args):
    idx_list = [int(x) for x in open(
        "/home/yy785/projects/object_discovery/generate_cluster_mask/meta_data/bwfw40_test_idx.txt", "r").readlines()]
    idx_list = np.array(idx_list)
    if args.total_part > 1:
        idx_list = np.array_split(
            idx_list, args.total_part)[args.part]
    gt_save_path = "/home/yy785/datasets/lyft_release_test/training/gt_segmentation/"
    os.makedirs(gt_save_path, exist_ok=True)
    for origin_idx in tqdm(idx_list):
        ptc = kitti_util.load_velo_scan(
            osp.join(ptc_path, f"{origin_idx:06d}.bin"))
        calib = kitti_util.Calibration(
            osp.join(calib_path, f"{origin_idx:06d}.txt"))
        labels = kitti_util.read_label(
            osp.join(label_path, f"{origin_idx:06d}.txt"))
        gt_labels = gen_gt_labels(ptc, calib, labels, within_image=False)
        np.save(osp.join(gt_save_path, f"{origin_idx:06d}.npy"), gt_labels)

if __name__ == "__main__":
    main(parse_args())
