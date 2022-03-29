import argparse
import os

import numpy as np
import kitti_util as utils
from sklearn.linear_model import RANSACRegressor


def extract_ransac(calib_dir, lidar_dir, planes_dir, min_h=1.5, max_h=2, split_file=None):
    if split_file is not None:
        with open(split_file) as f:
            data_idx_list = sorted([x.strip()
                                   for x in f.readlines() if len(x) > 1])
    else:
        data_idx_list = [x[:-4]
                         for x in os.listdir(lidar_dir) if x[-4:] == '.bin']
        data_idx_list = sorted(data_idx_list)

    if not os.path.isdir(planes_dir):
        os.makedirs(planes_dir, exist_ok=True)

    w_default = [0, -1, 0]
    h_default = 1.65
    for data_idx in data_idx_list:

        print('------------- ', data_idx)
        calib = calib_dir + '/' + data_idx + '.txt'
        calib = utils.Calibration(calib)
        pc_velo = lidar_dir + '/' + data_idx + '.bin'
        pc_velo = np.fromfile(pc_velo, dtype=np.float32).reshape(-1, 4)
        pc_rect = calib.project_velo_to_rect(pc_velo[:, :3])
        valid_loc = (pc_rect[:, 1] > min_h) & \
                    (pc_rect[:, 1] < max_h) & \
                    (pc_rect[:, 2] > -10) & \
                    (pc_rect[:, 2] < 70) & \
                    (pc_rect[:, 0] > -20) & \
                    (pc_rect[:, 0] < 20)
        pc_rect = pc_rect[valid_loc]
        if len(pc_rect) < 5:
            w = w_default
            h = h_default
            print("go defualt!!")
        else:
            reg = RANSACRegressor().fit(pc_rect[:, [0, 2]], pc_rect[:, 1])
            w = np.zeros(3)
            w[0] = reg.estimator_.coef_[0]
            w[2] = reg.estimator_.coef_[1]
            w[1] = -1.0
            h = reg.estimator_.intercept_
            norm = np.linalg.norm(w)
            w = w / norm
            h = h / norm
        # if h < 1.4 or h > 1.9 or abs(w[1]) < 0.98:
        #     w = w_default
        #     h = h_default
        # w_default = w
        # h_default = h
        print(w)
        print(h)

        lines = ['# Plane', 'Width 4', 'Height 1']

        plane_file = os.path.join(planes_dir, data_idx + '.txt')
        result_lines = lines[:3]
        result_lines.append("{:e} {:e} {:e} {:e}".format(w[0], w[1], w[2], h))
        result_str = '\n'.join(result_lines)
        with open(plane_file, 'w') as f:
            f.write(result_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib_dir', default='KITTI/object/training/calib')
    parser.add_argument(
        '--lidar_dir', default='KITTI/object/training/velodyne')
    parser.add_argument(
        '--planes_dir', default='KITTI/object/training/velodyne_planes')
    parser.add_argument('--min_h', type=float, default=1.5)
    parser.add_argument('--max_h', type=float, default=1.8)
    parser.add_argument('--split_file', type=str, default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.planes_dir):
        extract_ransac(args.calib_dir, args.lidar_dir,
                       args.planes_dir, args.min_h, args.max_h, args.split_file)
