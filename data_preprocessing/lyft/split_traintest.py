import argparse
import os.path as osp
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_list_file", type=str,
                        default="./meta_data/lyft_2019_train_sample_tracks.pkl")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--save_root", type=str, default='./meta_data/')

    parser.add_argument("--max_allow_dist", type=float, default=3.)
    parser.add_argument("--disable_only_forward", dest="only_forward",
                        action='store_false')
    parser.add_argument("--prefix", type=str, default="fw70_2m_")

    return parser.parse_args()


def main(args):
    track_list = pickle.load(open(args.track_list_file, "rb"))
    cutoff = 1700  # a simple cut off by the geo location
    train_track = []
    test_track = []
    print(f"Reading oxts from {osp.join(args.data_root, 'training/oxts')}")
    for seq_id, seq in enumerate(tqdm(track_list)):
        seq_oxts = []
        for i in seq:
            with open(osp.join(args.data_root, f"training/oxts/{i:06d}.txt"), "r") as f:
                seq_oxts.append(np.array([float(x)
                                for x in f.readline().split()]))
        seq_oxts = np.array(seq_oxts)
        if (seq_oxts[:, 1] >= cutoff).sum() == len(seq_oxts):
            test_track.append(seq)
        if (seq_oxts[:, 1] < cutoff).sum() == len(seq_oxts):
            train_track.append(seq)

    oxts_path = osp.join(args.data_root, "training/oxts")
    for name, track_list in zip(["train",],
                                [train_track,],):
        poses = []
        for seq in tqdm(track_list):
            poses.append([])
            for idx in seq:
                with open(osp.join(oxts_path, f"{idx:06d}.txt"), "r") as f:
                    info = np.array([float(x) for x in f.readline().split()])
                    trans = np.eye(4)
                    trans[:3, 3] = info[:3]
                    trans[:3, :3] = R.from_euler('xyz', info[3:]).as_matrix()
                    poses[-1].append(trans)

        loc_cache = {}
        for seq_id in range(len(track_list)):
            loc_cache[seq_id] = np.array(
                [pose[:2, 3] for pose in poses[seq_id]])

        # check how many valid scenes
        # dis_choice = [5, 10, 15, 20]
        dis_choice = np.arange(2, 71, 2)
        valid_idx = {}
        pbar = tqdm(track_list)
        for origin_seq_id, origin_seq in enumerate(pbar):
            for origin_frame in range(len(origin_seq)):
                origin_pose = poses[origin_seq_id][origin_frame]
                origin_idx = track_list[origin_seq_id][origin_frame]
                valid_seq = []
                for seq_id, seq in enumerate(track_list):
                    if seq_id == origin_seq_id:
                        continue
                    distance = np.linalg.norm(
                        loc_cache[seq_id] - origin_pose[:2, 3], axis=1)
                    min_dist_indices = np.argmin(distance)
                    min_dist = distance[min_dist_indices]
                    if min_dist > args.max_allow_dist:
                        continue
                    # pick samples
                    indices = [min_dist_indices]
                    if args.only_forward:
                        forward = origin_pose[0,:3] @ poses[seq_id][min_dist_indices][0, :3] > 0
                        for dis in dis_choice:
                            temp = np.where(distance > dis)[0]
                            if forward:
                                if len(temp[temp > min_dist_indices]) == 0:
                                    break
                                indices.append(
                                    temp[temp > min_dist_indices].min())
                            else:
                                if len(temp[temp < min_dist_indices]) == 0:
                                    break
                                indices.append(
                                    temp[temp < min_dist_indices].max())
                        if len(indices) < len(dis_choice) + 1:
                            continue
                    else:
                        for dis in dis_choice:
                            temp = np.where(distance > dis)[0]
                            if len(temp[temp < min_dist_indices]) == 0:
                                break
                            indices.append(temp[temp < min_dist_indices].max())
                            if len(temp[temp > min_dist_indices]) == 0:
                                break
                            indices.append(temp[temp > min_dist_indices].min())
                        if len(indices) < 2 * len(dis_choice) + 1:
                            continue
                    valid_seq.append((seq_id, indices))
                if len(valid_seq) > 1:
                    valid_idx[origin_idx] = (
                        origin_seq_id, origin_frame, valid_seq)
            pbar.set_postfix({'count': len(valid_idx)})
        print(f"#{name}: {len(valid_idx)}")
        pickle.dump(track_list, open(osp.join(args.save_root,
            f"{args.prefix}{name}_track_list.pkl"), "wb"))
        pickle.dump(valid_idx, open(osp.join(args.save_root,
            f"{args.prefix}valid_{name}_idx_info.pkl"), "wb"))
        with open(osp.join(args.save_root, f"{args.prefix}{name}_idx.txt"), 'w') as f:
            f.write("\n".join([f"{x:06d}" for x in valid_idx.keys()]))

    full_test_idx = [item for sublist in test_track for item in sublist]
    print(f"#test: {len(full_test_idx)}")
    with open(osp.join(args.save_root, f"{args.prefix}full_test_idx.txt"), 'w') as f:
            f.write("\n".join([f"{x:06d}" for x in full_test_idx]))

if __name__ == "__main__":
    main(parse_args())
