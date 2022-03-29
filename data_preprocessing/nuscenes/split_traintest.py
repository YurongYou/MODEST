import argparse
import os.path as osp
import pickle
import numpy as np
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation as R

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mata_file_root", type=str,
                        default="./meta_data/")
    parser.add_argument("--data_20hz_root", type=str)
    parser.add_argument("--max_allow_dist", type=float, default=3.)
    parser.add_argument("--disable_only_forward", dest="only_forward",
                        action='store_false')
    parser.add_argument("--save_root", type=str, default='./meta_data/')
    parser.add_argument("--prefix", type=str, default="")
    return parser.parse_args()

def main(args):
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

    cutoff = 1500
    train_seq_id = set()
    val_seq_id = set()
    poses = []
    for seq_id, seq in enumerate(tqdm(track_list)):
        seq_oxts = []
        poses.append([])
        for i in seq:
            with open(osp.join(args.data_20hz_root, f"training/oxts/{i:06d}.txt"), "r") as f:
                info = np.array([float(x) for x in f.readline().split()])
                seq_oxts.append(info)
                trans = np.eye(4)
                trans[:3, 3] = info[:3]
                trans[:3, :3] = R.from_euler('xyz', info[3:]).as_matrix()
                poses[-1].append(trans)
        seq_oxts = np.array(seq_oxts)
        if (seq_oxts[:, 0] >= cutoff).sum() == len(seq_oxts):
            val_seq_id.add(seq_id)
        if (seq_oxts[:, 0] < cutoff).sum() == len(seq_oxts):
            train_seq_id.add(seq_id)

    loc_cache = {}
    for seq_id in range(len(track_list)):
        loc_cache[seq_id] = np.array([pose[:2, 3] for pose in poses[seq_id]])

    # check how many valid scenes
    dis_choice = np.linspace(0, 30, 16)[1:]
    # dis_choice = np.linspace(0, 20, 21)[1:]
    valid_idx = {}
    pbar = tqdm(labeledidx2seq_frame.items())
    for origin_idx, (origin_seq_id, origin_frame) in pbar:
        origin_pose = poses[origin_seq_id][origin_frame]

        seq_id_list = []
        if origin_seq_id in train_seq_id:
            seq_id_list = train_seq_id
        elif origin_seq_id in val_seq_id:
            seq_id_list = val_seq_id

        valid_seq = []
        for seq_id in (seq_id_list):
            seq = track_list[seq_id]
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
            # determine the direction
            if args.only_forward:
                forward = origin_pose[0,:3] @ poses[seq_id][min_dist_indices][0, :3] > 0
                for dis in dis_choice:
                    temp = np.where(distance > dis)[0]
                    if forward:
                        if len(temp[temp > min_dist_indices]) == 0:
                            break
                        indices.append(temp[temp > min_dist_indices].min())
                    else:
                        if len(temp[temp < min_dist_indices]) == 0:
                            break
                        indices.append(temp[temp < min_dist_indices].max())
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
            valid_idx[origin_idx] = (origin_seq_id, origin_frame, valid_seq)
        pbar.set_postfix({'count': len(valid_idx)})

    pickle.dump(valid_idx, open(
        osp.join(args.save_root, f"{args.prefix}valid_idx_info.pkl"), "wb"))
    pickle.dump(track_list, open(
        osp.join(args.save_root, f"{args.prefix}track_list.pkl"), "wb"))

    train_idx = [x for x in valid_idx.keys() if labeledidx2seq_frame[x]
                [0] in train_seq_id]
    test_idx = [x for x in valid_idx.keys() if labeledidx2seq_frame[x]
            [0] in val_seq_id]
    print(f"#train: {len(train_idx)}, #test: {len(test_idx)}")
    with open(osp.join(args.save_root, f"{args.prefix}train_idx.txt"), 'w') as f:
        f.write("\n".join([f"{x:06d}" for x in train_idx]))
    with open(osp.join(args.save_root, f"{args.prefix}test_idx.txt"), 'w') as f:
        f.write("\n".join([f"{x:06d}" for x in test_idx]))

if __name__ == "__main__":
    main(parse_args())
