# modified from https://github.com/cxy1997/3D_adapt_auto_driving/blob/master/convert/nusc2kitti.py
import os
import json
from typing import List, Dict, Any
import argparse
from pyquaternion import Quaternion
import numpy as np
from nuscenes.utils.geometry_utils import view_points
import fire
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed, parallel_backend
import pickle
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from nuscenes.utils.kitti import KittiDB
from nuscenes.eval.detection.utils import category_to_detection_name

from typing import cast


def find_closest_integer_in_ref_arr(query_int: int, ref_arr: np.ndarray):
    closest_ind = np.argmin(np.absolute(ref_arr - query_int))
    closest_int = cast(int, ref_arr[closest_ind])
    int_diff = np.absolute(query_int - closest_int)
    return closest_ind, closest_int, int_diff

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


CLASS_MAP = {
    "bicycle": "Dynamic",
    "bus": "Dynamic",
    "car": "Dynamic",
    "construction_vehicle": "Dynamic",
    "motorcycle": "Dynamic",
    "pedestrian": "Dynamic",
    "trailer": "Dynamic",
    "truck": "Dynamic"
}


def box_to_string(name, box, bbox_2d, truncation, occlusion, alpha, instance_token):
    v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
    yaw = -np.arctan2(v[2], v[0])

    # Prepare output.
    name += ' '
    trunc = '{:.2f} '.format(truncation)
    occ = '{:d} '.format(occlusion)
    a = '{:.2f} '.format(alpha)
    bb = '{:.2f} {:.2f} {:.2f} {:.2f} '.format(bbox_2d[0], bbox_2d[1], bbox_2d[2], bbox_2d[3])
    hwl = '{:.2} {:.2f} {:.2f} '.format(box.wlh[2], box.wlh[0], box.wlh[1])  # height, width, length.
    xyz = '{:.2f} {:.2f} {:.2f} '.format(box.center[0], box.center[1], box.center[2])  # x, y, z.
    y = '{:.2f}'.format(yaw)  # Yaw angle.

    output = name + trunc + occ + a + bb + hwl + xyz + y

    return output


def postprocessing(objs, height, width):
    _map = np.ones((height, width), dtype=np.int8) * -1
    objs = sorted(objs, key=lambda x: x["depth"], reverse=True)
    for i, obj in enumerate(objs):
        _map[int(round(obj["bbox_2d"][1])):int(round(obj["bbox_2d"][3])), int(round(obj["bbox_2d"][0])):int(round(obj["bbox_2d"][2]))] = i
    unique, counts = np.unique(_map, return_counts=True)
    counts = dict(zip(unique, counts))
    for i, obj in enumerate(objs):
        if i not in counts.keys():
            counts[i] = 0
        occlusion = 1.0 - counts[i] / (obj["bbox_2d"][3] - obj["bbox_2d"][1]) / (obj["bbox_2d"][2] - obj["bbox_2d"][0])
        obj["occluded"] = int(np.clip(occlusion * 4, 0, 3))
    return objs


def project_to_2d(box, p_left, height, width):
    box = box.copy()

    # KITTI defines the box center as the bottom center of the object.
    # We use the true center, so we need to adjust half height in negative y direction.
    box.translate(np.array([0, -box.wlh[2] / 2, 0]))

    # Check that some corners are inside the image.
    corners = np.array([corner for corner in box.corners().T if corner[2] > 0]).T
    if len(corners) == 0:
        return None

    # Project corners that are in front of the camera to 2d to get bbox in pixel coords.
    imcorners = view_points(corners, p_left, normalize=True)[:2]
    bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))


    inside = (0 <= bbox[1] < height and 0 < bbox[3] <= height) and (0 <= bbox[0] < width and 0 < bbox[2] <= width)
    valid = (0 <= bbox[1] < height or 0 < bbox[3] <= height) and (0 <= bbox[0] < width or 0 < bbox[2] <= width)
    if not valid:
        return None

    truncated = valid and not inside
    if truncated:
        _bbox = [0] * 4
        _bbox[0] = max(0, bbox[0])
        _bbox[1] = max(0, bbox[1])
        _bbox[2] = min(width, bbox[2])
        _bbox[3] = min(height, bbox[3])

        truncated = 1.0 - ((_bbox[2] - _bbox[0]) * (_bbox[3] - _bbox[1])) / ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        bbox = _bbox
    else:
        truncated = 0.0
    return {"bbox": bbox, "truncated": truncated}


def form_trans_mat(translation, rotation):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = translation
    return mat


class KittiConverter:
    def __init__(self,
                 nusc_kitti_dir: str = '/media/xiangyu/wd01/dreaming/nuscenes-in-kitti-format-boston-train-track',
                 nusc_dir: str = '/media/xiangyu/wd01/driving/nuscenes/',
                 cam_name: str = 'CAM_FRONT',
                 lidar_name: str = 'LIDAR_TOP',
                 image_count: int = 10,
                 nusc_version: str = 'v1.0-trainval',
                 parallel_n_jobs: int = 8,
                 convert_labels: bool = False,
                 convert_images: bool = True
                 ):
        """
        :param nusc_kitti_dir: Where to write the KITTI-style annotations.
        :param cam_name: Name of the camera to export. Note that only one camera is allowed in KITTI.
        :param lidar_name: Name of the lidar sensor.
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param split: Dataset split to use.
        """
        self.nusc_kitti_dir = os.path.expanduser(nusc_kitti_dir)
        self.cam_name = cam_name
        self.lidar_name = lidar_name
        self.image_count = image_count
        self.nusc_version = nusc_version
        self.split = 'train'
        self.parallel_n_jobs = parallel_n_jobs
        self.convert_labels = convert_labels
        self.convert_images = convert_images

        # Create nusc_kitti_dir.
        if not os.path.isdir(self.nusc_kitti_dir):
            os.makedirs(self.nusc_kitti_dir)

        # Select subset of the data to look at.
        self.nusc = NuScenes(version=nusc_version, dataroot=nusc_dir)

        print("self.nusc.sample", len(self.nusc.sample))

    def nuscenes_gt_to_kitti(self) -> None:
        """
        Converts nuScenes GT annotations to KITTI format.
        """

        # Create output folders.
        self.label_folder = os.path.join(self.nusc_kitti_dir, 'training', 'label_2')
        self.calib_folder = os.path.join(self.nusc_kitti_dir, 'training', 'calib')
        self.image_folder = os.path.join(self.nusc_kitti_dir, 'training', 'image_2')
        self.lidar_folder = os.path.join(self.nusc_kitti_dir, 'training', 'velodyne')
        self.oxts_folder = os.path.join(self.nusc_kitti_dir, 'training', 'oxts')
        self.l2e_folder = os.path.join(self.nusc_kitti_dir, 'training', 'l2e')
        for folder in [self.label_folder, self.oxts_folder, self.calib_folder, self.image_folder, self.lidar_folder, self.l2e_folder]:
            if not os.path.isdir(folder):
                if folder == self.label_folder and not self.convert_labels:
                    continue
                if folder == self.image_folder and not self.convert_images:
                    continue
                os.makedirs(folder)

        # Use only the samples from the current split.

        scene_splits = create_splits_scenes(verbose=False)
        scene_to_log = {scene['name']: (scene,
                                        self.nusc.get('log', scene['log_token'])['logfile'],
                                        self.nusc.get('log', scene['log_token'])['location']) for scene in self.nusc.scene}
        scenes = scene_splits[self.split]
        boston_scenes = []
        for scene in scenes:
            if scene_to_log[scene][2].startswith('boston'):
                boston_scenes.append(scene_to_log[scene][0])
        np.random.seed(1024)
        np.random.shuffle(boston_scenes)
        if self.convert_labels:
            scene_tokens = [scene['token'] for scene in boston_scenes]
            sample_tokens, sequence_mapping = self._split_to_samples_annotated(
                scene_tokens)
            lidar_tokens = [ld_token for ld_token, _, _ in sample_tokens]
            pickle.dump(lidar_tokens, open(os.path.join(self.nusc_kitti_dir,
                                              'training',
                                              "labeled_tokens.pkl"), "wb"))
            pickle.dump(sequence_mapping, open(os.path.join(self.nusc_kitti_dir,
                                              'training',
                                              "sequence_mapping.pkl"), "wb"))
            pickle.dump(scene_tokens, open(os.path.join(self.nusc_kitti_dir,
                                              'training',
                                              "scenes_tokens.pkl"), "wb"))
        else:
            sample_tokens, track_list = self._split_to_samples(
                boston_scenes)
            lidar_tokens = [ld_token for ld_token, _ in sample_tokens]
            pickle.dump((track_list, lidar_tokens), open(os.path.join(
                self.nusc_kitti_dir, 'training', 'track_infos.pkl'), 'wb'))

        # return
        print(len(sample_tokens))
        if self.convert_labels:
            with parallel_backend("threading", n_jobs=self.parallel_n_jobs):
                Parallel()(delayed(self.process_token_to_kitti)(idx, lidar_token, cam_front_token, sample_annotation_tokens)
                        for idx, (lidar_token, cam_front_token, sample_annotation_tokens) in tqdm(list(enumerate(sample_tokens))))
        else:
            with parallel_backend("threading", n_jobs=self.parallel_n_jobs):
                Parallel()(delayed(self.process_token_to_kitti)(idx, lidar_token, cam_front_token)
                        for idx, (lidar_token, cam_front_token) in tqdm(list(enumerate(sample_tokens))))

    def process_token_to_kitti(self, idx, lidar_token, cam_front_token,
                               sample_annotation_tokens=None):
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        # Get sample data.
        sample_name = "%06d" % idx

        # Retrieve sensor records.
        sd_record_cam = self.nusc.get('sample_data', cam_front_token)
        sd_record_lid = self.nusc.get('sample_data', lidar_token)
        cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
        cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

        ego_record_lid = self.nusc.get(
            "ego_pose", sd_record_lid["ego_pose_token"])

        l2e_path = os.path.join(self.l2e_folder, f"{sample_name}")
        l2e = form_trans_mat(cs_record_lid['translation'], cs_record_lid['rotation'])
        np.save(l2e_path, l2e)

        oxts_path = os.path.join(self.oxts_folder, f"{sample_name}.txt")
        with open(oxts_path, "w") as f:
            rotation = ego_record_lid['rotation'][1:] + [ego_record_lid['rotation'][0]]
            oxts_info = [x for x in ego_record_lid['translation']] + \
                [x for x in R.from_quat(rotation).as_euler('xyz')]
            oxts_info = [str(x) for x in oxts_info]
            f.write(" ".join(oxts_info))


        # Combine transformations and convert to KITTI format.
        # Note: cam uses same conventions in KITTI and nuScenes.
        lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                        inverse=False)
        ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                        inverse=True)
        velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

        # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
        velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

        # Currently not used.
        # imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
        imu_to_velo_kitti =  transform_matrix(
                cs_record_lid["translation"], Quaternion(cs_record_lid["rotation"]), inverse=True
            )
        imu_to_velo_kitti = np.dot(kitti_to_nu_lidar_inv.transformation_matrix, imu_to_velo_kitti)
        expected_kitti_imu_to_velo_rot = np.eye(3)
        assert (imu_to_velo_kitti[:3, :3].round(0) == expected_kitti_imu_to_velo_rot).all(), imu_to_velo_kitti[:3, :3].round(0)
        r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

        # Projection matrix.
        p_left_kitti = np.zeros((3, 4))
        p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

        # Create KITTI style transforms.
        velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
        velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

        # Check that the rotation has the same format as in KITTI.
        assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
        assert (velo_to_cam_trans[1:3] < 0).all()

        # Retrieve the token from the lidar.
        # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
        # not the camera.
        filename_cam_full = sd_record_cam['filename']
        filename_lid_full = sd_record_lid['filename']

        # Create calibration file.
        kitti_transforms = dict()
        kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
        kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
        kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
        kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
        calib_path = os.path.join(self.calib_folder, sample_name + '.txt')
        with open(calib_path, "w") as calib_file:
            for (key, val) in kitti_transforms.items():
                val = val.flatten()
                val_str = '%.12e' % val[0]
                for v in val[1:]:
                    val_str += ' %.12e' % v
                calib_file.write('%s: %s\n' % (key, val_str))

        if self.convert_images:
            # Convert image (jpg to png).
            src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
            dst_im_path = os.path.join(self.image_folder, sample_name + '.png')
            if not os.path.exists(dst_im_path):
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")

        # Convert lidar.
        # Note that we are only using a single sweep, instead of the commonly used n sweeps.
        src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
        dst_lid_path = os.path.join(self.lidar_folder, sample_name + '.bin')
        assert not dst_lid_path.endswith('.pcd.bin')
        pcl = LidarPointCloud.from_file(src_lid_path)
        # pcl, _ = LidarPointCloud.from_file_multisweep_future(self.nusc, sample, self.lidar_name, self.lidar_name, nsweeps=5)
        pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
        with open(dst_lid_path, "w") as lid_file:
            pcl.points.T.tofile(lid_file)


        # # Write label file.
        if self.convert_labels:
            # sample_annotation_tokens = sample['anns']
            label_path = os.path.join(self.label_folder, sample_name + '.txt')
            if os.path.exists(label_path):
                print('Skipping existing file: %s' % label_path)
                return

            objects = []
            for sample_annotation_token in sample_annotation_tokens:
                sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)

                # Get box in LIDAR frame.
                _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                        selected_anntokens=[sample_annotation_token])
                box_lidar_nusc = box_lidar_nusc[0]

                obj = dict()

                # Convert nuScenes category to nuScenes detection challenge category.
                obj["detection_name"] = category_to_detection_name(sample_annotation['category_name'])

                # Skip categories that are not part of the nuScenes detection challenge.
                if obj["detection_name"] is None or obj["detection_name"] not in CLASS_MAP.keys():
                    continue

                obj["detection_name"] = CLASS_MAP[obj["detection_name"]]

                # Convert from nuScenes to KITTI box format.
                obj["box_cam_kitti"] = KittiDB.box_nuscenes_to_kitti(
                    box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                bbox_2d = project_to_2d(obj["box_cam_kitti"], p_left_kitti, imsize[1], imsize[0])
                if bbox_2d is None:
                    continue
                obj["bbox_2d"] = bbox_2d["bbox"]
                obj["truncated"] = bbox_2d["truncated"]

                # Set dummy score so we can use this file as result.
                obj["box_cam_kitti"].score = 0

                v = np.dot(obj["box_cam_kitti"].rotation_matrix, np.array([1, 0, 0]))
                rot_y = -np.arctan2(v[2], v[0])
                obj["alpha"] = -np.arctan2(obj["box_cam_kitti"].center[0], obj["box_cam_kitti"].center[2]) + rot_y
                obj["depth"] = np.linalg.norm(np.array(obj["box_cam_kitti"].center[:3]))
                obj["instance_token"] = sample_annotation['instance_token']
                objects.append(obj)

            objects = postprocessing(objects, imsize[1], imsize[0])

            with open(label_path, "w") as label_file:
                for obj in objects:
                    # Convert box to output string format.
                    output = box_to_string(name=obj["detection_name"],
                                            box=obj["box_cam_kitti"],
                                            bbox_2d=obj["bbox_2d"],
                                            truncation=obj["truncated"],
                                            occlusion=obj["occluded"],
                                            alpha=obj["alpha"],
                                            instance_token=obj['instance_token'])
                    label_file.write(output + '\n')


    def render_kitti(self, render_2d: bool) -> None:
        """
        Renders the annotations in the KITTI dataset from a lidar and a camera view.
        :param render_2d: Whether to render 2d boxes (only works for camera data).
        """
        if render_2d:
            print('Rendering 2d boxes from KITTI format')
        else:
            print('Rendering 3d boxes projected from 3d KITTI format')

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_dir, splits=(self.split,))

        # Create output folder.
        render_dir = os.path.join(self.nusc_kitti_dir, 'render')
        if not os.path.isdir(render_dir):
            os.mkdir(render_dir)

        # Render each image.
        for token in kitti.tokens[:self.image_count]:
            for sensor in ['lidar', 'camera']:
                out_path = os.path.join(render_dir, '%s_%s.png' % (token, sensor))
                print('Rendering file to disk: %s' % out_path)
                kitti.render_sample_data(token, sensor_modality=sensor, out_path=out_path, render_2d=render_2d)
                plt.close()  # Close the windows to avoid a warning of too many open windows.

    def kitti_res_to_nuscenes(self, meta: Dict[str, bool] = None) -> None:
        """
        Converts a KITTI detection result to the nuScenes detection results format.
        :param meta: Meta data describing the method used to generate the result. See nuscenes.org/object-detection.
        """
        # Dummy meta data, please adjust accordingly.
        if meta is None:
            meta = {
                'use_camera': False,
                'use_lidar': True,
                'use_radar': False,
                'use_map': False,
                'use_external': False,
            }

        # Init.
        results = {}

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_dir, splits=(self.split, ))

        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(self.split, self.nusc)

        # Use only the samples from the current split.
        sample_tokens = self._split_to_samples(split_logs)
        sample_tokens = sample_tokens[:self.image_count]

        for sample_token in sample_tokens:
            # Get the KITTI boxes we just generated in LIDAR frame.
            kitti_token = '%s_%s' % (self.split, sample_token)
            boxes = kitti.get_boxes(token=kitti_token)

            # Convert KITTI boxes to nuScenes detection challenge result format.
            sample_results = [self._box_to_sample_result(sample_token, box) for box in boxes]

            # Store all results for this image.
            results[sample_token] = sample_results

        # Store submission file to disk.
        submission = {
            'meta': meta,
            'results': results
        }
        submission_path = os.path.join(self.nusc_kitti_dir, 'submission.json')
        print('Writing submission to: %s' % submission_path)
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)

    def _box_to_sample_result(self, sample_token: str, box: Box, attribute_name: str = '') -> Dict[str, Any]:
        # Prepare data
        translation = box.center
        size = box.wlh
        rotation = box.orientation.q
        velocity = box.velocity
        detection_name = box.name
        detection_score = box.score

        # Create result dict
        sample_result = dict()
        sample_result['sample_token'] = sample_token
        sample_result['translation'] = translation.tolist()
        sample_result['size'] = size.tolist()
        sample_result['rotation'] = rotation.tolist()
        sample_result['velocity'] = velocity.tolist()[:2]  # Only need vx, vy.
        sample_result['detection_name'] = detection_name
        sample_result['detection_score'] = detection_score
        sample_result['attribute_name'] = attribute_name

        return sample_result

    def _split_to_samples(self, split_tokens: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        samples = []
        track_list = []
        # time_diff = []
        # timestamps = []
        cnt = 0
        for scene in split_tokens:
            track_list.append([])
            first_sample_token = scene['first_sample_token']
            first_sample = self.nusc.get("sample", first_sample_token)
            ld_token = first_sample['data'][self.lidar_name]
            ld_tokens = []
            ld_timestamp = []
            while(ld_token != ''):
                ld_tokens.append(ld_token)
                sample_data = self.nusc.get("sample_data", ld_token)
                ld_timestamp.append(sample_data['timestamp'])
                ld_token = sample_data['next']
            ld_tokens = np.array(ld_tokens)
            ld_timestamp = np.array(ld_timestamp)

            cam_tokens = []
            cam_timestamp = []
            cam_token = first_sample['data'][self.cam_name]
            while(cam_token != ''):
                cam_tokens.append(cam_token)
                sample_data = self.nusc.get("sample_data", cam_token)
                cam_timestamp.append(sample_data['timestamp'])
                cam_token = sample_data['next']
            cam_timestamp = np.array(cam_timestamp)
            closest_cam_tokens = []
            for i in range(ld_tokens.shape[0]):
                closest_ind, closest_int, int_diff = find_closest_integer_in_ref_arr(
                    ld_timestamp[i], cam_timestamp)
                closest_cam_tokens.append(cam_tokens[closest_ind])
            samples += list(zip(ld_tokens, closest_cam_tokens))
            for _ in ld_tokens:
                track_list[-1].append(cnt)
                cnt += 1
        return samples, track_list

    def _split_to_samples_annotated(self, split_tokens: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        samples = []
        cnt = 0
        sequence_mapping = dict()
        for sample in self.nusc.sample:
            if sample['scene_token'] not in split_tokens:
                continue
            lidar_token = sample['data'][self.lidar_name]
            cam_front_token = sample['data'][self.cam_name]
            samples.append([lidar_token, cam_front_token])
            if self.convert_labels:
                samples[-1].append(sample['anns'])
            if sample["scene_token"] in sequence_mapping:
                sequence_mapping[sample["scene_token"]].append(cnt)
            else:
                sequence_mapping[sample["scene_token"]] = [cnt]
            cnt += 1
        return samples, sequence_mapping

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nusc_dir', type=str, default='/media/xiangyu/wd01/driving/nuscenes/')
    parser.add_argument(
        '--nusc_kitti_dir', default='/media/xiangyu/wd01/dreaming/nuscenes-in-kitti-format-boston-train-track', type=str)
    parser.add_argument(
        '--parallel_n_jobs', default=8, type=int)
    parser.add_argument(
        '--disable_convert_labels', dest='convert_labels', action='store_false')
    parser.add_argument(
        '--disable_convert_images', dest='convert_images', action='store_false')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    kc = KittiConverter(
        nusc_dir=args.nusc_dir,
        nusc_kitti_dir=args.nusc_kitti_dir,
        parallel_n_jobs=args.parallel_n_jobs,
        convert_labels=args.convert_labels,
        convert_images=args.convert_images)
    kc.nuscenes_gt_to_kitti()
