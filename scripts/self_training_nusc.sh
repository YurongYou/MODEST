#!/bin/bash
set -e

start_iter=1
max_iter=10
formatstring="dynamic_%s_round_0"
base_dataset="nuscenes_boston"
model="pointrcnn_dynamic_obj"
combine_arg=""

while getopts "M:S:F:b:C:m:" opt
do
    case $opt in
        M) max_iter=$OPTARG ;;
        S) start_iter=$OPTARG ;;
        F) formatstring=$OPTARG ;;
        b) base_dataset=$OPTARG ;;
        C) combine_arg=$OPTARG ;;
        m) model=$OPTARG ;;
        *)
            echo "there is unrecognized parameter."
            exit 1
            ;;
    esac
done

set -x
proj_root_dir=$(pwd)

for ((i = ${start_iter} ; i <= ${max_iter} ; i++)); do
    iter_name=$(printf $formatstring ${i})
    pre_iter_name=$(printf $formatstring $((i-1)))

    # check if the iteration has been finished
    if [ -f "${proj_root_dir}/downstream/OpenPCDet/output/nuscenes_boston_models/${model}/${iter_name}/eval/epoch_80/train/trainset_0/result.pkl" ]; then
        echo "${iter_name} has finished!"
        continue
    fi

    # generate the combined labels
    echo "=> Generating ${iter_name} labels"
    cd /home/yy785/projects/object_discovery/generate_cluster_mask
    python combine_labels.py fov_only=True data_paths=fw70_2m \
        det_result_path=${proj_root_dir}/downstream/OpenPCDet/output/nuscenes_boston_models/${model}/${pre_iter_name}/eval/epoch_80/train/trainset_0/result.pkl \
        save_path=/home/yy785/projects/object_discovery/generate_cluster_mask/intermediate_labels/nusc_${iter_name}_labels ${combine_arg}

    # create the dataset
    echo "=> Generating ${iter_name} dataset"
    cd ${proj_root_dir}/downstream/OpenPCDet/data
    mkdir nusc_${iter_name}
    cp -r ./${base_dataset}/training ./nusc_${iter_name}
    ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/ImageSets ./nusc_${iter_name}/
    ln -s ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/kitti_infos_val.pkl ./nusc_${iter_name}/
    cd ./nusc_${iter_name}/training
    if [ -L "label_2" ]; then
        rm label_2
    fi
    ln -s /home/yy785/projects/object_discovery/generate_cluster_mask/intermediate_labels/nusc_${iter_name}_labels label_2

    # run data pre-processing
    echo "=> pre-processing ${iter_name} dataset"
    cd ${proj_root_dir}/downstream/OpenPCDet
    python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/nuscenes_boston_dynamic_obj.yaml ../data/nusc_${iter_name}

    # start training
    echo "=> ${iter_name} training"
    cd ${proj_root_dir}/downstream/OpenPCDet/tools
    bash scripts/dist_train_tmp.sh 4 --cfg_file cfgs/nuscenes_boston_models/${model}.yaml \
        --extra_tag ${iter_name} --merge_all_iters_to_one_epoch \
        --fix_random_seed \
        --set DATA_CONFIG.DATA_PATH ../data/nusc_${iter_name}

    # generate the preditions on the training set
    bash scripts/dist_test_tmp.sh 4 --cfg_file cfgs/nuscenes_boston_models/${model}.yaml \
        --extra_tag ${iter_name} --eval_tag trainset_0 \
        --ckpt ../output/nuscenes_boston_models/${model}/${iter_name}/ckpt/checkpoint_epoch_80.pth \
        --set DATA_CONFIG.DATA_PATH ../data/nusc_${iter_name} \
        DATA_CONFIG.DATA_SPLIT.test train DATA_CONFIG.INFO_PATH.test kitti_infos_train.pkl
done
