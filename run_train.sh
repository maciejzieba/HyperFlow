#! /bin/bash

python train.py \
--log_name experiment_airplanes_flow_bigger_sphere_1_00001 \
--lr 2e-3 \
--dataset_type shapenet15k \
--data_dir /home/datasets/ShapeNetCore.v2.PC15k \
--cates airplane \
--dims 64-16-64 \
--hyper_dims 256-512-1024 \
--latent_dims 256-256 \
--num_blocks 1 \
--latent_num_blocks 1 \
--batch_size 50 \
--zdim 128 \
--epochs 4000 \
--save_freq 50 \
--viz_freq 50 \
--log_freq 1 \
--val_freq 6000 \
--gpu 0 \
--batch_norm True \
--use_latent \
--use_sphere_dist \
--start_var 1.0 \
--stop_var 0.001 \
--step_var 0.001 \

echo "Done"
exit 0
