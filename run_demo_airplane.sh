#! /bin/bash

python demo.py \
--cates airplane \
--resume_checkpoint pretrained_model/demo/checkpoint-airplane.pt \
--hyper_dims 256-512-1024 \
--dims 64-16-64 \
--latent_dims 256-256 \
--gpu 0 \
--batch_size 1 \
--use_latent_flow \
--use_sphere_dist \
--demo_mode 0 \

echo "Done"
exit 0
