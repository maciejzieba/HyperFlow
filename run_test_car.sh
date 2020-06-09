#! /bin/bash

python test.py \
--cates car \
--resume_checkpoint pretrained_model/car/checkpoint-car.pt \
--hyper_dims 256-512-1024 \
--dims 64-16-64 \
--latent_dims 256-256 \
--gpu 0 \
--batch_size 50 \
--use_latent_flow \
--use_sphere_dist \

echo "Done"
exit 0
