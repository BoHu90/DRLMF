 CUDA_VISIBLE_DEVICES=0 python -u DRLMF.py \
 --database MWV \
 --model_name DRLMF \
 --conv_base_lr 0.00001 \
 --epochs 100 \
 --train_batch_size 8 \
 --print_samples 1000 \
 --num_workers 6 \
 --ckpt_path ckpts \
 --decay_ratio 0.95 \
 --decay_interval 2 \
 --exp_version 0 \
 --loss_type plcc \
 --resize 256 \
 --crop_size 224 \
 >> logs/train_DRLMF_plcc_resize_256_crop_size_224_exp_version_0.log
