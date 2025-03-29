# DRLMF
Source codes for paper "[Blind Quality Assessment of Wide-angle Videos Based on Deformation Representation Learning and Multi-dimensional Feature Fusion]

![image](https://github.com/BoHu90/DRLMF/blob/main/DRLMF_frame.png)

## Usages

### Install Requirements
```
pytorch
opencv
scipy
numpy
pandas
matplotlib
torchvision
torchvideo
```

### Download databases
[MWV](https://github.com/BoHu90/MWV)

[LSVQ](https://github.com/baidut/PatchVQ)

[KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html)

[CVD2014](https://qualinet.github.io/databases/video/cvd2014_video_database/)

[Live-vqc](https://live.ece.utexas.edu/research/LIVEVQC/index.html)

### Test the model
You can download the trained model via [Baidu Drive](https://pan.baidu.com/s/1oNr0UzcS3tD5SJTksTij1Q?pwd=0421).

If you want to perform inference/testing with our model, you can first download the trained weights and modify the path, then run the following script.
```
python -u test.py >> logs/test.log
```

### Training on VQA databases

1. Extract frames from a video.
```
python -u extract_frame.py >> logs/extract_frame.log
```

2. Crop video frames.
```
python -u Split_frame.py >> logs/Split_frame.log
```

3. Extract motion feature from VideoMAEv2.
   Path: DRLMF/VideoMAEv2-master/extract_tad_feature.py
```
python -u extract_tad_feature.py >> logs/extract_tad_feature.log
```

4. Training on MWV and other datasets.
```
python ./DRLMF_*.py
```
```
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
```

### Acknowledgement
The basic code is partially from the below repos.
- [ModularVQA](https://github.com/winwinwenwen77/ModularBVQA)
- [KSVQE](https://lixinustc.github.io/projects/KVQ/)
- [SimpleVQA](https://github.com/sunwei925/SimpleVQA)
- [VideoMae V2](https://github.com/OpenGVLab/VideoMAEv2)
