# DRLMF
Source codes for paper "[Blind Quality Assessment of Wide-angle Videos Based on Deformation Representation Learning and Multi-dimensional Feature Fusion]

![image](https://github.com/BoHu90/DRLMF/blob/main/frame.png)

## Usages
### Training on VQA databases

1. Extract frames from a video.
```
python ./extract_frame.py
```
2.Crop video frames.
```
python ./Split_frame.py
```
3.training on MWV and other datasets
```
python ./DRLMF_*.py
```
