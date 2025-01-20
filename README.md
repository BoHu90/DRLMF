# DRLMF
Source codes for paper "[Blind Quality Assessment of Wide-angle Videos Based on Deformation Representation Learning and Multi-dimensional Feature Fusion]

![image](https://github.com/BoHu90/DRLMF/blob/main/frame.png)

## Usages

### Download databases
[MWV](https://github.com/BoHu90/MWV)
[LSVQ](https://github.com/baidut/PatchVQ)
[KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html)
[CVD2014](https://qualinet.github.io/databases/video/cvd2014_video_database/)
[Live-vqc](https://live.ece.utexas.edu/research/LIVEVQC/index.html)

### Test the model
You can download the trained model via [Google Drive](https://drive.google.com/drive/my-drive?dmr=1&ec=wgc-drive-globalnav-goto).
### Training on VQA databases

1. Extract frames from a video.
```
python ./extract_frame.py
```

2. Crop video frames.
```
python ./Split_frame.py
```

3. training on MWV and other datasets
```
python ./DRLMF_*.py
```
