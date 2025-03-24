"""Extract features for temporal action detection datasets"""
import argparse
import os
import random
import cv2

import numpy as np
import pandas as pd
import torch
from timm.models import create_model
from torchvision import transforms

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
from dataset.loader import get_video_loader


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--data_set',
        default='TEST_VIDEO',
        choices=['THUMOS14', 'FINEACTION', 'TEST_VIDEO'],
        type=str,
        help='dataset')

    parser.add_argument(
        '--data_path',
        default='/data/dataset/YoutubeYGC',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--save_path',
        default='/data/user/gbb/SimpleVQA-main/videomeav2_youtube_ugc_feature',
        type=str,
        help='path for saving features')

    parser.add_argument(
        '--model',
        default='vit_giant_patch14_224',
        type=str,
        metavar='MODEL',
        help='Name of model')
    parser.add_argument(
        '--ckpt_path',
        default='/data/user/gbb/SimpleVQA-main/VideoMAEv2-master/vit_g_hybrid_pt_1200e_k710_ft.pth',
        help='load from checkpoint')

    return parser.parse_args()


def get_start_idx_range(data_set):
    def thumos14_range(num_frames):
        return range(0, num_frames - 15, 4)

    def fineaction_range(num_frames):
        return range(0, num_frames - 15, 16)

    def test_video_range(num_frames, video_frame_rate):
        return range(0, num_frames - 15, video_frame_rate)  # 有的视频帧率会小于16，导致取最后一段clip时会导致越界

    if data_set == 'THUMOS14':
        return thumos14_range
    elif data_set == 'FINEACTION':
        return fineaction_range
    elif data_set == 'TEST_VIDEO':
        return test_video_range  # todo 每16帧作为一个clip，把两个相邻的clip合起来作为一个feature(对应slowfast中的32帧)，总共8个feature
    else:
        raise NotImplementedError()

def adjust_array(arr):
    # 获取当前数组的第一维（行数）和第二维（列数）
    current_rows, cols = arr.shape
    if current_rows < 10:
        # 如果行数小于 8，用最后一行进行复制，直到补足 8 行
        last_row = arr[-1]  # 获取最后一行
        rows_to_add = 10 - current_rows  # 需要补充的行数
        arr = np.vstack([arr, np.tile(last_row, (rows_to_add, 1))])  # 复制最后一行并堆叠
    elif current_rows > 10:
        # 如果行数大于 8，则截取前 8 行
        arr = arr[:10, :]
    return arr

def extract_feature(args):
    dataset = 'youtube_ugc_train'
    datainfo_train = None
    if dataset == 'wide_angle_video_train':
        datainfo_train = '/data/dataset/MMVAV_csv/MMWAV_train.csv'
        column_names = ['Video Name','Deformation','Shake','Blur','Exposure','MOS','Width','Height','Frame']
    elif dataset == 'wide_angle_video_test':
        datainfo_train = '/data/dataset/MMVAV_csv/MMWAV_test.csv'
        column_names = ['Video Name', 'Deformation', 'Shake', 'Blur', 'Exposure', 'MOS', 'Width', 'Height', 'Frame']
    elif dataset == 'LSVQ_test':
        datainfo_train = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test.csv'
        column_names = ['name', 'p1', 'p2', 'p3',
                        'height', 'width', 'mos_p1',
                        'mos_p2', 'mos_p3', 'mos',
                        'frame_number', 'fn_last_frame', 'left_p1',
                        'right_p1', 'top_p1', 'bottom_p1',
                        'start_p1', 'end_p1', 'left_p2',
                        'right_p2', 'top_p2', 'bottom_p2',
                        'start_p2', 'end_p2', 'left_p3',
                        'right_p3', 'top_p3', 'bottom_p3',
                        'start_p3', 'end_p3', 'top_vid',
                        'left_vid', 'bottom_vid', 'right_vid',
                        'start_vid', 'end_vid', 'is_test', 'is_valid']
    elif dataset == 'LSVQ_test_1080p':
        datainfo_train = '/data/user/zhaoyimeng/ModularBVQA/data/LSVQ_whole_test_1080p.csv'
        column_names = ['name', 'p1', 'p2', 'p3',
                        'height', 'width', 'mos_p1',
                        'mos_p2', 'mos_p3', 'mos',
                        'frame_number', 'fn_last_frame', 'left_p1',
                        'right_p1', 'top_p1', 'bottom_p1',
                        'start_p1', 'end_p1', 'left_p2',
                        'right_p2', 'top_p2', 'bottom_p2',
                        'start_p2', 'end_p2', 'left_p3',
                        'right_p3', 'top_p3', 'bottom_p3',
                        'start_p3', 'end_p3', 'top_vid',
                        'left_vid', 'bottom_vid', 'right_vid',
                        'start_vid', 'end_vid', 'is_valid']

    dataInfo = pd.read_csv(datainfo_train, header=0, sep=',', names=column_names, index_col=False,
                           encoding="utf-8-sig")
    video_names = dataInfo['video_names']




    # preparation
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    video_loader = get_video_loader()


    transform = transforms.Compose(
        [ToFloatTensorInZeroOne(),
         Resize((224, 224))])


    # get video path
    # vid_list = [args.data_path + name for name in video_names]
    vid_list = video_names

    # vid_list = os.listdir(args.data_path)
    # random.shuffle(vid_list)

    # get model & load ckpt
    model = create_model(
        args.model,
        img_size=224,
        pretrained=False,
        num_classes=710,
        all_frames=16,
        tubelet_size=2,
        drop_path_rate=0.3,
        use_mean_pooling=True)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
    model.load_state_dict(ckpt)
    model.eval()
    model.to(device)

    # extract feature
    num_videos = len(vid_list)
    for idx, vid_name in enumerate(vid_list):
        # url = os.path.join(args.save_path, vid_name.split('.')[0] + '.npy')
        if dataset == 'wide_angle_video' or dataset == 'LSVQ_train' or dataset == 'LSVQ_test' or dataset == 'LSVQ_test_1080p' or dataset == 'youtube_ugc_train' or dataset == 'youtube_ugc_test':
            url = os.path.join(args.save_path, vid_name + '.npy')
            if not os.path.exists(args.save_path + '/' + vid_name.split('/')[0]):
                os.makedirs(args.save_path + '/' + vid_name.split('/')[0])
            vid_name = vid_name + '.mp4'
        else:
            url = os.path.join(args.save_path, vid_name[0:-4] + '.npy')
        if os.path.exists(url):
            continue

        video_path = os.path.join(args.data_path, vid_name)
        print(video_path)
        vr = video_loader(video_path)

        cap = cv2.VideoCapture(video_path)
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        start_idx_range = get_start_idx_range(args.data_set)

        feature_list = []
        for start_idx in start_idx_range(len(vr), video_frame_rate):
            data = vr.get_batch(np.arange(start_idx, start_idx + 16)).asnumpy()
            frame = torch.from_numpy(data)  # torch.Size([16, 566, 320, 3])
            frame_q = transform(frame)  # torch.Size([3, 16, 224, 224])
            input_data = frame_q.unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model.forward_features(input_data)
                feature_list.append(feature.cpu().numpy())

        # [N, C]
        feature_list_numpy = np.vstack(feature_list)

        feature_list_numpy_adjust = adjust_array(feature_list_numpy)

        print(f'feature_list_numpy_adjust shape: {feature_list_numpy_adjust.shape}')
        np.save(url, feature_list_numpy_adjust)

        print(f'[{idx + 1} / {num_videos}]: save feature on {url}')


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args = get_args()
    extract_feature(args)
