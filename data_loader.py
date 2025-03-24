import os
import random
import csv
import pandas as pd
from PIL import Image
import argparse
import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
import cv2
from torchvision import transforms

class VideoDataset_images_Aes_features(data.Dataset):

    def __init__(self, data_dir, filename_path, transform, database_name, crop_size):
        super(VideoDataset_images_Aes_features, self).__init__()

        if database_name == 'KoNViD-1k':
            dataInfo = scio.loadmat(filename_path)  #将Scipy中的matlab格式的数据文件转化为python中的数据结构
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            self.video_names = video_names
            self.score = score

        elif database_name == 'youtube_ugc':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][0][i])
            self.video_names = video_names
            self.score = score

        elif database_name == 'LSVQ_train':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test_1080p':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name == 'KoNViD-1k' \
                or self.database_name == 'youtube_ugc':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]  # 去除4个字符的拓展名
        elif self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test' or self.database_name == 'LSVQ_test_1080p':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]  # 获取文件名中字符串部分

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))  # 将numpy数组转换为pytorch形式的floattensor
        path_name = os.path.join(self.videos_dir, video_name_str)  # 连接两个文件名

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        if self.database_name == 'KoNViD-1k' or self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test' or self.database_name == 'LSVQ_test_1080p':
            video_length_read = 8  # 16
        elif self.database_name == 'youtube_ugc':
            video_length_read = 20
        # video_length_read就是batch

        video_channel = 3
        transformed_video = torch.zeros(
            [video_length_read, video_channel, video_height_crop, video_width_crop])  # 8*3*448*448

        for i in range(video_length_read):  # 处理一个batch中的数据变成transform形式
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        return video_score, transformed_video, video_name_str



class AADB_images(data.Dataset):

    def __init__(self, data_dir, filename_path, transform, database_name, crop_size):
        super(AADB_images, self).__init__()
        if database_name == 'huyy':
            column_names = ['Image Name','Quality Score','Text']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['score'].tolist()

        """
        if database_name == 'AADB':
            column_names = [ 'ImageFile','BalacingElements','ColorHarmony',\
                           'Content','DoF','Light','MotionBlur','Object','Repetition',\
                             'RuleOfThirds','Symmetry','VividColor','score']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.AADBImage_names = dataInfo['ImageFile'].tolist()
            self.score = dataInfo['score'].tolist()
            self.object_emphasis = dataInfo['Object'].tolist()
            self.vivid_color = dataInfo['VividColor'].tolist()
            self.good_light = dataInfo['Light'].tolist()
            self.color_harmony = dataInfo['ColorHarmony'].tolist()
            self.Interesing_content = dataInfo['Content'].tolist()
            self.shallow_depth_of_field = dataInfo['DoF'].tolist()"""

        self.crop_size = crop_size
        self.Image_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        if self.database_name == 'AADB' :
            AADBImage_name = self.AADBImage_names[idx]
            AADBImage_name_str = AADBImage_name[:-4]   #去除4个字符的拓展名

        AADBImage_score = torch.FloatTensor(np.array(float(self.score[idx])))
        AADBImage_object_emphasis = torch.FloatTensor(np.array(float(self.object_emphasis[idx])))
        AADBImage_vivid_color = torch.FloatTensor(np.array(float(self.vivid_color[idx])))
        AADBImage_good_light = torch.FloatTensor(np.array(float(self.good_light[idx])))
        AADBImage_color_harmony = torch.FloatTensor(np.array(float(self.color_harmony[idx])))
        AADBImage_Interesing_content = torch.FloatTensor(np.array(float(self.Interesing_content[idx])))
        AADBImage_shallow_depth_of_field = torch.FloatTensor(np.array(float(self.shallow_depth_of_field[idx])))

        path_name = os.path.join(self.AADBImage_dir, AADBImage_name_str)"""
        """
        channel = 3

        AADBImage_height_crop = self.crop_size
        AADBImage_width_crop = self.crop_size
        
        transformed_AADBImage = torch.zeros(
            [channel, AADBImage_height_crop, AADBImage_width_crop])  # 8*3*448*448"""

        if self.database_name == 'huyy':
            Image_name = self.video_names[idx]
            Image_name_str = Image_name[:-4]

        Image_score = torch.FloatTensor(np.array(float(self.score[idx])))
        path_name = os.path.join(self.Image_dir, Image_name_str)

        imge_name = os.path.join(path_name + '.png')
        read_frame = Image.open(imge_name)
        read_frame = read_frame.convert('RGB')
        read_frame = self.transform(read_frame)
        transformed_Image = read_frame


        return  transformed_Image, Image_score


class VideoDataset_images_with_motion_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    #videos_dir, feature_dir, AADB_dir, datainfo_train, transformations_train,'LSVQ_train', config.crop_size, 'SlowFast'
    def __init__(self, data_dir, data_dir_3D, filename_path, transform, database_name, crop_size, feature_type):
        super(VideoDataset_images_with_motion_features, self).__init__()
        #获得数据集的名称和分数
        if database_name == 'KoNViD-1k':
            dataInfo = scio.loadmat(filename_path)  #将Scipy中的matlab格式的数据文件转化为python中的数据结构
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            self.video_names = video_names
            self.score = score

        elif database_name == 'youtube_ugc':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][0][i])
            self.video_names = video_names
            self.score = score

        elif database_name == 'LSVQ_train':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test_1080p':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name == 'KoNViD-1k' \
            or self.database_name == 'youtube_ugc' :
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]   #去除4个字符的拓展名
        elif self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test' or self.database_name == 'LSVQ_test_1080p':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]  #获取文件名中字符串部分

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))  #将numpy数组转换为pytorch形式的floattensor

        path_name = os.path.join(self.videos_dir, video_name_str)  #连接两个文件名 中间会自动加上/

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
       
        if self.database_name == 'KoNViD-1k' or self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test' or self.database_name == 'LSVQ_test_1080p':
            video_length_read = 8  #16
        elif self.database_name == 'youtube_ugc':
            video_length_read = 20
        #video_length_read就是batch
        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])        #8*3*448*448


        for i in range(video_length_read):   #处理一个batch中的数据变成transform形式
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # read 3D features  读取slowfast的3D特征
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                i_index = i   
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048+256])    #8*(2048+256)
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D


        return transformed_video, transformed_feature,video_score, video_name
        #返回一个视频的空间和时间特征


class VideoDataset_images_VQA_dataset_with_motion_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D ,filename_path, transform, database_name, crop_size, feature_type, exp_id, state = 'train'):
        super(VideoDataset_images_VQA_dataset_with_motion_features, self).__init__()

        if database_name == 'KoNViD-1k':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][exp_id]
            if state == 'train':
                index = index_all[:int(n*0.8)]
            elif state == 'val':
                index = index_all[int(n*0.8):]

            for i in index:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            self.video_names = video_names
            self.score = score

        elif database_name == 'youtube_ugc':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][exp_id]
            if state == 'train':
                index = index_all[:int(n*0.8)]
            elif state == 'val':
                index = index_all[int(n*0.8):]

            for i in index:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][0][i])
            self.video_names = video_names
            self.score = score


        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
       
        if self.database_name == 'KoNViD-1k':
            video_length_read = 8
        elif self.database_name == 'youtube_ugc':
            video_length_read = 10

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])             


        for i in range(video_length_read):
            if self.database_name == 'youtube_ugc':
                imge_name = os.path.join(path_name, '{:03d}'.format(i*2) + '.png')
            else:
                imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # read 3D features
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                if self.database_name == 'KoNViD-1k':
                    i_index = i
                elif self.database_name == 'youtube_ugc':
                    i_index = int(i/2)   
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                if self.database_name == 'KoNViD-1k':
                    i_index = i
                elif self.database_name == 'youtube_ugc':
                    i_index = int(i/2) 
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048+256])
            for i in range(video_length_read):
                if self.database_name == 'KoNViD-1k':
                    i_index = i
                elif self.database_name == 'youtube_ugc':
                    i_index = int(i/2) 
                feature_3D_slow = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

       
        return transformed_video, transformed_feature, video_score, video_name


class VideoDataset_NR_LSVQ_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, filename_path, transform, resize, is_test_1080p = False):
        super(VideoDataset_NR_LSVQ_SlowFast_feature, self).__init__()
        if is_test_1080p:
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_valid']
        else:
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
                                        
        dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")

        self.video_names = dataInfo['name']
        self.score = dataInfo['mos']
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx])))/20   #将视频的评分缩到0-5

        filename=os.path.join(self.videos_dir, video_name + '.mp4')

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap=cv2.VideoCapture(filename)

        video_channel = 3
        
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))     #获得视频的总帧数
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))  #获得视频的帧率

        video_clip = int(video_length/video_frame_rate)     #获得整个视频的时间
       
        video_clip_min = 8

        video_length_clip = 32

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []
        
        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #将BRG格式图片转化为RGB格式
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1


        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]  #复制前一帧
 
        video_capture.release()  #释放占用的资源

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i*video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[i*video_frame_rate : (i*video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i*video_frame_rate)] = transformed_frame_all[i*video_frame_rate :]
                for j in range((video_length - i*video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i*video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])
       
        return transformed_video_all, video_score, video_name


class VideoDataset_NR_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, filename_path, transform, resize, database_name):
        super(VideoDataset_NR_SlowFast_feature, self).__init__()

        if database_name == 'KoNViD-1k':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            for i in range(n):
                video_names.append(dataInfo['video_names'][i][0][0])
            self.video_names = video_names

        elif database_name == 'youtube_ugc':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            for i in range(n):
                video_names.append(dataInfo['video_names'][i][0][0])
            self.video_names = video_names

        self.transform = transform           
        self.videos_dir = data_dir
        self.resize = resize
        self.database_name = database_name
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]
        filename=os.path.join(self.videos_dir, video_name)

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap=cv2.VideoCapture(filename)

        video_channel = 3
        
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        if video_frame_rate == 0:
           video_clip = 10
        else:
            video_clip = int(video_length/video_frame_rate)

        if self.database_name == 'KoNViD-1k':
            video_clip_min = 8
        elif self.database_name == 'youtube_ugc':
            video_clip_min = 20

        video_length_clip = 32             

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []
        
        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1


        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
 
        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i*video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[i*video_frame_rate : (i*video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i*video_frame_rate)] = transformed_frame_all[i*video_frame_rate :]
                for j in range((video_length - i*video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i*video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])
       
        return transformed_video_all, video_name_str


class wide_VideoDataset_NR_LSVQ_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize, is_test_1080p=False):
        super(wide_VideoDataset_NR_LSVQ_SlowFast_feature, self).__init__()

        column_names = ['Video Name','User ID','Deformation','Shake','Blur','Exposure','MOS']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")

        self.video_names = dataInfo['Video Name']
        self.score = dataInfo['MOS']
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx]))) / 2  # 将视频的评分缩到0-5

        filename = os.path.join(self.videos_dir, video_name )

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        video_channel = 3

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获得视频的总帧数
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))  # 获得视频的帧率

        video_clip = int(video_length / video_frame_rate)  # 获得整个视频的时间

        video_clip_min = 8

        video_length_clip = 32

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []

        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 将BRG格式图片转化为RGB格式
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]  # 复制前一帧

        video_capture.release()  # 释放占用的资源

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[
                                    i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])

        return transformed_video_all, video_score, video_name


class wide_VideoDataset_images_with_motion_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    # videos_dir, feature_dir, AADB_dir, datainfo_train, transformations_train,'LSVQ_train', config.crop_size, 'SlowFast'
    def __init__(self, data_dir, data_dir_3D, filename_path, transform, database_name, crop_size, feature_type):
        super(wide_VideoDataset_images_with_motion_features, self).__init__()
        if database_name == 'wide_angle_video_deformation_train':
            column_names = ['Video Name','Deformation','MOS']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['Video Name'].tolist()
            self.score = dataInfo['MOS'].tolist()

        if database_name == 'wide_angle_video_deformation_test':
            column_names = ['Video Name','Deformation','MOS']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['Video Name'].tolist()
            self.score = dataInfo['MOS'].tolist()

        if database_name == 'wide_angle_video_train':
            column_names = ['Video Name','Deformation','Shake','Blur','Exposure','MOS','Width','Height','Frame']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['Video Name'].tolist()
            self.score = dataInfo['MOS'].tolist()

        if database_name == 'wide_angle_video_test':
            column_names = ['Video Name','Deformation','Shake','Blur','Exposure','MOS','Width','Height','Frame']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['Video Name'].tolist()
            self.score = dataInfo['MOS'].tolist()

        # 获得数据集的名称和分数
        if database_name == 'KoNViD-1k':
            dataInfo = scio.loadmat(filename_path)  # 将Scipy中的matlab格式的数据文件转化为python中的数据结构
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            self.video_names = video_names
            self.score = score

        elif database_name == 'youtube_ugc':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][0][i])
            self.video_names = video_names
            self.score = score

        elif database_name == 'LSVQ_train':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test_1080p':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name == 'KoNViD-1k' \
                or self.database_name == 'youtube_ugc':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]  # 去除4个字符的拓展名
        elif self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test' or self.database_name == 'LSVQ_test_1080p':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]  # 获取文件名中字符串部分
        elif self.database_name == 'wide_angle_video_train' or self.database_name =='wide_angle_video_test':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name == 'wide_angle_video_deformation_train' or self.database_name =='wide_angle_video_deformation_test':
            video_name = self.video_names[idx]
            video_name_str = video_name

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))  # 将numpy数组转换为pytorch形式的floattensor

        path_name = os.path.join(self.videos_dir, video_name_str)  # 连接两个文件名 中间会自动加上

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        if self.database_name == 'KoNViD-1k' or self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test' or self.database_name == 'LSVQ_test_1080p' or self.database_name =='wide_angle_video_train' or self.database_name =='wide_angle_video_test' or self.database_name =='wide_angle_video_deformation_test' or self.database_name =='wide_angle_video_deformation_train':
            video_length_read = 8  # 16
        elif self.database_name == 'youtube_ugc':
            video_length_read = 20
        # video_length_read就是batch
        transformed_video = torch.zeros(
            [video_length_read, video_channel, video_height_crop, video_width_crop])  # 8*3*448*448

        for i in range(video_length_read):  # 处理一个batch中的数据变成transform形式
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # read 3D features  读取slowfast的3D特征
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str )
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str )
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])  # 8*(2048+256)
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        return transformed_video, transformed_feature, video_score, video_name
        # 返回一个视频的空间和时间特征


class wide_VideoDataset_images(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    # videos_dir, feature_dir, AADB_dir, datainfo_train, transformations_train,'LSVQ_train', config.crop_size, 'SlowFast'
    def __init__(self, data_dir, filename_path, transform, database_name, crop_size):
        super(wide_VideoDataset_images, self).__init__()
        if database_name == 'wide_angle_video_train':
            column_names = ['Video Name','Deformation','Shake','Blur','Exposure','MOS','Width','Height','Frame']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['Video Name'].tolist()
            self.score = dataInfo['MOS'].tolist()

        if database_name == 'wide_angle_video_test':
            column_names = ['Video Name','Deformation','Shake','Blur','Exposure','MOS','Width','Height','Frame']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['Video Name'].tolist()
            self.score = dataInfo['MOS'].tolist()

        # 获得数据集的名称和分数
        if database_name == 'KoNViD-1k':
            dataInfo = scio.loadmat(filename_path)  # 将Scipy中的matlab格式的数据文件转化为python中的数据结构
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            self.video_names = video_names
            self.score = score

        elif database_name == 'youtube_ugc':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][0][i])
            self.video_names = video_names
            self.score = score

        elif database_name == 'LSVQ_train':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test_1080p':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.video_names)
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name == 'KoNViD-1k' \
                or self.database_name == 'youtube_ugc':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]  # 去除4个字符的拓展名
        elif self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test' or self.database_name == 'LSVQ_test_1080p':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]  # 获取文件名中字符串部分
        elif self.database_name == 'wide_angle_video_train' or self.database_name =='wide_angle_video_test':
            video_name = self.video_names[idx]
            video_name_str = video_name

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))  # 将numpy数组转换为pytorch形式的floattensor

        path_name = os.path.join(self.videos_dir, video_name_str)  # 连接两个文件名 中间会自动加上

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        if self.database_name == 'KoNViD-1k' or self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test' or self.database_name == 'LSVQ_test_1080p' or self.database_name =='wide_angle_video_train' or self.database_name =='wide_angle_video_test':
            video_length_read = 8  # 16
        elif self.database_name == 'youtube_ugc':
            video_length_read = 20
        # video_length_read就是batch
        transformed_video = torch.zeros(
            [video_length_read, video_channel, video_height_crop, video_width_crop])  # 8*3*448*448

        for i in range(video_length_read):  # 处理一个batch中的数据变成transform形式
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame



        return transformed_video, video_score, video_name


def read_float_with_comma(num):
    return float(num.replace(",", "."))

class wide_VideoDataset_images_with_motion_features_and_deformation(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    # videos_dir, feature_dir, AADB_dir, datainfo_train, transformations_train,'LSVQ_train', config.crop_size, 'SlowFast'
    def __init__(self, data_dir, frame_dir,data_dir_3D, filename_path, transform, transform2, database_name, crop_size, feature_type, seed=0):
        super(wide_VideoDataset_images_with_motion_features_and_deformation, self).__init__()
        if database_name == 'wide_angle_video_deformation_train':
            column_names = ['Video Name','Deformation','MOS']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['Video Name'].tolist()
            self.score = dataInfo['MOS'].tolist()

        if database_name == 'wide_angle_video_deformation_test':
            column_names = ['Video Name','Deformation','MOS']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['Video Name'].tolist()
            self.score = dataInfo['MOS'].tolist()

        if database_name == 'wide_angle_video_train':
            column_names = ['Video Name','Deformation','Shake','Blur','Exposure','MOS','Width','Height','Frame']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['Video Name'].tolist()
            self.score = dataInfo['MOS'].tolist()

        if database_name == 'wide_angle_video_test':
            column_names = ['Video Name','Deformation','Shake','Blur','Exposure','MOS','Width','Height','Frame']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['Video Name'].tolist()
            self.score = dataInfo['MOS'].tolist()

        # 获得数据集的名称和分数
        if database_name == 'live_vqc_train':
            column_names = ['video_name','score']  # 将Scipy中的matlab格式的数据文件转化为python中的数据结构
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['video_name'].tolist()
            self.score = dataInfo['score'].tolist()
        elif database_name == 'live_vqc_test':
            column_names = ['video_name','score']  # 将Scipy中的matlab格式的数据文件转化为python中的数据结构
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['video_name'].tolist()
            self.score = dataInfo['score'].tolist()

        # elif database_name == 'KoNViD-1k_train':
        #     column_names = ['video_name','score']  # 将Scipy中的matlab格式的数据文件转化为python中的数据结构
        #     dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
        #                            encoding="utf-8-sig")
        #     self.video_names = dataInfo['video_name'].tolist()
        #     self.score = dataInfo['score'].tolist()
        # elif database_name == 'KoNViD-1k_test':
        #     column_names = ['video_name','score']  # 将Scipy中的matlab格式的数据文件转化为python中的数据结构
        #     dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
        #                            encoding="utf-8-sig")
        #     self.video_names = dataInfo['video_name'].tolist()
        #     self.score = dataInfo['score'].tolist()

        elif database_name == 'youtube_ugc_train':
            column_names = ['video_names', 'scores']  # 将Scipy中的matlab格式的数据文件转化为python中的数据结构
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['video_names'].tolist()
            self.score = dataInfo['scores'].tolist()
        elif database_name == 'youtube_ugc_test':
            column_names = ['video_names', 'scores']  # 将Scipy中的matlab格式的数据文件转化为python中的数据结构
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['video_names'].tolist()
            self.score = dataInfo['scores'].tolist()

        elif database_name == 'LSVQ_train':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test_1080p':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                                   encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name[:6] == 'KoNViD':
            m = scio.loadmat(filename_path)
            n = len(m['video_names'])
            video_names = []
            score = []
            index_all = m['index'][0]
            for i in index_all:
                # video_names.append(dataInfo['video_names'][i][0][0])
                # video_names.append(m['video_names'][i][0][0].split('_')[0] + '.mp4')
                video_names.append(m['video_names'][i][0][0].split('_')[0])
                score.append(m['scores'][i][0])

            if database_name == 'KoNViD-1k':
                self.video_names = video_names
                self.score = score
            else:
                dataInfo = pd.DataFrame(video_names)
                dataInfo['score'] = score
                dataInfo.columns = ['file_names', 'MOS']
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.8)]
                print(f'KoNViD-1k train_index: {train_index}')
                # val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                # print(f'KoNViD-1k val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'KoNViD-1k test_index: {test_index}')
                if database_name == 'KoNViD-1k_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                # elif database_name == 'KoNViD-1k_val':
                #     self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                #     self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'KoNViD-1k_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:7] == 'LiveVQC':
            m = scio.loadmat(filename_path)
            dataInfo = pd.DataFrame(m['video_list'])
            dataInfo['MOS'] = m['mos']
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names'].str.slice(2, 10)
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'LiveVQC':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                '''
                train_index = index_rd[0:int(length * 0.6)]
                print(f'LiveVQC train_index: {train_index}')
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                print(f'LiveVQC val_index: {val_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LiveVQC test_index: {test_index}')
                if database_name == 'LiveVQC_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()
                '''
                train_index = index_rd[0:int(length * 0.8)]
                print(f'LiveVQC train_index: {train_index}')
                test_index = index_rd[int(length * 0.8):]
                print(f'LiveVQC test_index: {test_index}')
                if database_name == 'LiveVQC_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'LiveVQC_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        elif database_name[:7] == 'CVD2014':
            file_names = []
            mos = []
            openfile = open("/data/user/zhaoyimeng/ModularBVQA/data/CVD2014_Realignment_MOS.csv", 'r', newline='')
            lines = csv.DictReader(openfile, delimiter=';')
            for line in lines:
                if len(line['File_name']) > 0:
                    file_names.append(line['File_name'])
                if len(line['Realignment MOS']) > 0:
                    mos.append(read_float_with_comma(line['Realignment MOS']))
            dataInfo = pd.DataFrame(file_names)
            dataInfo['MOS'] = mos
            dataInfo.columns = ['file_names', 'MOS']
            dataInfo['file_names'] = dataInfo['file_names'].astype(str)
            dataInfo['file_names'] = dataInfo['file_names']
            video_names = dataInfo['file_names'].tolist()
            score = dataInfo['MOS'].tolist()
            if database_name == 'CVD2014':
                self.video_names = video_names
                self.score = score
            else:
                length = dataInfo.shape[0]
                random.seed(seed)
                np.random.seed(seed)
                index_rd = np.random.permutation(length)
                train_index = index_rd[0:int(length * 0.6)]
                val_index = index_rd[int(length * 0.6):int(length * 0.8)]
                test_index = index_rd[int(length * 0.8):]
                if database_name == 'CVD2014_train':
                    self.video_names = dataInfo.iloc[train_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[train_index]['MOS'].tolist()
                elif database_name == 'CVD2014_val':
                    self.video_names = dataInfo.iloc[val_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[val_index]['MOS'].tolist()
                elif database_name == 'CVD2014_test':
                    self.video_names = dataInfo.iloc[test_index]['file_names'].tolist()
                    self.score = dataInfo.iloc[test_index]['MOS'].tolist()

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.frame_dir = frame_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.transform2 = transform2
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database_name == 'KoNViD-1k_train' or self.database_name == 'KoNViD-1k_test'\
                or self.database_name == 'live_vqc_train'or self.database_name == 'live_vqc_test':
            video_name = self.video_names[idx]
            video_name_str = str(video_name)  # 去除4个字符的拓展名
        elif self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test' or self.database_name == 'LSVQ_test_1080p'or self.database_name[:7] == 'CVD2014':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]  # 获取文件名中字符串部分
        elif self.database_name == 'LiveVQC_train' or self.database_name =='LiveVQC_test':
            video_name = self.video_names[idx]
            video_name_str = video_name[:-4]
        elif self.database_name == 'wide_angle_video_train' or self.database_name =='wide_angle_video_test':
            video_name = self.video_names[idx]
            video_name_str = video_name
        elif self.database_name == 'youtube_ugc_train' or self.database_name == 'youtube_ugc_test':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name


        video_score = torch.FloatTensor(np.array(float(self.score[idx])))  # 将numpy数组转换为pytorch形式的floattensor


        path_name = os.path.join(self.videos_dir, video_name_str)  # 连接两个文件名 中间会自动加上
        path_frame = os.path.join(self.frame_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
        center_width = 96
        center_height = 96
        block_width = 96
        block_height = 96

        if (self.database_name == 'KoNViD-1k_train' or self.database_name == 'KoNViD-1k_test' or self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test'
                or self.database_name == 'LSVQ_test_1080p'  or self.database_name =='wide_angle_video_deformation_test' or self.database_name =='wide_angle_video_deformation_train'or self.database_name =='live_vqc_test'
                or self.database_name =='live_vqc_train'or self.database_name== 'LiveVQC_train'or self.database_name== 'LiveVQC_test'or self.database_name[:7] == 'CVD2014'):
            video_length_read = 8  # 16

        if self.database_name == 'wide_angle_video_train' or self.database_name =='wide_angle_video_test' :
            video_length_read = 10
        elif self.database_name == 'youtube_ugc_train' or self.database_name == 'youtube_ugc_test':
            video_length_read = 8
        # video_length_read就是batch
        transformed_video = torch.zeros(
            [video_length_read, video_channel, video_height_crop, video_width_crop])  # 8*3*448*448

        center_video = torch.zeros([video_length_read, video_channel, center_height, center_width])
        block_video1 = torch.zeros([video_length_read, video_channel, block_height, block_width])
        block_video2 = torch.zeros([video_length_read, video_channel, block_height, block_width])
        block_video3 = torch.zeros([video_length_read, video_channel, block_height, block_width])
        block_video4 = torch.zeros([video_length_read, video_channel, block_height, block_width])

        for i in range(video_length_read):  # 处理一个batch中的数据变成transform形式
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # 返回中心的96*96，和从左下角到右上角的4块32*32
        for i in range(video_length_read):
            imge_name = os.path.join(path_frame, '{:03d}'.format(i) +'_top_left.' + '.png')
            read_frame = Image.open(imge_name)
            width, height = read_frame.size

            # 计算中心区域的坐标
            start_x = (width - 96) // 2
            start_y = (height - 96) // 2
            end_x = start_x + 96
            end_y = start_y + 96

            gap_X = (width - 96*4) // 3
            gap_Y = (height - 96*4) // 3
            start_x1 = 0
            start_x2 = start_x1 + gap_X + 96
            start_x3 = start_x2 + gap_X + 96
            start_x4 = start_x3 + gap_X + 96
            start_y1 = 0
            start_y2 = start_y1 + gap_Y + 96
            start_y3 = start_y2 + gap_Y + 96
            start_y4 = start_y3 + gap_Y + 96
            end_x1 = start_x1 + 96
            end_x2 = start_x2 + 96
            end_x3 = start_x3 + 96
            end_x4 = start_x4 + 96
            end_y1 = start_y1 + 96
            end_y2 = start_y2 + 96
            end_y3 = start_y3 + 96
            end_y4 = start_y4 + 96

            # 裁剪出中心区域和4块
            central_region = read_frame.crop((start_x, start_y, end_x, end_y))
            block1 = read_frame.crop((start_x1, start_y1, end_x1, end_y1))
            block2 = read_frame.crop((start_x2, start_y2, end_x2, end_y2))
            block3 = read_frame.crop((start_x3, start_y3, end_x3, end_y3))
            block4 = read_frame.crop((start_x4, start_y4, end_x4, end_y4))


            central_region = central_region.convert('RGB')
            block1 = block1.convert('RGB')
            block2 = block2.convert('RGB')
            block3 = block3.convert('RGB')
            block4 = block4.convert('RGB')

            central_region = self.transform2(central_region)
            block1 = self.transform2(block1)
            block2 = self.transform2(block2)
            block3 = self.transform2(block3)
            block4 = self.transform2(block4)

            center_video[i] = central_region
            block_video1[i] = block1
            block_video2[i] = block2
            block_video3[i] = block3
            block_video4[i] = block4

        if self.feature_type == 'tad':
            video_name_str = video_name_str[:-4]
            video_name_str = video_name_str + '.npy'
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            # transformed_feature = torch.zeros([video_length_read, 1408])
            feature_3D = np.load(feature_folder_name)
            feature_3D = torch.from_numpy(feature_3D)
            feature_3D = feature_3D.squeeze()
            transformed_feature = feature_3D

        if self.feature_type == 'tad_lsvq' or self.feature_type == 'tad_konvid' or self.feature_type == 'tad_live-vqc'or self.feature_type == 'tad_CVD2014':
            # video_name_str = video_name_str[:-4]
            video_name_str = video_name_str + '.npy'
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            # transformed_feature = torch.zeros([video_length_read, 1408])
            feature_3D = np.load(feature_folder_name)
            feature_3D = torch.from_numpy(feature_3D)
            feature_3D = feature_3D.squeeze()
            transformed_feature = feature_3D
        # read 3D features  读取slowfast的3D特征
        if self.feature_type == 'Slow':
            video_name_str = video_name_str[:-4]
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str )
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            video_name_str = video_name_str[:-4]
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            video_name_str = video_name_str[:-4]
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str )
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])  # 8*(2048+256)
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        return transformed_video, transformed_feature, video_score, video_name, center_video, block_video1, block_video2, block_video3, block_video4