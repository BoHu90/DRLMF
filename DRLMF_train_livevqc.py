# -*- coding: utf-8 -*-
import argparse
import os

import math
import numpy as np
import torch
import torch.optim as optim
from swin_model import swin_tiny_patch4_window7_224 as create_model
import torch.nn as nn
from matplotlib import pyplot as plt
from swin3d.video_swin_transformer_train import SwinTransformer3D

from data_loader import wide_VideoDataset_images
from utils import performance_fit
from utils import L1RankLoss
from data_loader import wide_VideoDataset_images_with_motion_features
from data_loader import wide_VideoDataset_images_with_motion_features_and_deformation
from model import UGC_BVQA_model
from model import modular
from torchvision import transforms
import time
import torchvision.models as models
import torch.nn.functional as F


# 论文2的模型，在MWV数据集
def plcc_loss(y, y_pred):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = F.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = F.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

class AdaptiveWeightFusion(nn.Module):
    def __init__(self, global_dim=512, local_dim=2048):
        super(AdaptiveWeightFusion, self).__init__()
        # MLP to learn adaptive weights for both global and local features
        self.weight_net = nn.Sequential(
            nn.Linear(1024, 512),  # 768 + 2048
            nn.ReLU(),
            nn.Linear(512, 1),  # output a single weight for each frame
            nn.Sigmoid()
        )
        # Linear layer to project local features to the same dimension as global features
        self.local_projection = nn.Linear(local_dim, global_dim)  # project local to 768

    def forward(self, global_features, local_features):
        # Project local features to match global features dimension
        local_features_proj = self.local_projection(local_features)  # (batch*frame)*768

        # Concatenate features along the last dimension
        concat_features = torch.cat([global_features, local_features_proj], dim=-1)  # (batch*frame)*(768+768)

        # Generate the adaptive weight (alpha)
        alpha = self.weight_net(concat_features)  # (batch*frame)*1

        # Perform weighted fusion
        fused_features = alpha * global_features + (1 - alpha) * local_features_proj  # (batch*frame)*768
        return fused_features


class CrossAttentionFusion(nn.Module):
    def __init__(self, spatial_dim=512, temporal_dim=1408, attention_dim=512):
        super(CrossAttentionFusion, self).__init__()

        # Linear layers to project the features into the same space
        self.spatial_projection = nn.Linear(spatial_dim, attention_dim)  # (batch*frame, attention_dim)
        self.temporal_projection = nn.Linear(temporal_dim, attention_dim)  # (batch*frame, attention_dim)

        # Attention mechanism
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, spatial_features, temporal_features):
        # Project spatial and temporal features
        # spatial_proj = self.spatial_projection(spatial_features)  # (batch*frame, attention_dim)
        temporal_proj = self.temporal_projection(temporal_features)  # (batch*frame, attention_dim)

        # Compute attention weights
        attention_scores = torch.bmm(spatial_features.unsqueeze(1), temporal_proj.unsqueeze(2))
        # attention_scores = torch.bmm(spatial_proj.unsqueeze(1), temporal_proj.unsqueeze(2))# (batch*frame, 1, attention_dim) x (batch*frame, attention_dim, 1)
        attention_weights = self.softmax(attention_scores)  # (batch*frame, 1, 1)

        # Apply attention weights to temporal features
        attended_temporal = attention_weights.squeeze(1) * temporal_proj  # (batch*frame, attention_dim)

        # Concatenate projected spatial features and attended temporal features
        fused_features = torch.cat((spatial_features, attended_temporal), dim=1) # (batch*frame, attention_dim)

        return fused_features

class convNet(nn.Module):
    #constructor
    def __init__(self, resnet, swin_transformer, Adaptive_WeightFusion, cross_attention_layer):
        super(convNet, self).__init__()
        self.resnet = resnet
        self.Adaptive_WeightFusion = Adaptive_WeightFusion
        self.cross_attention_layer = cross_attention_layer
        self.swin_transformer = swin_transformer
        self.k1 = nn.Parameter(torch.Tensor([0.5]))  # 初始化为0.5，但可以是任何值
        self.k2 = nn.Parameter(torch.Tensor([0.5]))
        self.k3 = nn.Parameter(torch.Tensor([0.5]))
        self.k4 = nn.Parameter(torch.Tensor([0.5]))
        self.quality = self.quality_regression(512+512, 128, 1)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block


    def forward(self, x_img, x_3D_features, block_video1, block_video2, block_video3, block_video4):

        x_size = x_img.size()
        x = self.swin_transformer(x_img)

        x_3D_features_size = x_3D_features.size()
        x_3D_features = x_3D_features.view(-1, x_3D_features_size[2])
        # x = x.view(x_size[0], x_size[1], -1)
        block_video1_size = block_video1.shape
        block_video2_size = block_video2.shape
        block_video3_size = block_video3.shape
        block_video4_size = block_video4.shape
        block_video1 = block_video1.view(-1, block_video1_size[2], block_video1_size[3], block_video1_size[4])
        block_video2 = block_video2.view(-1, block_video2_size[2], block_video2_size[3], block_video2_size[4])
        block_video3 = block_video3.view(-1, block_video3_size[2], block_video3_size[3], block_video3_size[4])
        block_video4 = block_video4.view(-1, block_video4_size[2], block_video4_size[3], block_video4_size[4])
        block_video1 = self.resnet(block_video1)
        block_video2 = self.resnet(block_video2)
        block_video3 = self.resnet(block_video3)
        block_video4 = self.resnet(block_video4)
        block_video1 = torch.flatten(block_video1, 1)
        block_video2 = torch.flatten(block_video2, 1)
        block_video3 = torch.flatten(block_video3, 1)
        block_video4 = torch.flatten(block_video4, 1)
        block_video = self.k1 * block_video1 + self.k2 * block_video2 + self.k3 * block_video3 + self.k4 * block_video4
        x = self.Adaptive_WeightFusion(x, block_video)

        x = self.cross_attention_layer(x, x_3D_features)

        # x = torch.cat((x, x_3D_features), dim=1)
        x = self.quality(x)
        x = x.view(x_size[0], x_size[1])
        x = torch.mean(x, dim=1)

        return x


def main(config):
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Adaptive_WeightFusion = AdaptiveWeightFusion(global_dim=512, local_dim=2048)
    cross_attention_layer = CrossAttentionFusion(spatial_dim=512, temporal_dim=1408)
    swin_transformer = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=10, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2).float()
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.to(device)

    model = convNet(resnet, swin_transformer, Adaptive_WeightFusion, cross_attention_layer).to(device)

    if config.trained_model is not None:
        # load the trained model
        print('loading the pretrained model')
        # model.load_state_dict(torch.load(config.trained_model))

        # 模型使用DataParallel后多了"module."
        state_dict = torch.load(config.trained_model)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)

        print('loading model success!')

    optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.0000001)


    if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)


    # lr_scheduler.StepLR调整学习率机制,一般情况下我们会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果。
    # 等间隔调整学习率，调整倍数为gamma倍学习率,调整间隔为step_size,
    # last_epoch:最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的epoch。默认为-1表示从头开始训练
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)  # 2 0.9
    if config.loss_type == 'plcc':
        criterion = plcc_loss

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))    #prod是计算所有元素的乘积，从而这个代码表示统计所以参数的总和
    print('Trainable params: %.2f million' % (param_num / 1e6))

    videos_dir = '/data/user/zhaoyimeng/ModularBVQA/data/livevqc_image_all_fps1'
    frame_dir = '/data/user/gbb/SimpleVQA/livevqc_image_all_fps1_motion_4kuai'
    feature_dir = '/data/user/zhaoyimeng/VideoMAEv2-master/save_feat/LiveVQC_VideoMAE_feat'

    datainfo_train = '/data/user/zhaoyimeng/ModularBVQA/data/LiveVQC_data.mat'
    datainfo_test = '/data/user/zhaoyimeng/ModularBVQA/data/LiveVQC_data.mat'
    # datainfo_train = '/data/user/gbb/SimpleVQA-main/data/live_vqc_train_data.csv'
    # datainfo_test = '/data/user/gbb/SimpleVQA-main/data/live_vqc_test_data.csv'

    transformations_train = transforms.Compose(
        # transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC)  BILINEAR NEAREST
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.RandomCrop(config.crop_size), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.CenterCrop(config.crop_size), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations_frame_train = transforms.Compose(
        [ transforms.ToTensor(),  # 520 448
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations_frame_test = transforms.Compose(
        [ transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    #返回包含空间信息和时间信息的特征
    trainset = wide_VideoDataset_images_with_motion_features_and_deformation(videos_dir, frame_dir, feature_dir, datainfo_train,
                                                             transformations_train, transformations_frame_train,
                                                             'LiveVQC_train', config.crop_size, 'tad_live-vqc')
    testset =wide_VideoDataset_images_with_motion_features_and_deformation(videos_dir, frame_dir, feature_dir, datainfo_test,
                                                            transformations_test,transformations_frame_test,
                                                            'LiveVQC_test', config.crop_size, 'tad_live-vqc')
    #testset_1080p = VideoDataset_images_with_motion_features(videos_dir, feature_dir, datainfo_test_1080p,transformations_test, 'LSVQ_test_1080p', config.crop_size,'SlowFast')

    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,  # 8
                                               shuffle=True, num_workers=config.num_workers)  # 6
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)
    #test_loader_1080p = torch.utils.data.DataLoader(testset_1080p, batch_size=1, shuffle=False, num_workers=config.num_workers)

    best_test_criterion = -1  # SROCC min
    best_test = []
    #best_test_1080p = []

    print('Starting training:')

    old_save_name = None

    avg_loss_train = []  # 用于记录loss值，打印loss曲线

    for epoch in range(config.epochs):  # 10
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (video, feature_3D, mos, video_name, center_video, block_video1, block_video2, block_video3, block_video4) in enumerate(train_loader):  # feature_3D是提取出来的运动信息

            video = video.to(device)

            feature_3D = feature_3D.to(device)
            labels = mos.to(device).float()
            center_video = center_video.to(device)
            block_video1 = block_video1.to(device)
            block_video2 = block_video2.to(device)
            block_video3 = block_video3.to(device)
            block_video4 = block_video4.to(device)
            #print(center_video.shape)
           # print(labels)

            outputs = model(video, feature_3D, block_video1, block_video2, block_video3, block_video4)
            #print(outputs.shape)
            optimizer.zero_grad()

            loss = criterion(labels, outputs)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()

            optimizer.step()

            if (i + 1) % (config.print_samples // config.train_batch_size) == 0:  #每125batch打印一次结果
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples // config.train_batch_size)
                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' %
                      (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, avg_loss_epoch))
                batch_losses_each_disp = []
                print('CostTime: {:.4f}'.format(session_end_time - session_start_time)) #跑125个batch时间
                session_start_time = time.time()

        avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)
        avg_loss_train.append(avg_loss)
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr = scheduler.get_last_lr()  #调用了 scheduler 对象的 get_last_lr 方法，用于获取最近一次学习率更新后的学习率值，并将其赋值给变量 lr。
        print('The current learning rate is {:.06f}'.format(lr[0]))

        # do validation after each epoch
        with torch.no_grad():    #下文中不会计算梯度
            model.eval()
            label = np.zeros([len(testset)])
            y_output = np.zeros([len(testset)])
            for i, (video, feature_3D, mos, video_name, center_video, block_video1, block_video2, block_video3, block_video4) in enumerate(test_loader):

                video = video.to(device)
                feature_3D = feature_3D.to(device)
                label[i] = mos.item()
                center_video = center_video.to(device)
                block_video1 = block_video1.to(device)
                block_video2 = block_video2.to(device)
                block_video3 = block_video3.to(device)
                block_video4 = block_video4.to(device)
                outputs = model(video, feature_3D, block_video1, block_video2, block_video3, block_video4)
                y_output[i] = outputs.item()

            test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(label, y_output)

            print(
                'Epoch {} completed. The result on the test databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'
                .format(epoch + 1, test_SRCC, test_KRCC, test_PLCC, test_RMSE))

            #注释了1080p在下方注释

            if test_SRCC > best_test_criterion:
                print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                best_test_criterion = test_SRCC
                best_test = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                #best_test_1080p = [test_SRCC_1080p, test_KRCC_1080p, test_PLCC_1080p, test_RMSE_1080p]
                print('Saving model...')
                #将最好的参数存储到ckpt中
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)

                if epoch > 0:
                    if os.path.exists(old_save_name):
                        os.remove(old_save_name)

                save_model_name = os.path.join(config.ckpt_path, config.model_name + '_' +
                                               config.database + '_' + config.loss_type + '_NR_v' +
                                               str(config.exp_version) +
                                               '_epoch_%d_SRCC_%f.pth' % (epoch + 1, test_SRCC))
                torch.save(model.state_dict(), save_model_name)
                old_save_name = save_model_name

    print('Training completed.')
    print(
        'The best training result on the test dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'
        .format(best_test[0], best_test[1], best_test[2], best_test[3]))
    #print('The best training result on the test_1080p dataset SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(best_test_1080p[0], best_test_1080p[1], best_test_1080p[2], best_test_1080p[3]))

    x = range(1, config.epochs + 1)
    # 绘制loss折线图
    plt.grid(ls='-', color='grey')  # 设置网格，颜色为灰色
    plt.plot(x, avg_loss_train, color='red', label=u'train')  # train_loss曲线
    plt.legend()  # 让图例生效
    plt.margins(0)
    plt.xlabel("epoch", loc="right")  # X轴标签
    plt.title("loss")  # 标题
    plt.show()


""" label_1080p = np.zeros([len(testset_1080p)])
            y_output_1080p = np.zeros([len(testset_1080p)])
            for i, (video, feature_3D, mos, _) in enumerate(test_loader_1080p):
                video = video.to(device)
                feature_3D = feature_3D.to(device)
                label_1080p[i] = mos.item()
                outputs = model(video, feature_3D)
                y_output_1080p[i] = outputs.item()

            test_PLCC_1080p, test_SRCC_1080p, test_KRCC_1080p, test_RMSE_1080p = performance_fit(label_1080p, y_output_1080p)

            print(
                'Epoch {} completed. The result on the test_1080p databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    epoch + 1, test_SRCC_1080p, test_KRCC_1080p, test_PLCC_1080p, test_RMSE_1080p))  """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str, default='livevqc')  # LSVQ
    parser.add_argument('--model_name', type=str, default='UGC_BVQA_model')  # UGC_BVQA_model
    parser.add_argument('--num_classes', type=int, default=5)
    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.93)  # 0.9
    parser.add_argument('--decay_interval', type=int, default=2)  #多少步长来控制权重衰减
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int, default=0)  # 0
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=8) #16
    parser.add_argument('--num_workers', type=int, default=6)  # 6
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weights', type=str, default='/data/user/gbb/SimpleVQA-main/model/swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--trained_model', type=str,
                        default='/data/user/XXX/ckpts/DRLMF.pth')
    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts')   #用于存储最好的效果
    parser.add_argument('--multi_gpu', action='store_true')  # 当命令行中触发multi_gpu参数时，返回为True，没有触发的时候返回False
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='plcc')

    config = parser.parse_args()
    main(config)

