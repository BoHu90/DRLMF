# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from utils import performance_fit
from data_loader import wide_VideoDataset_images_with_motion_features_and_deformation
from model import modular
from torchvision import transforms
import time
import torchvision.models as models
import torch.nn.functional as F

# ............................. test model ......................

# Loss
def plcc_loss(y, y_pred):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = F.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = F.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


# Content-adaptive Modulation.
class CaM(nn.Module):
    def __init__(self, global_dim=512, local_dim=2048):
        super(CaM, self).__init__()

        # W1Net
        self.weight_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.local_projection = nn.Linear(local_dim, global_dim)

    def forward(self, global_features, local_features):
        local_features_proj = self.local_projection(local_features)
        concat_features = torch.cat([global_features, local_features_proj], dim=-1)
        alpha = self.weight_net(concat_features)
        fused_features = alpha * global_features + (1 - alpha) * local_features_proj
        return fused_features


# Quality-adaptive Modulation.
class QaM(nn.Module):
    def __init__(self, spatial_dim=512, temporal_dim=1408, attention_dim=512):
        super(QaM, self).__init__()

        self.spatial_projection = nn.Linear(spatial_dim, attention_dim)
        self.temporal_projection = nn.Linear(temporal_dim, attention_dim)

        # W2Net
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, spatial_features, temporal_features):
        temporal_proj = self.temporal_projection(temporal_features)
        attention_scores = torch.bmm(spatial_features.unsqueeze(1), temporal_proj.unsqueeze(2))
        # attention_weights is β
        attention_weights = self.softmax(attention_scores)
        attended_temporal = attention_weights.squeeze(1) * temporal_proj
        fused_features = torch.cat((spatial_features, attended_temporal), dim=1)

        return fused_features


# This is an integration class for the model, which can combine the parameters of the three-stream network into a single model.
class convNet(nn.Module):
    # constructor
    def __init__(self, resnet, CLIP, Cam, Qam):
        super(convNet, self).__init__()
        self.resnet = resnet
        self.Cam = Cam
        self.Qam = Qam
        self.CLIP = CLIP
        self.k1 = nn.Parameter(torch.Tensor([0.5]))  # init = 0.5
        self.k2 = nn.Parameter(torch.Tensor([0.5]))
        self.k3 = nn.Parameter(torch.Tensor([0.5]))
        self.k4 = nn.Parameter(torch.Tensor([0.5]))
        self.quality = self.quality_regression(512 + 512, 128, 1)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, x_img, x_3D_features, block_video1, block_video2, block_video3, block_video4):
        x_size = x_img.size()
        x = self.CLIP(x_img)

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
        x = self.Cam(x, block_video)

        x = self.Qam(x, x_3D_features)

        # x = torch.cat((x, x_3D_features), dim=1)
        x = self.quality(x)
        x = x.view(x_size[0], x_size[1])
        x = torch.mean(x, dim=1)

        return x


def main(config):
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Declare the modules in the model.
    Cam = CaM(global_dim=512, local_dim=2048)
    Qam = QaM(spatial_dim=512, temporal_dim=1408)
    CLIP = modular.ViTbCLIP_SpatialTemporal_modular_dropout(feat_len=10, sr=True, tr=True, dropout_sp=0.2,
                                                            dropout_tp=0.2).float()
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.to(device)

    model = convNet(resnet, CLIP, Cam, Qam).to(device)

    # model load
    if config.trained_model is not None:
        # load the trained model
        print('loading the pretrained model')
        # model.load_state_dict(torch.load(config.trained_model))
        state_dict = torch.load(config.trained_model)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)

        print('loading model success!')

    optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=0.0000001)

    # Multi-GPU training.
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
    if config.loss_type == 'plcc':
        criterion = plcc_loss

    # Statistics of model parameters.
    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    videos_dir = '/data/user/xxx/DRLMF/XXX_frame'  # frame_path
    frame_dir = '/data/user/xxx/DRLMF/XXX_1frame'  # four_split_path
    feature_dir = '/data/user/xxx/videomeav2_XXX_feature'  # motion_feature_path
    datainfo_train = '/data/dataset/XXX_train.csv'  # train.csv_path
    datainfo_test = '/data/dataset/XXX_test.csv'  # test.csv_path

    transformations_train = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.RandomCrop(config.crop_size), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose(
        [transforms.Resize(config.resize, interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.CenterCrop(config.crop_size), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations_frame_train = transforms.Compose(
        [transforms.ToTensor(),  # 520 448
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations_frame_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Return the training and testing data.
    trainset = wide_VideoDataset_images_with_motion_features_and_deformation(videos_dir, frame_dir, feature_dir,
                                                                             datainfo_train,
                                                                             transformations_train,
                                                                             transformations_frame_train,
                                                                             'XXX_train', config.crop_size, 'tad')
    testset = wide_VideoDataset_images_with_motion_features_and_deformation(videos_dir, frame_dir, feature_dir,
                                                                            datainfo_test,
                                                                            transformations_test,
                                                                            transformations_frame_test,
                                                                            'XXX_test', config.crop_size, 'tad')

    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,  # 8
                                               shuffle=True, num_workers=config.num_workers)  # 6
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers)

    best_test_criterion = -1  # SROCC min
    best_test = []

    print('Starting training:')

    old_save_name = None

    avg_loss_train = []

    for epoch in range(config.epochs):  # 10
        # do validation after each epoch
        with torch.no_grad():
            # test
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

            if test_SRCC > best_test_criterion:
                print("Update best model using best_test_criterion in epoch {}".format(epoch + 1))
                best_test_criterion = test_SRCC
                best_test = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]

                print('Saving model...')
                # Saving best model
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

    # Print the loss curve.
    x = range(1, config.epochs + 1)
    plt.grid(ls='-', color='grey')
    plt.plot(x, avg_loss_train, color='red', label=u'train')
    plt.legend()
    plt.margins(0)
    plt.xlabel("epoch", loc="right")
    plt.title("loss")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--database', type=str, default='MWV')
    parser.add_argument('--model_name', type=str, default='DRLMF')
    parser.add_argument('--num_classes', type=int, default=5)
    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=1e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.95)  # 0.9
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--n_trial', type=int, default=0)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--exp_version', type=int, default=0)
    parser.add_argument('--print_samples', type=int, default=1000)
    parser.add_argument('--train_batch_size', type=int, default=8)  # 16
    parser.add_argument('--num_workers', type=int, default=6)  # 6
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weights', type=str, default='/data/user/XXX/model/swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--trained_model', type=str,
                        default='/DRLMF_MWV.pth')
    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--loss_type', type=str, default='plcc')

    config = parser.parse_args()
    main(config)

