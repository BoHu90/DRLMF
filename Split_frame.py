import os
import csv
import pandas as pd
from PIL import Image
import scipy.io as scio

def split_image_diagonally(image_path, output_folder):
    # 打开图片
    image = Image.open(image_path)
    width, height = image.size

    # 计算对角线的分割点
    mid_width = width // 2
    mid_height = height // 2

    # 创建四个区域
    top_left = image.crop((0, 0, mid_width, mid_height))
    top_right = image.crop((mid_width, 0, width, mid_height))
    bottom_left = image.crop((0, mid_height, mid_width, height))
    bottom_right = image.crop((mid_width, mid_height, width, height))

    # 定义文件名（不包含路径）
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    # 保存四个区域
    top_left.save(os.path.join(output_folder, f"{name}_top_left.{ext}"))
    top_right.save(os.path.join(output_folder, f"{name}_top_right.{ext}"))
    bottom_left.save(os.path.join(output_folder, f"{name}_bottom_left.{ext}"))
    bottom_right.save(os.path.join(output_folder, f"{name}_bottom_right.{ext}"))


def batch_split_images(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, filename)
            split_image_diagonally(image_path, output_folder)


if __name__ == "__main__":

    # filename_path = '/data/user/gbb/SimpleVQA-main/wide.csv'
    # column_names = ['Video Name', 'User ID', 'Deformation', 'Shake', 'Blur', 'Exposure', 'MOS']
    #
    # dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
    #
    # video_names = dataInfo['Video Name']
    # n_video = len(video_names)
    # videos_dir = '/data/user/gbb/SimpleVQA-main/wide_angle_video_frame'
    #
    # save_folder = '/data/user/gbb/SimpleVQA-main/wide_angle_video_1frame'
    # for i in range(n_video):
    #     video_name = video_names.iloc[i]
    #     print('start extract {}th video: {}'.format(i, video_name))
    #     input_folder = os.path.join(videos_dir, video_name)
    #     output_folder = os.path.join(save_folder, video_name)
    #     batch_split_images(input_folder, output_folder)
    # 读取Excel文件并提取数据
    def read_float_with_comma(num):
        return float(num.replace(",", "."))

    file_names = []
    mos = []
    openfile = open("/data/user/XX/ModularBVQA/data/CVD2014_Realignment_MOS.csv", 'r', newline='')
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

    # 视频总数
    n_video = len(video_names)

    # 视频所在的目录和保存目录
    videos_dir = '/data/user/XX/ModularBVQA/data/cvd2014_image_all_fps1'
    save_folder = '/data/user/XX/DRLMF/cvd2014_all_fps1_motion_4kuai'

    # 遍历所有视频并提取图像
    for i in range(n_video):
        video_name = video_names[i]  # 获取当前视频文件名
        print(f"Start extract {i}th video: {video_name}")

        # 定义输入输出目录
        input_folder = os.path.join(videos_dir, str(video_name))
        output_folder = os.path.join(save_folder, str(video_name))

        # 执行图像提取操作（假设 batch_split_images 是有效的函数）
        batch_split_images(input_folder, output_folder)
