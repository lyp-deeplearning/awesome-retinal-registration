import os
import cv2
import numpy as np



def resize_images(input_folder, output_folder, target_size):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的图片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # 读取图片
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)

            # 调整图像大小
            resized_img = cv2.resize(img, target_size)

            # 保存到输出文件夹
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img)

def scale_coordinates(input_folder, output_folder, input_size, target_size):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的txt文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            # 读取txt文件中的坐标
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r') as file:
                lines = file.readlines()

            # 将坐标从input_size映射到target_size
            scaled_coordinates = []
            for line in lines:
                x, y ,x1,y1= map(float, line.strip().split())
                scaled_x = float(x * (target_size[0] / input_size[0]))
                scaled_y = float(y * (target_size[1] / input_size[1]))
                scaled_x1 = float(x1 * (target_size[0] / input_size[0]))
                scaled_y1 = float(y1 * (target_size[1] / input_size[1]))
                scaled_coordinates.append((scaled_x, scaled_y, scaled_x1, scaled_y1))

            # 保存到输出文件夹
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w') as file:
                for coord in scaled_coordinates:
                    file.write(f"{coord[0]} {coord[1]} {coord[2]} {coord[3]}\n")

# 输入文件夹a、b和输出文件夹c、d的路径
folder_a = '/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/flori_data/val/align_img/'
folder_b = '/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/flori_data/val/align_gt/'
folder_c = '/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/flori_data/train/align_img/'
folder_d = '/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/flori_data/train/align_gt/'

# 调整图像大小
resize_images(folder_a, folder_c, (768, 768))

# 将坐标从2912映射到768
scale_coordinates(folder_b, folder_d, (2912, 2912), (768, 768))
