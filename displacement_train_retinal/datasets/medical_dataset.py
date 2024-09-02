import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
from PIL import Image
import random
import yaml
from sklearn.model_selection import train_test_split
import numpy as np
from utils.post_process import *



class Mixture_Dataset(data.Dataset):
    """
    读取混合以后的四个数据集的数据，随机打乱读取分配成moving image和fixed image
    """
    def __init__(self, list_ids, configs) -> None:
        super().__init__()
        self.list_ids = list_ids
        self.length = len(self.list_ids)
        self.configs = configs
        self.model_image_width = self.configs['train']["model_image_width"]
        self.model_image_height = self.configs['train']["model_image_width"]
        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])
        self.read_method = self.configs['dataset']["read_sequence"]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_ids)
    
    def select_sequence(self, index):
        "按照list读取的顺序进行训练,比如0,1;2,3;3,2;1,0"
        mid = self.length / 2
        if (index < mid):
            fixed = self.list_ids[2 * index] + ".jpg"
            moving = self.list_ids[2 * index + 1]+ ".jpg"
        else:
            new_ = int(self.length - 1 - (index - mid) * 2)
            fixed = self.list_ids[new_]+ ".jpg"
            if (new_ - 1) < 0:
                moving = self.list_ids[new_]+ ".jpg"
            else:
                moving  = self.list_ids[new_ - 1]+ ".jpg"
        #print("single, ",index, fixed, moving)
        return fixed, moving
    
    def select_randm(self, index):
        "按照乱序进行随机组合"
        fixed = self.list_ids[index] + ".jpg"
        moving = self.list_ids[random.randint(1, self.length-1)] + ".jpg"
        return fixed, moving
            
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if (self.read_method == "sequential"):
            id1, id2 = self.select_sequence(index)
        elif (self.read_method == "random"):
            id1, id2 = self.select_randm(index) 
        fixed_path = self.configs['train']["train_fire_dir"] + id1
        moving_path = self.configs['train']["train_fire_dir"] + id2
        fixed_image = Image.open(fixed_path).convert("RGB")
        # 转换为灰度图像
        fixed_image = fixed_image.convert('L')
        fixed_image = np.asarray(fixed_image).astype(np.uint8)
        moving_image = Image.open(moving_path).convert("RGB")
        # 转换为灰度图像
        moving_image = moving_image.convert('L')
        moving_image = np.asarray(moving_image).astype(np.uint8)
        ## using single channel
        fixed_image = Image.fromarray(fixed_image)
        fixed_image = self.transforms(fixed_image)
        moving_image = Image.fromarray(moving_image)
        moving_image = self.transforms(moving_image)

        return fixed_image, moving_image
 

## 读取fire的数据代码，左右对称读取，id:1,id:2
class Fire_Dataset(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, configs):
        'Initialization'
        self.list_IDs = list_IDs
        self.configs = configs
        self.model_image_width = self.configs['train']["model_image_width"]
        self.model_image_height = self.configs['train']["model_image_width"]

        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        # data process follow super-retinal, original super-retinal use single green channel , and input is uint8, to_tensor操作自动除255归一化
        # 加入随机性，随机左右互换
        random_num = random.randint(1, 7)
        if (random_num % 2 == 0):
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_1.jpg'
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_2.jpg'
        else:
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_2.jpg'
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_1.jpg'
        fixed_image = Image.open(fixed_path).convert("RGB")
        # 转换为灰度图像
        fixed_image = fixed_image.convert('L')
        fixed_image = np.asarray(fixed_image).astype(np.uint8)
        moving_image = Image.open(moving_path).convert("RGB")
        # 转换为灰度图像
        moving_image = moving_image.convert('L')
        moving_image = np.asarray(moving_image).astype(np.uint8)
        ## using single channel
        fixed_image = Image.fromarray(fixed_image)
        fixed_image = self.transforms(fixed_image)
        moving_image = Image.fromarray(moving_image)
        moving_image = self.transforms(moving_image)

        return fixed_image, moving_image

class Fire_Dataset_seg(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, configs):
        'Initialization'
        self.list_IDs = list_IDs
        self.configs = configs
        self.model_image_width = self.configs['train']["model_image_width"]
        self.model_image_height = self.configs['train']["model_image_width"]

        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        # data process follow super-retinal, original super-retinal use single green channel , and input is uint8, to_tensor操作自动除255归一化
        # 加入随机性，随机左右互换
        random_num = random.randint(1, 7)
        if (random_num % 2 == 0):
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_1.png'
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_2.png'
        else:
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_2.png'
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_1.png'
        fixed_image = Image.open(fixed_path)
        # 转换为灰度图像
        fixed_image = fixed_image
        fixed_image = np.asarray(fixed_image).astype(np.uint8)
        moving_image = Image.open(moving_path)
        # 转换为灰度图像
        moving_image = moving_image
        moving_image = np.asarray(moving_image).astype(np.uint8)
        ## using single channel
        fixed_image = Image.fromarray(fixed_image)
        fixed_image = self.transforms(fixed_image)
        moving_image = Image.fromarray(moving_image)
        moving_image = self.transforms(moving_image)
        
        
        return fixed_image, moving_image

class Fire_Dataset_seg_points(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, configs):
        'Initialization'
        self.list_IDs = list_IDs
        self.configs = configs
        self.model_image_width = self.configs['train']["model_image_width"]
        self.model_image_height = self.configs['train']["model_image_width"]

        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])
        self.scale_point = configs["train"]["model_image_width"] / configs["train"]["image_original_size"]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        # data process follow super-retinal, original super-retinal use single green channel , and input is uint8, to_tensor操作自动除255归一化
        # 加入随机性，随机左右互换
        # 关键点的size大小是batch_size,10，2
        txt_anno_path = self.configs['train']["anno_file_dir"] + "control_points_" + ID + '_1_2.txt'
        fixed_point = []
        moved_point = []

        with open(txt_anno_path, "r") as anno_file:
            for line_txt in anno_file:
                # 将每行数据分割成四个数
                if (len(fixed_point) < self.configs['train']["keypoint_number"]):
                    point_data = line_txt.strip().split()
                    fixed_point.append([float(point_data[0]) * self.scale_point, float(point_data[1]) * self.scale_point])
                    moved_point.append([float(point_data[2]) * self.scale_point, float(point_data[3]) * self.scale_point])
        fixed_point = np.array(fixed_point)
        moved_point = np.array(moved_point)

        random_num = random.randint(1, 7)
        if (random_num % 2 == 0):
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_1.jpg'
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_2.jpg'
            fixed_point_ = fixed_point
            moved_point_ = moved_point            
        else:
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_2.jpg'
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_1.jpg'
            fixed_point_ = moved_point
            moved_point_ = fixed_point

        fixed_image = Image.open(fixed_path)
        # 转换为灰度图像
        fixed_image = fixed_image.convert('L')
        fixed_image = np.asarray(fixed_image).astype(np.uint8)
        moving_image = Image.open(moving_path)
        # 转换为灰度图像
        moving_image = moving_image.convert('L')
        moving_image = np.asarray(moving_image).astype(np.uint8)
        ## using single channel
        fixed_image = Image.fromarray(fixed_image)
        fixed_image = self.transforms(fixed_image)
        moving_image = Image.fromarray(moving_image)
        moving_image = self.transforms(moving_image)
        
        return fixed_image, moving_image, torch.tensor(fixed_point_), torch.tensor(moved_point_)


class Fire_Dataset_points(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, configs):
        'Initialization'
        self.list_IDs = list_IDs
        self.configs = configs
        self.model_image_width = self.configs['train']["model_image_width"]
        self.model_image_height = self.configs['train']["model_image_width"]

        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])
        self.scale_point = configs["train"]["model_image_width"] / configs["train"]["image_original_size"]
        self.point_select = 2 # 1代表随机，2代表根据梯度来挑选
        #定义高斯卷积核
        kernel_size = 8  # 核的大小
        sigma = 3  # 高斯分布的标准差
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        self.cv_kernel = kernel * kernel.T
        self.score_maps = {}

        gt_dis_map = cv2.imread("/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/utils/guide_dis_map.png", 0)
        gt_dis_map = torch.tensor(gt_dis_map) / 255
        self.gt_dis_map = gt_dis_map

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        # data process follow super-retinal, original super-retinal use single green channel , and input is uint8, to_tensor操作自动除255归一化
        # 加入随机性，随机左右互换
        # 关键点的size大小是batch_size,10，2
        txt_anno_path = self.configs['train']["anno_file_dir"] + "control_points_" + ID + '_1_2.txt'
        fixed_point = []
        moved_point = []

        with open(txt_anno_path, "r") as anno_file:
            for line_txt in anno_file:
                # 将每行数据分割成四个数
                point_data = line_txt.strip().split()
                fixed_point.append([float(point_data[0]) * self.scale_point, float(point_data[1]) * self.scale_point])
                moved_point.append([float(point_data[2]) * self.scale_point, float(point_data[3]) * self.scale_point])
        fixed_point = np.array(fixed_point)
        moved_point = np.array(moved_point)

        random_num = random.randint(1, 7)
        if (random_num % 2 == 0):
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_1.jpg'
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_2.jpg'
            fixed_point_ = fixed_point
            moved_point_ = moved_point            
        else:
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_2.jpg'
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_1.jpg'
            fixed_point_ = moved_point
            moved_point_ = fixed_point

        fixed_image = Image.open(fixed_path)
        # 转换为灰度图像
        fixed_image = fixed_image.convert('L')
        fixed_image = np.asarray(fixed_image).astype(np.uint8)
        moving_image = Image.open(moving_path)
        # 转换为灰度图像
        moving_image = moving_image.convert('L')
        moving_image = np.asarray(moving_image).astype(np.uint8)

        new_im = visulize_points(fixed_point_, fixed_image, self.model_image_width)
        ddd
        ## score map
        if (self.configs['train']["use_score_map"]):
            fix_map = generate_score_map(fixed_point, self.cv_kernel)
            mov_map = generate_score_map(moved_point, self.cv_kernel) 
            resize_ = transforms.Resize((96, 96), interpolation=2)  # 2表示双线性插值
            fix_map = resize_(fix_map).squeeze()
            mov_map = resize_(mov_map).squeeze()

            fix_map = fix_map.view(1,9216).t()
            mov_map = mov_map.view(1,9216)
            gt_score_map = torch.matmul(fix_map, mov_map)
            ## draw the score map
            #visulize_color_map_demo()

            gt_score_map = (gt_score_map + self.gt_dis_map) * 0.5
        else:
            gt_score_map = torch.tensor(0)
       
        # 选取关键点
        if (self.point_select == 1):
            fixed_point_ = fixed_point_[:60, :2]
            moved_point_ = moved_point_[:60, :2]
        elif(self.point_select == 2):
            # 关键点排序，先求取图像梯度大小，根据梯度大小对点进行排序
            magnitude_scaled, direction_scaled = get_gradient(fixed_image)
            fixed_point_, moved_point_= extract_and_sort_points(fixed_point_, moved_point_, magnitude_scaled)
            fixed_point_ = fixed_point_[:60, :2]
            moved_point_ = moved_point_[:60, :2]
        
        # 可视化调试
        #new_im = visulize_points(fixed_point_, fixed_image)
        
        ## using single channel
        fixed_image = Image.fromarray(fixed_image)
        fixed_image = self.transforms(fixed_image)
        moving_image = Image.fromarray(moving_image)
        moving_image = self.transforms(moving_image)

        mask_guide = generate_guide_map(fixed_point_ , self.model_image_width)
        mask_guide = torch.tensor(mask_guide).to(torch.float32)

        return fixed_image, moving_image, torch.tensor(fixed_point_), torch.tensor(moved_point_), mask_guide, gt_score_map

class Fire_Dataset_points_val(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, configs):
        'Initialization'
        self.list_IDs = list_IDs
        self.configs = configs
        self.model_image_width = self.configs['train']["model_image_width"]
        self.model_image_height = self.configs['train']["model_image_width"]

        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])
        self.scale_point = configs["train"]["model_image_width"] / configs["train"]["image_original_size"]
        self.point_select = 2 # 1代表随机，2代表根据梯度来挑选
        #定义高斯卷积核
        kernel_size = 8  # 核的大小
        sigma = 3  # 高斯分布的标准差
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        self.cv_kernel = kernel * kernel.T
        self.score_maps = {}

        gt_dis_map = cv2.imread("/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/utils/guide_dis_map.png", 0)
        gt_dis_map = torch.tensor(gt_dis_map) / 255
        self.gt_dis_map = gt_dis_map

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        txt_anno_path = "/home/yepeng_liu/code_python/keypoints_benchmark/unofficial-voxelmorph/VoxelMorph-PyTorch-master/train_datas/super_gt_generate_fire/gt_aling_txt/" + "control_points_" + ID + '_1_2.txt'
        fixed_point = []
        moved_point = []

        with open(txt_anno_path, "r") as anno_file:
            for line_txt in anno_file:
                # 将每行数据分割成四个数
                if (len(fixed_point) < self.configs['train']["keypoint_number"]):
                    point_data = line_txt.strip().split()
                    fixed_point.append([float(point_data[0]) * self.scale_point, float(point_data[1]) * self.scale_point])
                    moved_point.append([float(point_data[2]) * self.scale_point, float(point_data[3]) * self.scale_point])
        fixed_point = np.array(fixed_point)
        moved_point = np.array(moved_point)

       
        fixed_path = self.configs['train']["train_fire_dir"] + ID + '_1.jpg'
        moving_path = self.configs['train']["train_fire_dir"] + ID + '_2.jpg'
        fixed_point_ = fixed_point
        moved_point_ = moved_point            
       

        fixed_image = Image.open(fixed_path)
        # 转换为灰度图像
        fixed_image = fixed_image.convert('L')
        fixed_image = np.asarray(fixed_image).astype(np.uint8)
        moving_image = Image.open(moving_path)
        # 转换为灰度图像
        moving_image = moving_image.convert('L')
        moving_image = np.asarray(moving_image).astype(np.uint8)
        ## score map
        if (self.configs['train']["use_score_map"]):
            fix_map = generate_score_map(fixed_point, self.cv_kernel)
            mov_map = generate_score_map(moved_point, self.cv_kernel) 
            resize_ = transforms.Resize((96, 96), interpolation=2)  # 2表示双线性插值
            fix_map = resize_(fix_map).squeeze()
            mov_map = resize_(mov_map).squeeze()

            fix_map = fix_map.view(1,9216).t()
            mov_map = mov_map.view(1,9216)
            gt_score_map = torch.matmul(fix_map, mov_map)
            ## draw the score map
            #visulize_color_map_demo()

            gt_score_map = (gt_score_map + self.gt_dis_map) * 0.5
        else:
            gt_score_map = torch.tensor(0)
       
        # # 选取关键点
        # if (self.point_select == 1):
        #     fixed_point_ = fixed_point_[:60, :2]
        #     moved_point_ = moved_point_[:60, :2]
        # elif(self.point_select == 2):
        #     # 关键点排序，先求取图像梯度大小，根据梯度大小对点进行排序
        #     magnitude_scaled, direction_scaled = get_gradient(fixed_image)
        #     fixed_point_, moved_point_= extract_and_sort_points(fixed_point_, moved_point_, magnitude_scaled)
        #     fixed_point_ = fixed_point_[:60, :2]
        #     moved_point_ = moved_point_[:60, :2]
        
        # 可视化调试
        new_im = visulize_points(fixed_point_, fixed_image)
        ddd
        ## using single channel
        fixed_image = Image.fromarray(fixed_image)
        fixed_image = self.transforms(fixed_image)
        moving_image = Image.fromarray(moving_image)
        moving_image = self.transforms(moving_image)

        mask_guide = generate_guide_map(fixed_point_ ,  self.model_image_width)
        mask_guide = torch.tensor(mask_guide).to(torch.float32)

        return fixed_image, moving_image, torch.tensor(fixed_point_), torch.tensor(moved_point_), mask_guide, gt_score_map

class mixture_Dataset_points(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, configs):
        'Initialization'
        self.list_IDs = list_IDs
        self.configs = configs
        self.model_image_width = self.configs['train']["model_image_width"]
        self.model_image_height = self.configs['train']["model_image_width"]

        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])
        self.scale_point = 1

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        # data process follow super-retinal, original super-retinal use single green channel , and input is uint8, to_tensor操作自动除255归一化
        # 加入随机性，随机左右互换
        # 关键点的size大小是batch_size,10，2
        txt_anno_path = self.configs['train']["anno_file_dir"] + "control_points_" + ID + '_1_2.txt'
        fixed_point = []
        moved_point = []
        # 判断多少行
        with open(txt_anno_path, 'r') as file:
            line_count = sum(1 for line in file)

        if (line_count < self.configs['train']["keypoint_number"]):
            with open("/home/yepeng_liu/code_python/keypoints_benchmark/retina_data_train/mix_generate/fire-fundus-image-registration-dataset_fire_gt_super/control_points_mix_439_1_1_2.txt", "r") as anno_file:
                for line_txt in anno_file:
                    # 将每行数据分割成四个数
                    if (len(fixed_point) < self.configs['train']["keypoint_number"]):
                        point_data = line_txt.strip().split()
                        fixed_point.append([float(point_data[0]) * self.scale_point, float(point_data[1]) * self.scale_point])
                        moved_point.append([float(point_data[2]) * self.scale_point, float(point_data[3]) * self.scale_point])
        else:
            with open(txt_anno_path, "r") as anno_file:
                
                for line_txt in anno_file:
                    # 将每行数据分割成四个数
                    if (len(fixed_point) < self.configs['train']["keypoint_number"]):
                        point_data = line_txt.strip().split()
                        fixed_point.append([float(point_data[0]) * self.scale_point, float(point_data[1]) * self.scale_point])
                        moved_point.append([float(point_data[2]) * self.scale_point, float(point_data[3]) * self.scale_point])
        

        fixed_point = np.array(fixed_point)
        moved_point = np.array(moved_point)
        
        random_num = random.randint(1, 7)
        if (random_num % 2 == 0):
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_1.jpg'
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_2.jpg'
           
            fixed_point_ = fixed_point
            moved_point_ = moved_point            
        else:
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_2.jpg'
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_1.jpg'
            
            fixed_point_ = moved_point
            moved_point_ = fixed_point
        
        fixed_image = Image.open(fixed_path)
        # 转换为灰度图像
        fixed_image = fixed_image.convert('L')
        fixed_image = np.asarray(fixed_image).astype(np.uint8)
        moving_image = Image.open(moving_path)
        # 转换为灰度图像
        moving_image = moving_image.convert('L')
        moving_image = np.asarray(moving_image).astype(np.uint8)
        ## using single channel
        fixed_image = Image.fromarray(fixed_image)
        fixed_image = self.transforms(fixed_image)
        moving_image = Image.fromarray(moving_image)
        moving_image = self.transforms(moving_image)
      
        
        return fixed_image, moving_image, torch.tensor(fixed_point_), torch.tensor(moved_point_)

def generate_score_map(fixed_point, kennel_size, size=768):
    # 生成score map
     ## 生成attention score map
    fix_map = generate_guide_map(fixed_point , size)
    # 使用高斯卷积来模糊图像
    fix_map = cv2.filter2D(fix_map, -1, kennel_size)
    fix_map = torch.tensor(fix_map).to(torch.float32).unsqueeze(0)
    max_value = torch.max(fix_map)
    min_value = torch.min(fix_map)
    # 进行归一化
    fix_map = (fix_map - min_value) / (max_value - min_value)
    
    return fix_map

# 可视化生成的响应图
def visulize_color_map(score_map):
    score_map_np = score_map.numpy() * 255
    score_map_np = score_map_np.astype(np.uint8)
    # 将单通道矩阵转换为3通道图像
    colored_image = cv2.applyColorMap(score_map_np, cv2.COLORMAP_HOT)
    cv2.imwrite("./color_score_map.png", colored_image)

def visulize_color_map_demo():
     #定义高斯卷积核
    kernel_size = 3  # 核的大小
    sigma = 3  # 高斯分布的标准差
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kennel_size = kernel * kernel.T

    key_map1 = np.zeros((20,20))
    key_map2 = np.zeros((20,20))
    keypoints1 = [[17,15], [4,5], [9,3], [12,8], [10,11]]
    keypoints2 = [[17,13], [5,5], [8,2], [14,8], [9,10]]
    for i in range(len(keypoints1)):
        key_map1[keypoints1[i][1]][keypoints1[i][0]] = 1
        key_map2[keypoints2[i][1]][keypoints2[i][0]] = 1
    # 使用高斯卷积来模糊图像
    key_map1 = cv2.filter2D(key_map1, -1, kennel_size)
    key_map1 = torch.tensor(key_map1).to(torch.float32).unsqueeze(0)
    max_value = torch.max(key_map1)
    min_value = torch.min(key_map1)
    # 进行归一化
    key_map1 = (key_map1 - min_value) / (max_value - min_value)
    key_map1_ = key_map1.permute(1,2,0).numpy()
    np.save("./scoremap_left.npy",key_map1_)

    key_map2 = cv2.filter2D(key_map2, -1, kennel_size)
    key_map2 = torch.tensor(key_map2).to(torch.float32).unsqueeze(0)
    max_value = torch.max(key_map2)
    min_value = torch.min(key_map2)
    # 进行归一化
    key_map2 = (key_map2 - min_value) / (max_value - min_value)
    key_map2_ = key_map2.permute(1,2,0).numpy()
    np.save("./scoremap_right.npy",key_map2_)

    key_map1 = key_map1.squeeze()
    key_map2 = key_map2.squeeze()
    key_map1 = key_map1.view(1,400).t()
    key_map2 = key_map2.view(1,400)
    gt_score_map = torch.matmul(key_map1, key_map2)
    gt_score_map = gt_score_map.numpy()
    np.save("./final_score_map.npy", gt_score_map)


    key_map1 = key_map1.numpy() * 255
    key_map1 = key_map1.astype(np.uint8)
    print(key_map1, key_map1.dtype, key_map1.shape)
    # 将单通道矩阵转换为3通道图像
    colored_image = cv2.applyColorMap(key_map1, cv2.COLORMAP_AUTUMN)
    cv2.imwrite("./color_score_map.png", colored_image)

    
class mixture_Dataset_points_select(data.Dataset):
    """
    Dataset class for converting the data into batches.
    The data.Dataset class is a pyTorch class which help
    in speeding up  this process with effective parallelization
    """
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, configs):
        'Initialization'
        self.list_IDs = list_IDs
        self.configs = configs
        self.model_image_width = self.configs['train']["model_image_width"]
        self.model_image_height = self.configs['train']["model_image_width"]

        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])
        self.scale_point = self.model_image_width / self.configs['train']["image_original_size"]
        self.point_select = 2 # 1代表随机，2代表根据梯度来挑选
        #定义高斯卷积核
        kernel_size = 8  # 核的大小
        sigma = 3  # 高斯分布的标准差
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        self.cv_kernel = kernel * kernel.T
        self.score_maps = {}

        gt_dis_map = cv2.imread("/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/utils/guide_dis_map.png", 0)
        gt_dis_map = torch.tensor(gt_dis_map) / 255
        self.gt_dis_map = gt_dis_map

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        txt_anno_path = self.configs['train']["anno_file_dir"] + "control_points_" + ID + '_1_2.txt'
        fixed_point = []
        moved_point = []
        
        with open(txt_anno_path, "r") as anno_file:
            for line_txt in anno_file:
                # 将每行数据分割成四个数
                # if (len(fixed_point) < self.configs['train']["keypoint_number"]):
                point_data = line_txt.strip().split()
                fixed_point.append([float(point_data[0]) * self.scale_point, float(point_data[1]) * self.scale_point])
                moved_point.append([float(point_data[2]) * self.scale_point, float(point_data[3]) * self.scale_point])
        fixed_point = np.array(fixed_point)
        moved_point = np.array(moved_point)
        

        random_num = random.randint(1, 7)
        if (random_num % 2 == 0):
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_1.jpg'
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_2.jpg'
            fixed_point_ = fixed_point
            moved_point_ = moved_point            
        else:
            moving_path = self.configs['train']["train_fire_dir"] + ID + '_2.jpg'
            fixed_path = self.configs['train']["train_fire_dir"] + ID + '_1.jpg'
            fixed_point_ = moved_point
            moved_point_ = fixed_point
        # 转换为灰度图像
        fixed_image = Image.open(fixed_path)
        fixed_image = fixed_image.convert('L')
        moving_image = Image.open(moving_path)
        moving_image = moving_image.convert('L')
        fixed_image = np.asarray(fixed_image).astype(np.uint8)
        moving_image = np.asarray(moving_image).astype(np.uint8)
        fixed_image = cv2.resize(fixed_image, (self.model_image_width,self.model_image_width))
        moving_image = cv2.resize(moving_image, (self.model_image_width,self.model_image_width))
        ## score map
        if (self.configs['train']["use_score_map"]):
            fix_map = generate_score_map(fixed_point, self.cv_kernel)
            mov_map = generate_score_map(moved_point, self.cv_kernel) 
            resize_ = transforms.Resize((96, 96), interpolation=2)  # 2表示双线性插值
            fix_map = resize_(fix_map).squeeze()
            mov_map = resize_(mov_map).squeeze()

            fix_map = fix_map.view(1,9216).t()
            mov_map = mov_map.view(1,9216)
            gt_score_map = torch.matmul(fix_map, mov_map)
            ## draw the score map
            #visulize_color_map_demo()

            gt_score_map = (gt_score_map + self.gt_dis_map) * 0.5
        else:
            gt_score_map = torch.tensor(0)
       
        # 选取关键点
        if (self.point_select == 1):
            fixed_point_ = fixed_point_[:30, :2]
            moved_point_ = moved_point_[:30, :2]
        elif(self.point_select == 2):
            # 关键点排序，先求取图像梯度大小，根据梯度大小对点进行排序
            magnitude_scaled, direction_scaled = get_gradient(fixed_image)
            fixed_point_, moved_point_= extract_and_sort_points(fixed_point_, moved_point_, magnitude_scaled)
            fixed_point_ = fixed_point_[:30, :2]
            moved_point_ = moved_point_[:30, :2]
        
        # new_im = visulize_points(fixed_point_, fixed_image, self.model_image_width)
        # cv2.imwrite("./flori_debug_fix.jpg", new_im)
        # new_im2 = visulize_points(moved_point_, moving_image, self.model_image_width)
        # cv2.imwrite("./flori_debug_mov.jpg", new_im2)

        # ddd
        
       ## using single channel
        fixed_image = Image.fromarray(fixed_image)
        fixed_image = self.transforms(fixed_image)
        moving_image = Image.fromarray(moving_image)
        moving_image = self.transforms(moving_image)

        mask_guide = generate_guide_map(fixed_point_ , self.model_image_width)
        mask_guide = torch.tensor(mask_guide).to(torch.float32)

        return fixed_image, moving_image, torch.tensor(fixed_point_), torch.tensor(moved_point_), mask_guide, gt_score_map
       



class flori_points(data.Dataset):
    def __init__(self, list_IDs, configs):
        'Initialization'
        self.list_IDs = list_IDs
        self.configs = configs
        self.model_image_width = self.configs['train']["model_image_width"]
        self.model_image_height = self.configs['train']["model_image_width"]

        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])
        self.scale_point = self.model_image_width / self.configs['train']["image_original_size"]
        self.point_select = 2 # 1代表随机，2代表根据梯度来挑选
        #定义高斯卷积核
        kernel_size = 8  # 核的大小
        sigma = 3  # 高斯分布的标准差
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        self.cv_kernel = kernel * kernel.T
        self.score_maps = {}

        gt_dis_map = cv2.imread("/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/utils/guide_dis_map.png", 0)
        gt_dis_map = torch.tensor(gt_dis_map) / 255
        self.gt_dis_map = gt_dis_map
         # 图像预处理操作
        self.my_image_process = image_train_process(data_aug_type= configs["train"]["data_aug_type"])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        txt_anno_path = self.configs['train']["anno_file_dir"]  + ID + '.txt'
        fixed_point = []
        moved_point = []
        with open(txt_anno_path, "r") as anno_file:
            for line_txt in anno_file:
                point_data = line_txt.strip().split()
                fixed_point.append([float(point_data[0]) * self.scale_point, float(point_data[1]) * self.scale_point])
                moved_point.append([float(point_data[2]) * self.scale_point, float(point_data[3]) * self.scale_point])
        fixed_point = np.array(fixed_point)
        moved_point = np.array(moved_point)
        

        random_num = random.randint(1, 7)
        if (random_num % 2 == 0):
            moving_path = self.configs['train']["train_fire_dir"] + "Raw_FA_" + ID.split('_')[3] + "_Subject_" + ID.split('_')[-1] + ".jpg"
            fixed_path = self.configs['train']["train_fire_dir"] + "Montage_Subject_" + ID.split('_')[-1] + ".jpg"
            fixed_point_ = fixed_point
            moved_point_ = moved_point            
        else:
            moving_path = self.configs['train']["train_fire_dir"] + "Montage_Subject_" + ID.split('_')[-1] + ".jpg"
            fixed_path = self.configs['train']["train_fire_dir"] + "Raw_FA_" + ID.split('_')[3] + "_Subject_" + ID.split('_')[-1] + ".jpg"
            fixed_point_ = moved_point
            moved_point_ = fixed_point
        # 转换为灰度图像
        fixed_image = Image.open(fixed_path)
        fixed_image = np.asarray(fixed_image).astype(np.uint8)
        moving_image = Image.open(moving_path)
        moving_image = np.asarray(moving_image).astype(np.uint8)
        fixed_image = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY)
        moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2GRAY)
        # 生成掩膜
        mask = np.logical_and(fixed_image != 0, moving_image != 0)
        # 将掩膜应用于两幅图像
        moving_image = moving_image * mask
        fixed_image = fixed_image * mask
        
        # fixed_image = self.my_image_process.process_train_data(fixed_image)
        # moving_image = self.my_image_process.process_train_data(moving_image)

        ## score map
        if (self.configs['train']["use_score_map"]):
            fix_map = generate_score_map(fixed_point, self.cv_kernel)
            mov_map = generate_score_map(moved_point, self.cv_kernel) 
            resize_ = transforms.Resize((96, 96), interpolation=2)  # 2表示双线性插值
            fix_map = resize_(fix_map).squeeze()
            mov_map = resize_(mov_map).squeeze()

            fix_map = fix_map.view(1,9216).t()
            mov_map = mov_map.view(1,9216)
            gt_score_map = torch.matmul(fix_map, mov_map)
            ## draw the score map
            #visulize_color_map_demo()

            gt_score_map = (gt_score_map + self.gt_dis_map) * 0.5
        else:
            gt_score_map = torch.tensor(0)
       
        # 选取关键点
        if (self.point_select == 1):
            fixed_point_ = fixed_point_[:self.configs['train']["keypoint_number"], :2]
            moved_point_ = moved_point_[:self.configs['train']["keypoint_number"], :2]
        elif(self.point_select == 2):
            # 关键点排序，先求取图像梯度大小，根据梯度大小对点进行排序
            magnitude_scaled, direction_scaled = get_gradient(fixed_image)
            fixed_point_, moved_point_= extract_and_sort_points(fixed_point_, moved_point_, magnitude_scaled)
            fixed_point_ = fixed_point_[:14, :2]
            moved_point_ = moved_point_[:14, :2]
            fixed_point_[fixed_point_>self.model_image_width] =self.model_image_width -1
            moved_point_[moved_point_>self.model_image_width] =self.model_image_width -1
            
        
        # # 可视化调试
        # # 定义结构元素（这里使用矩形结构元素，可以根据需要选择不同形状）
        # kernel = np.ones((11, 11), np.uint8)
        # # 应用白顶帽（White Top Hat）操作
        # fixed_image = cv2.morphologyEx(fixed_image, cv2.MORPH_TOPHAT, kernel)
        # moving_image = cv2.morphologyEx(moving_image, cv2.MORPH_TOPHAT, kernel)
       
        # new_im = visulize_points(fixed_point_, fixed_image, self.model_image_width)
        # cv2.imwrite("./flori_debug_fix.jpg", new_im)
        # new_im2 = visulize_points(moved_point_, moving_image, self.model_image_width)
        # cv2.imwrite("./flori_debug_mov.jpg", new_im2)

        # ddd
        
        ## using single channel
        fixed_image = Image.fromarray(fixed_image)
        fixed_image = self.transforms(fixed_image)
        moving_image = Image.fromarray(moving_image)
        moving_image = self.transforms(moving_image)

        mask_guide = generate_guide_map(fixed_point_ , self.model_image_width)
        mask_guide = torch.tensor(mask_guide).to(torch.float32)

        return fixed_image, moving_image, torch.tensor(fixed_point_), torch.tensor(moved_point_), mask_guide, gt_score_map

class flori_points_val(data.Dataset):
    def __init__(self, list_IDs, configs):
        'Initialization'
        self.list_IDs = list_IDs
        self.configs = configs
        self.train_fire_dir = "/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/flori_data/val/align_img/"
        self.anno_file_dir = "/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/flori_data/val/align_gt/"
        self.model_image_width = self.configs['train']["model_image_width"]
        self.model_image_height = self.configs['train']["model_image_width"]

        self.transforms = transforms.Compose([
            transforms.Resize((self.model_image_width, self.model_image_height)),
            transforms.ToTensor(),
        ])
        self.scale_point = self.model_image_width / 2912
        self.point_select = 2 # 1代表随机，2代表根据梯度来挑选
        #定义高斯卷积核
        kernel_size = 8  # 核的大小
        sigma = 3  # 高斯分布的标准差
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        self.cv_kernel = kernel * kernel.T
        self.score_maps = {}

        gt_dis_map = cv2.imread("/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/utils/guide_dis_map.png", 0)
        gt_dis_map = torch.tensor(gt_dis_map) / 255
        self.gt_dis_map = gt_dis_map
         # 图像预处理操作
        self.my_image_process = image_train_process(data_aug_type= configs["train"]["data_aug_type"])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        txt_anno_path = self.anno_file_dir  + ID + '.txt'
        fixed_point = []
        moved_point = []
        with open(txt_anno_path, "r") as anno_file:
            for line_txt in anno_file:
                point_data = line_txt.strip().split()
                fixed_point.append([float(point_data[0]) * self.scale_point, float(point_data[1]) * self.scale_point])
                moved_point.append([float(point_data[2]) * self.scale_point, float(point_data[3]) * self.scale_point])
        fixed_point = np.array(fixed_point)
        moved_point = np.array(moved_point)
        
        
        moving_path = self.train_fire_dir + "Raw_FA_" + ID.split('_')[3] + "_Subject_" + ID.split('_')[-1] + ".jpg"
        fixed_path = self.train_fire_dir + "Montage_Subject_" + ID.split('_')[-1] + ".jpg"
        fixed_point_ = fixed_point
        moved_point_ = moved_point            
        # 转换为灰度图像
        fixed_image = Image.open(fixed_path)
        fixed_image = np.asarray(fixed_image).astype(np.uint8)
        moving_image = Image.open(moving_path)
        moving_image = np.asarray(moving_image).astype(np.uint8)
        fixed_image = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY)
        moving_image = cv2.cvtColor(moving_image, cv2.COLOR_BGR2GRAY)
        # 生成掩膜
        mask = np.logical_and(fixed_image != 0, moving_image != 0)
        # 将掩膜应用于两幅图像
        moving_image = moving_image * mask
        fixed_image = fixed_image * mask
        
        ## score map
        if (self.configs['train']["use_score_map"]):
            fix_map = generate_score_map(fixed_point, self.cv_kernel)
            mov_map = generate_score_map(moved_point, self.cv_kernel) 
            resize_ = transforms.Resize((96, 96), interpolation=2)  # 2表示双线性插值
            fix_map = resize_(fix_map).squeeze()
            mov_map = resize_(mov_map).squeeze()

            fix_map = fix_map.view(1,9216).t()
            mov_map = mov_map.view(1,9216)
            gt_score_map = torch.matmul(fix_map, mov_map)
            ## draw the score map
            #visulize_color_map_demo()

            gt_score_map = (gt_score_map + self.gt_dis_map) * 0.5
        else:
            gt_score_map = torch.tensor(0)
       
        # 选取关键点
        if (self.point_select == 1):
            fixed_point_ = fixed_point_[:self.configs['train']["keypoint_number"], :2]
            moved_point_ = moved_point_[:self.configs['train']["keypoint_number"], :2]
        elif(self.point_select == 2):
            # 关键点排序，先求取图像梯度大小，根据梯度大小对点进行排序
            magnitude_scaled, direction_scaled = get_gradient(fixed_image)
            fixed_point_, moved_point_= extract_and_sort_points(fixed_point_, moved_point_, magnitude_scaled)
            fixed_point_ = fixed_point_[:14, :2]
            moved_point_ = moved_point_[:14, :2]
            fixed_point_[fixed_point_>self.model_image_width] =self.model_image_width -1
            moved_point_[moved_point_>self.model_image_width] =self.model_image_width -1
             
        ## using single channel
        fixed_image = Image.fromarray(fixed_image)
        fixed_image = self.transforms(fixed_image)
        moving_image = Image.fromarray(moving_image)
        moving_image = self.transforms(moving_image)

        mask_guide = generate_guide_map(fixed_point_ , self.model_image_width)
        mask_guide = torch.tensor(mask_guide).to(torch.float32)

        return fixed_image, moving_image, torch.tensor(fixed_point_), torch.tensor(moved_point_), mask_guide, gt_score_map
# config_path = '../config/train_config/train.yaml'
# if os.path.exists(config_path):
#     with open(config_path) as f:
#         config = yaml.safe_load(f)
# else:
#     raise FileNotFoundError("Config File doesn't Exist")
# print("config is ", config)


if __name__ == "__main__":
    """
    dataloader的单元测试用例
    """
    print(" unit test for dataset")
    params = {'batch_size': config['train']["batch_size"],
              'shuffle': False,
              'num_workers': 6,
              'worker_init_fn': np.random.seed(42)
              }
    # 1, mixture datasets read for sequence
    partition = {}
    partition['train'] = list(set([x.split('_')[0]
                         for x in os.listdir("/home/yepeng_liu/code_python/keypoints_benchmark/unofficial-voxelmorph/VoxelMorph-PyTorch-master/fire_train")]))
        
    # Generators
    configs = config
    # training_set = Mixture_Dataset(partition['train'], configs)
    # training_generator = data.DataLoader(training_set, **params)
    # print("original_datset is :,", len(partition['train']), partition['train'])

    # validation_set = Mixture_Dataset(partition['validation'], configs)
    # validation_generator = data.DataLoader(validation_set, **params)

    training_set = Fire_Dataset_seg_points(partition['train'], configs=config)
    training_generator = data.DataLoader(training_set, **params)
    for epoch in range(2):
        print("=========================================")
        for fix, mov,fix_points,mov_points in training_generator:
        # 在这里进行模型训练
            print(fix.shape, mov.shape, fix_points.shape, mov_points.shape)
            print("==fix", fix_points[0,:,:])
            print("==mov", mov_points[0,:,:])
            pass