from models import voxelmorph2d as vm2d
from models import voxelmorph3d as vm3d
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
from skimage.transform import resize
import multiprocessing as mp
from tqdm import tqdm
import gc
import time
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
from PIL import Image
import yaml
import datetime
import random
import shutil
from utils.post_process import *
from datasets.medical_dataset import Mixture_Dataset, Fire_Dataset_points, Fire_Dataset_points_val,Fire_Dataset_seg, flori_points, mixture_Dataset_points, mixture_Dataset_points_select

current_time = datetime.datetime.now()
print("============== data time is ================", current_time)

config_path = './config/train_config/train.yaml'
if os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError("Config File doesn't Exist")
print("config is ", config)

def seed_everything(seed=37):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

class VoxelMorph():
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """

    def __init__(self, input_dims, config_train, is_2d=False, use_gpu=False):
        self.dims = input_dims
        self.config_train = config_train
        self.device = torch.device(config_train['train']["device"] if use_gpu else "cpu")
        if is_2d:
            self.vm = vm2d
            print("use_gpu",use_gpu)
            self.voxelmorph = vm2d.VoxelMorph2d(input_dims[0] * 2, config_train, use_gpu, self.device)
        else:
            self.vm = vm3d
            self.voxelmorph = vm3d.VoxelMorph3d(input_dims[0] * 2, use_gpu, self.device)
        
        # load the model
        if (config_train['train']["pretrain_model"] != "None"):
            if (config_train['train']["big_resolution"] == False):
                checkpoint = torch.load(config_train['train']["pretrain_model"], map_location= self.device)
                ## 删除第一层和最后一层形变层的预训练参数
                del checkpoint["unet.conv_encode1.0.weight"]
                del checkpoint["unet.conv_encode1.0.bias"]
                for key in set(checkpoint.keys()):
                    if "final_layer" in key:
                        del checkpoint[key]
            else:
                checkpoint = torch.load(config_train['train']["pretrain_model_big"], map_location= self.device)
                 ## 删除第一层和最后一层形变层的预训练参数
                del checkpoint["unet.conv_encode0.0.weight"]
                del checkpoint["unet.conv_encode0.0.bias"]
                for key in set(checkpoint.keys()):
                    if "final_layer" in key:
                        del checkpoint[key]
            
            self.voxelmorph.load_state_dict(checkpoint, strict=False)
            keys_a = set(self.voxelmorph.state_dict().keys())
            keys_b = set(checkpoint.keys())
            # 查找匹配的键和未匹配的键
            matching_keys = keys_a.intersection(keys_b)
            unmatched_keys_a = keys_a.difference(keys_b)
            unmatched_keys_b = keys_b.difference(keys_a)
            # 打印结果
            print(f'匹配的键数量：{len(matching_keys)}')
            print(f'未匹配的键数量（模型 a 中的未匹配键）：{len(unmatched_keys_a)}')
            print(f'未匹配的键数量（模型 b 中的未匹配键）：{len(unmatched_keys_b)}')

        #self.optimizer = optim.SGD(
        #    self.voxelmorph.parameters(), lr=self.config_train['train']["lr"], momentum=0.99)
        
        self.optimizer = optim.Adam(self.voxelmorph.parameters(), lr=self.config_train['train']["lr"], amsgrad=True)

        self.params = {'batch_size': config_train['train']["batch_size"],
                       'shuffle': True,
                       'num_workers': 6,
                       'worker_init_fn': np.random.seed(42)
                       }
       
        ## 损失函数用mse loss
        #self.loss_function = self.vm.MSE()
        self.loss_function = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.key_loss = nn.MSELoss(reduction="sum")
        self.loss_smooth = self.vm.smooth_loss()
        self.loss_ncc = self.vm.ncc_loss_fun(self.device, win=config_train['train']["window_size"])
        ## training params
        self.epoch = 0
        self.model_save_epoch = config_train['train']["save_epochs"]
        self.model_save_path = config_train['train']["save_files_dir"]
        self.lr = self.config_train['train']["lr"]
        self.best_dice_score = 100000.0
        self.keypoints_process_ = keypoints_process()
        self.train_image_size = config_train['train']["model_image_width"]

    def adjust_learning_rate(self, power=0.9):
        
        if (self.epoch in self.config_train['train']["ir_epochs"]):
            for index in range(len(self.config_train['train']["ir_epochs"])):
                if (self.epoch ==  self.config_train['train']["ir_epochs"][index]):
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] =  self.config_train['train']["lr"] * (0.5 ** (index + 1))
                        self.lr = self.config_train['train']["lr"] * (0.5 ** (index + 1))

    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError as e:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def calculate_loss(self, y, ytrue, deformation_matrix,mask_guide,  n=9, lamda=0.01, is_training=True):
        if (self.config_train['train']["loss"] == "l2"):
            sim_loss = self.loss_function(ytrue, y) * 1 #目前来看，权重不能设置的太大，太大会导致y方向完全为0
            if (self.config_train['train']["loss_with_smooth"] == True):
                smooth_loss = 1 * self.loss_smooth.smooothing_loss(deformation_matrix)
            return  sim_loss, smooth_loss
        elif (self.config_train['train']["loss"] == "ncc"):
            sim_loss = self.loss_ncc.ncc_loss(ytrue, y, mask_guide)
            if (self.config_train['train']["loss_with_smooth"] == True):
                smooth_loss = 1 * self.loss_smooth.smooothing_loss(deformation_matrix)
            
            return  sim_loss, smooth_loss
    
       
    def save_model(self, model_path):
        if self.epoch % self.model_save_epoch == 0:
            print(f'save model for epoch{self.epoch}')
            save_path = os.path.join(model_path, self.config_train['train']["model_save_prefix"]) + str(self.epoch) + ".pth" 
            torch.save(self.voxelmorph.state_dict(), save_path)

    def train_model(self, batch_moving, batch_fixed, fix_points, mov_points ,mask_guide,gt_score_map, n=9, lamda=0.01, return_metric_score=True):  
        self.adjust_learning_rate()
        # 数据转到cuda上
        batch_fixed, batch_moving = batch_fixed.to(
            self.device), batch_moving.to(self.device)
        
        fix_points, mov_points = fix_points.to(self.device), mov_points.to(self.device)
        fix_points = torch.round(fix_points).to(torch.int32)
        # mov的关键点不需要转成int，搞成float32即可
        mov_points = mov_points.to(torch.float32)
        mask_guide = mask_guide.to(self.device)
        gt_score_map = gt_score_map.to(self.device)
        if (self.config_train['train']["use_score_map"]):
            registered_image, deformation_matrix, attention_score_matrix = self.voxelmorph(batch_moving, batch_fixed , mask_guide)
            #attention_loss = self.l1_loss(gt_score_map, attention_score_matrix)
            attention_loss = F.smooth_l1_loss(gt_score_map, attention_score_matrix, reduction="mean", beta=1.0)
        else:
            registered_image, deformation_matrix = self.voxelmorph(batch_moving, batch_fixed , mask_guide)
        self.optimizer.zero_grad()
        # 计算相似度损失和平滑损失
        sim_loss, smooth_loss = self.calculate_loss(
            registered_image, batch_fixed, deformation_matrix, mask_guide, n, lamda)
        # 计算关键点的损失
        # 建立标准网格以及得到目前的采样器, 目前通道顺序应该是n,c,h,w。y坐标在前，x坐标在后
        grid_intergre = self.voxelmorph.transformer.grid + deformation_matrix
        # 根据fix图像坐标，取源头坐标mov*,关键点n,10,2
        key_point_loss = torch.tensor(0.0).to(self.device)
        batch_size = fix_points.shape[0]
        for b_i in range(batch_size):
            fix_ = fix_points[b_i] # 10*2
            mov_ = mov_points[b_i]
            grid_ = grid_intergre[b_i]
            # 用torch操作直接实现,fix_， mov_ 对应的是单通道的点（10，2），grid_intergre为（n,c,h,w)
            # 处理grid_intergre变成(768*768) *2, y方向在前，x方向在后
            grid_ = grid_.squeeze().permute(1,2,0)
            grid_flatten = grid_.view(-1,2)
            # 处理fix_点的值，将坐标点变成10*2
            fix_index = torch.zeros(self.config_train['train']["keypoint_number"],1).to(torch.int32)   
            fix_index[:,0] = fix_[:,1] * self.train_image_size + fix_[:,0]
            # 根据index取对应的坐标, 得到mov_pred
            mov_pred = grid_flatten[fix_index].squeeze()
            mov_pred = mov_pred[...,[1,0]]
            mov_ = mov_ / self.train_image_size
            mov_pred = mov_pred / self.train_image_size
            key_point_loss_ =self.key_loss(mov_, mov_pred)
            #key_point_loss_ = F.smooth_l1_loss(mov_, mov_pred, reduction="mean", beta=1.0)
            key_point_loss += key_point_loss_
            
        # 总的loss
        key_point_loss = key_point_loss / batch_size

        weight_list = self.config_train['train']["loss_weights"]
        if (self.config_train['train']["use_score_map"]):
            train_loss = weight_list[0] * sim_loss + weight_list[1] * smooth_loss + weight_list[2] * key_point_loss + attention_loss * weight_list[3]
            # if (np.random.randint(5) >3):
            #     print(f"iter {self.epoch}, ir is {self.lr}, sim_loss is {sim_loss.item()}, smooth loss is {smooth_loss.item()}, key {key_point_loss.item()}, {attention_loss.item()}")
        else:
            train_loss = weight_list[0] * sim_loss + weight_list[1] * smooth_loss + weight_list[2] * key_point_loss 
            # if (np.random.randint(5) >3):
            #     print(f"iter {self.epoch}, ir is {self.lr}, sim_loss is {sim_loss.item()}, smooth loss is { weight_list[1] * smooth_loss.item()}, key {weight_list[2] * key_point_loss.item()}")
        train_loss.backward()
        self.optimizer.step()
        if return_metric_score:
            train_dice_score = self.vm.dice_score(
                registered_image, batch_fixed)
            if (self.config_train['train']["use_score_map"]):
                return train_loss.item(), train_dice_score.item(), sim_loss.item(), smooth_loss.item(), key_point_loss.item(), attention_loss.item()
            else:
                return train_loss.item(), train_dice_score.item(),  sim_loss.item(), weight_list[1] * smooth_loss.item(), weight_list[2] * key_point_loss.item(), torch.tensor(0)
        return train_loss.item()

    def get_test_loss(self, batch_moving, batch_fixed, n=9, lamda=0.01):
        with torch.set_grad_enabled(False):
            batch_fixed, batch_moving = batch_fixed.to(
            self.device), batch_moving.to(self.device)
            registered_image, deformation_matrix = self.voxelmorph(batch_moving, batch_fixed) 
            val_dice_score = self.vm.dice_score(registered_image, batch_fixed)
            return val_dice_score.item()
    def get_test_point_err(self, batch_moving, batch_fixed, fix_points, mov_points, mask_guide):
        # 首先还是通过图像计算得到变形场deformation_matrix
        with torch.set_grad_enabled(False):
            batch_fixed, batch_moving = batch_fixed.to(
            self.device), batch_moving.to(self.device)
            mask_guide = mask_guide.to(self.device)
            
            if (self.config_train['train']["use_score_map"]):
                registered_image, deformation_matrix, attention_score_matrix = self.voxelmorph(batch_moving, batch_fixed , mask_guide)
                
            else:
                registered_image, deformation_matrix = self.voxelmorph(batch_moving, batch_fixed , mask_guide)
            flow_pre = self.voxelmorph.transformer.grid + deformation_matrix
            mov_points_np = mov_points.to(torch.int)
            mov_points_np = mov_points_np.numpy().squeeze()            
            dst_pred = self.keypoints_process_.points_sample_nearest_train(raw_point=mov_points_np, flow=flow_pre)
            dis_ = (fix_points - dst_pred) ** 2
            dis_ = np.sqrt(dis_[:, 0] + dis_[:, 1])
            avg_dist_ = dis_.mean()
            mae_ = dis_.max()
            
            return avg_dist_, mae_


def main():
    '''
    In this I'll take example of FIRE: Fundus Image Registration Dataset
    to demostrate the working of the API.
    '''
    
    use_gpu = torch.cuda.is_available()
    vm = VoxelMorph(
        (1, config['train']["model_image_width"], config['train']["model_image_width"]), config, is_2d=True, use_gpu=use_gpu)  # Object of the higher level class
    config['train']["batch_size"]
    # params = {'batch_size': 1,
    #           'shuffle': False,
    #           'num_workers': 1,
    #           }
    params = {'batch_size': config['train']["batch_size"],
              'shuffle': False,
              'num_workers': 6,
              }
    
    params_val = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1,
              }

    max_epochs = config['train']["train_epochs"]
    
    random_num = 25
    # Loop over epochs
    
    partition = {}
    # filename = list(set([x.split('_')[0]
    #                 for x in os.listdir(config['train']["train_fire_dir"])]))
    # fire dataset    
    partition['train'] = list(set([x.split('_')[0]
                        for x in os.listdir(config['train']["train_fire_dir"])]))
    
    
    # mixture dataset
    # partition['train'] = list(set([x.split('_')[0] + "_" + x.split('_')[1] + "_" + x.split('_')[2]
    #                     for x in os.listdir(config['train']["train_fire_dir"])]))
    
    
    #partition['train'] = partition['train'][:220]
    # test 1: remove the a and p catagroy datasets
    #partition['train'] = [x for x in partition['train'] if (x[0] != "P" and x[0] != "A")]
    # test 2: remove randomly 50% data
    #partition['train'] = [x for x in partition['train'] if (int(x[2]) % 2 == 0)]

    partition['validation'] = partition['train']

    print(partition['validation'])
    print(len(partition['train']),partition['train'])
        
    # Generators
    if (config['dataset']["dataset_name"] == "fire"):
        training_set = Fire_Dataset_points(partition['train'], configs=config)
        validation_set = Fire_Dataset_points(partition['validation'], configs=config)
    elif (config['dataset']["dataset_name"] == "mixture"):
        training_set = Mixture_Dataset(partition['train'], configs=config)
        validation_set = Mixture_Dataset(partition['validation'], configs=config)
    elif (config['dataset']["dataset_name"] == "fire_seg"):
        training_set = Fire_Dataset_seg(partition['train'], configs=config)
        validation_set = Fire_Dataset_seg(partition['validation'], configs=config)
    elif (config['dataset']["dataset_name"] == "fire_seg_points"):
        training_set = Fire_Dataset_points(partition['train'], configs=config)
        validation_set = Fire_Dataset_points_val(partition['validation'], configs=config)
    # elif (config['dataset']["dataset_name"] == "fire_seg_points"):
    #     training_set = mixture_Dataset_points_select(partition['train'], configs=config)
    #     validation_set = mixture_Dataset_points_select(partition['validation'], configs=config)

    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params_val)   
    # 保存loss\dice score的日志以及模型
    # 检查文件夹是否存在
    save_files_dir = config['train']["save_files_dir"] + config['train']["model_save_prefix"]
    if os.path.exists(save_files_dir) == False:
        # 创建空文件夹
        os.makedirs(save_files_dir)
    log_writer = SummaryWriter(log_dir= save_files_dir + "/logs/" + config['train']["model_save_prefix"])
    # 模型训练
    for epoch in range(max_epochs):
        ## *****every epoch random sample*****
        start_time = time.time()
        train_loss = 0
        train_dice_score = 0
        val_dice_score = 0
        error_avg = 0
        error_mae = 0
        train_sim_loss = 0
        train_smooth_loss = 0
        train_keypoint_loss = 0
        train_atten_loss = 0
        if (config['dataset']["dataset_name"] == "fire_seg_points"):
            for batch_fixed, batch_moving, fix_points, mov_points, mask_guide, score_map in training_generator:       
                loss, dice ,sim_loss, smooth_loss, keypoint_loss, atten_loss= vm.train_model(batch_moving, batch_fixed, fix_points, mov_points, mask_guide, score_map)
                train_dice_score += dice
                train_loss += loss
                train_sim_loss += sim_loss
                train_smooth_loss += smooth_loss
                train_keypoint_loss += keypoint_loss
                train_atten_loss += atten_loss
        else:
            for batch_fixed, batch_moving in training_generator:       
                loss, dice ,sim_loss, smooth_loss= vm.train_model(batch_moving, batch_fixed)
                train_dice_score += dice
                train_loss += loss
                train_sim_loss += sim_loss
                train_smooth_loss += smooth_loss
                train_keypoint_loss += keypoint_loss
        
        vm.epoch += 1
        # 保存模型以及写入日志
        vm.save_model(save_files_dir)
        log_writer.add_scalar("loss/all", train_loss * params['batch_size'] / len(training_set), vm.epoch)
        log_writer.add_scalar("loss/train_sim_loss", train_sim_loss * params['batch_size'] / len(training_set), vm.epoch)
        log_writer.add_scalar("loss/train_smooth_loss", train_smooth_loss * params['batch_size'] / len(training_set), vm.epoch)
        log_writer.add_scalar("loss/train_keypoint_loss", train_keypoint_loss * params['batch_size'] / len(training_set), vm.epoch)
        log_writer.add_scalar("loss/train_atten_loss", train_atten_loss * params['batch_size'] / len(training_set), vm.epoch)
        log_writer.add_scalar("train_dice_score", train_dice_score * params['batch_size'] / len(training_set), vm.epoch)

        print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'After', epoch + 1, 'epochs, the Average training loss is ', train_loss *
              params['batch_size'] / len(training_set), 'train_smooth_loss', train_smooth_loss * params['batch_size'] / len(training_set),
              "keypoints loss is", train_keypoint_loss * params['batch_size'] / len(training_set))
        # Testing time
        start_time = time.time()
        if (config['dataset']["dataset_name"] == "fire_seg_points") and (vm.epoch % 5 == 0):
            for batch_fixed, batch_moving,fixed_point_,moved_point_,mask_guide,_ in validation_generator:
                error_avg_, error_mae_ = vm.get_test_point_err(batch_moving, batch_fixed, fixed_point_, moved_point_, mask_guide)
                error_avg += error_avg_
                error_mae += error_mae_
            print(f" epoch {vm.epoch} avg_error: {error_avg / len(validation_set)}  , max err {error_mae / len(validation_set)}")
            log_writer.add_scalar("val_error_avg", error_avg/  len(validation_set), vm.epoch)
            log_writer.add_scalar("error_mae", error_mae / len(validation_set), vm.epoch)
            if (error_avg/  len(validation_set)) < vm.best_dice_score:
                vm.best_dice_score = error_avg/  len(validation_set)
                print(f'save best model for epoch{vm.epoch}')
                save_path = os.path.join(save_files_dir, "best.pth")
                torch.save(vm.voxelmorph.state_dict(), save_path)

def main_flori():
    '''
    In this I'll take example of FIRE: Fundus Image Registration Dataset
    to demostrate the working of the API.
    '''
    
    use_gpu = torch.cuda.is_available()
    vm = VoxelMorph(
        (1, config['train']["model_image_width"], config['train']["model_image_width"]), config, is_2d=True, use_gpu=use_gpu)  # Object of the higher level class
    
    params = {'batch_size': config['train']["batch_size"],
              'shuffle': False,
              'num_workers': 5,
              }
    
    params_val = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1,
              }

    max_epochs = config['train']["train_epochs"]
    partition = {}
   
    # fire dataset    
    partition['train'] = list(set([x.split('.')[0]
                        for x in os.listdir(config['train']["anno_file_dir"])]))

    partition['validation'] = partition['train']

    print(partition['validation'])
    print(len(partition['train']),partition['train'])
        
    # Generators
    if (config['dataset']["dataset_name"] == "fire"):
        training_set = Fire_Dataset_points(partition['train'], configs=config)
        validation_set = Fire_Dataset_points(partition['validation'], configs=config)
    elif (config['dataset']["dataset_name"] == "mixture"):
        training_set = Mixture_Dataset(partition['train'], configs=config)
        validation_set = Mixture_Dataset(partition['validation'], configs=config)
    elif (config['dataset']["dataset_name"] == "fire_seg"):
        training_set = Fire_Dataset_seg(partition['train'], configs=config)
        validation_set = Fire_Dataset_seg(partition['validation'], configs=config)
    elif (config['dataset']["dataset_name"] == "fire_seg_points"):
        training_set = Fire_Dataset_points(partition['train'], configs=config)
        validation_set = Fire_Dataset_points(partition['validation'], configs=config)
    elif (config['dataset']["dataset_name"] == "flori_points"):
        training_set = flori_points(partition['train'], configs=config)
        validation_set = flori_points(partition['validation'], configs=config)

    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params_val)   
    # 保存loss\dice score的日志以及模型
    # 检查文件夹是否存在
    save_files_dir = config['train']["save_files_dir"] + config['train']["model_save_prefix"]
    if os.path.exists(save_files_dir) == False:
        # 创建空文件夹
        os.makedirs(save_files_dir)
    log_writer = SummaryWriter(log_dir= save_files_dir + "/logs/" + config['train']["model_save_prefix"])
    # 模型训练
    for epoch in range(max_epochs):
        ## *****every epoch random sample*****
        start_time = time.time()
        train_loss = 0
        train_dice_score = 0
        val_dice_score = 0
        error_avg = 0
        error_mae = 0
        train_sim_loss = 0
        train_smooth_loss = 0
        train_keypoint_loss = 0
        train_atten_loss = 0
        if (config['dataset']["dataset_name"] == "fire_seg_points"):
            for batch_fixed, batch_moving, fix_points, mov_points, mask_guide, score_map in training_generator:       
                loss, dice ,sim_loss, smooth_loss, keypoint_loss, atten_loss= vm.train_model(batch_moving, batch_fixed, fix_points, mov_points, mask_guide, score_map)
                train_dice_score += dice
                train_loss += loss
                train_sim_loss += sim_loss
                train_smooth_loss += smooth_loss
                train_keypoint_loss += keypoint_loss
                train_atten_loss += atten_loss
        else:
            for batch_fixed, batch_moving, fix_points, mov_points, mask_guide, score_map in training_generator:       
                loss, dice ,sim_loss, smooth_loss, keypoint_loss, atten_loss= vm.train_model(batch_moving, batch_fixed, fix_points, mov_points, mask_guide, score_map)
                train_dice_score += dice
                train_loss += loss
                train_sim_loss += sim_loss
                train_smooth_loss += smooth_loss
                train_keypoint_loss += keypoint_loss
                train_atten_loss += atten_loss
        
        vm.epoch += 1
        # 保存模型以及写入日志
        vm.save_model(save_files_dir)
        log_writer.add_scalar("loss/all", train_loss * params['batch_size'] / len(training_set), vm.epoch)
        log_writer.add_scalar("loss/train_sim_loss", train_sim_loss * params['batch_size'] / len(training_set), vm.epoch)
        log_writer.add_scalar("loss/train_smooth_loss", train_smooth_loss * params['batch_size'] / len(training_set), vm.epoch)
        log_writer.add_scalar("loss/train_keypoint_loss", train_keypoint_loss * params['batch_size'] / len(training_set), vm.epoch)
        log_writer.add_scalar("loss/train_atten_loss", train_atten_loss * params['batch_size'] / len(training_set), vm.epoch)
        log_writer.add_scalar("train_dice_score", train_dice_score * params['batch_size'] / len(training_set), vm.epoch)

        print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'After', epoch + 1, 'epochs, the Average training loss is ', train_loss *
              params['batch_size'] / len(training_set), 'and average DICE score is', train_dice_score * params['batch_size'] / len(training_set))
        # Testing time
        start_time = time.time()
        if (vm.epoch % 5 == 0):
            for batch_fixed, batch_moving,fixed_point_,moved_point_,mask_guide,_ in validation_generator:
                error_avg_, error_mae_ = vm.get_test_point_err(batch_moving, batch_fixed, fixed_point_, moved_point_, mask_guide)
                error_avg += error_avg_
                error_mae += error_mae_
            print(f" epoch {vm.epoch} avg_error: {error_avg / len(validation_set)}  , max err {error_mae / len(validation_set)}")
            log_writer.add_scalar("val_error_avg", error_avg/  len(validation_set), vm.epoch)
            log_writer.add_scalar("error_mae", error_mae / len(validation_set), vm.epoch)
            if (error_avg/  len(validation_set)) < vm.best_dice_score:
                vm.best_dice_score = error_avg/  len(validation_set)
                print(f'save best model for epoch{vm.epoch}')
                save_path = os.path.join(save_files_dir, "best.pth")
                torch.save(vm.voxelmorph.state_dict(), save_path)

if __name__ == "__main__":
    seed_everything()
    #main_flori()
    main()
