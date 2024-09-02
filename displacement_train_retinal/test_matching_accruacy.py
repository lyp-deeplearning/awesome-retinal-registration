from models import  voxelmorph2d as vm2d
from models import  voxelmorph3d as vm3d
from train import VoxelMorph
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
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
import yaml
import cv2
from PIL import Image
from torchvision import transforms
from models.voxelmorph2d import dice_score
from models.voxelmorph2d import vxm_SpatialTransformer
from utils.post_process import *
import shutil
import neurite as ne

config_path = "./config/test_config/test.yaml"
if os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError("config file doesn't exist")

check_model_params = False

class vm_predictor:
    def __init__(self, config):
        self.config = config
        self.predict_config = config["PREDICT"]
        self.device = self.predict_config["device"]
        self.device = torch.device(self.device if torch.cuda.is_available else "cpu")
        self.save_path = self.predict_config["model_save_path"]
        
        # load model
        use_gpu = torch.cuda.is_available()
        save_path = "./save"
        channel_num = 1 if (config["PREDICT"]["is_gray"]) else 3
        print(channel_num) 
        self.vm = VoxelMorph(
            (channel_num, self.predict_config["model_image_width"], self.predict_config["model_image_height"]), config, is_2d=True, use_gpu=use_gpu)  # Object of the higher level class
        
        checkpoint = torch.load(self.save_path, map_location= self.device)
        self.vm.voxelmorph.load_state_dict(checkpoint)
        shape = (self.predict_config["model_image_width"],self.predict_config["model_image_width"])
        self.spatial_tran = vxm_SpatialTransformer(shape)

        if check_model_params == True:
            for param_tensor in self.vm.voxelmorph.state_dict():
                #打印 key value字典
                print(param_tensor,'\t',self.vm.voxelmorph.state_dict()[param_tensor].size())
            print(self.vm.voxelmorph.state_dict()["unet.final_layer.3.weight"])
        
        # 打印匹配到的网络的key的对数
        # 打印模型 a 和模型 b 的状态字典键
        keys_a = set(self.vm.voxelmorph.state_dict().keys())
        keys_b = set(checkpoint.keys())

        # 查找匹配的键和未匹配的键
        matching_keys = keys_a.intersection(keys_b)
        unmatched_keys_a = keys_a.difference(keys_b)
        unmatched_keys_b = keys_b.difference(keys_a)

        # 打印结果
        print(f'匹配的键数量：{len(matching_keys)}')
        print(f'未匹配的键数量（模型 a 中的未匹配键）：{len(unmatched_keys_a)}')
        print(f'未匹配的键数量（模型 b 中的未匹配键）：{len(unmatched_keys_b)}')
       
        self.vm.voxelmorph.to(self.device)
        self.spatial_tran.to(self.device)
        self.vm.voxelmorph.eval()
        
    def inference_on_images(self, fixed_image, moving_image, mask_guide):
        # 3 inference and get the result
        with torch.set_grad_enabled(False):
            batch_fixed, batch_moving = fixed_image.to(
                self.device), moving_image.to(self.device)
            #模型推理，返回配准后的图像和变形场
            #guide_map= torch.tensor(0)
            mask_guide = mask_guide.to(self.device).unsqueeze(0)
           
            if (self.config['train']["use_score_map"]):
                registered_image, deformation_matrix, attention_score_matrix = self.vm.voxelmorph(batch_moving, batch_fixed , mask_guide)
            else:
                registered_image, deformation_matrix =self.vm.voxelmorph(batch_moving, batch_fixed , mask_guide)
        return registered_image, deformation_matrix
    
    def spatial_trans(self, moving_image, deformation_matrix):
        moving_image, deformation_matrix = moving_image.to(
                self.device), deformation_matrix.to(self.device)
        registered_image = self.spatial_tran(moving_image, deformation_matrix)
        return registered_image


def test_two_images(img1_dir, img2_dir, ori_dir, txt_dir, vm_predictor_, is_gray = False):
    '''
    In this I'll take example of FIRE: Fundus Image Registration Dataset
    to demostrate the working of the API.
    '''
    im_read_engine = image_read()
    # 1、 load image
    transforms_ = transforms.Compose([
            transforms.Resize((config["PREDICT"]["model_image_width"], config["PREDICT"]["model_image_width"])),
            transforms.ToTensor(),
        ])
    # if (is_gray):
    #     # 如果图像本来就是灰度图，则不需要读入rgb再转换成灰度图
    #     if (config["PREDICT"]["image_single_channel"] == False):
    #         fixed_image = im_read_engine.read2gray_np(img1_dir, is_gray= False)
    #         moving_image = im_read_engine.read2gray_np(img2_dir, is_gray= False)
    #     else:
    #         fixed_image = im_read_engine.read2gray_np(img1_dir, is_gray= True)
    #         moving_image = im_read_engine.read2gray_np(img2_dir, is_gray= True)
    # else:
    #     fixed_image = im_read_engine.read2rgb_np(img1_dir)
    #     moving_image = im_read_engine.read2rgb_np(img2_dir)

    if (config["PREDICT"]["data_aug_type"] == 2):
        fixed_image = im_read_engine.read2green(img1_dir)
        moving_image = im_read_engine.read2green(img2_dir)
    elif (config["PREDICT"]["data_aug_type"] == 3):
        fixed_image = im_read_engine.read2gray_np(img1_dir, is_gray= False)
        moving_image = im_read_engine.read2gray_np(img2_dir, is_gray= False)
        fixed_image = im_read_engine.normalize(fixed_image)
        moving_image = im_read_engine.normalize(moving_image)

  
    ## using single channel
    fixed_image = transforms_(Image.fromarray(fixed_image))
    moving_image = transforms_(Image.fromarray(moving_image))
    fixed_image.unsqueeze_(0)
    moving_image.unsqueeze_(0)

    # # 2、使用 torch.sum() 统计等于零的元素数量
    # # 3、测试dice的相似度得分
    # if (is_gray == False):
    #     dice_test_score = dice_score(fixed_image * 255, moving_image * 255)
    # else:
    #     fixed_image_ori = Image.open(img1_dir).convert("RGB")
    #     moving_image_ori = Image.open(ori_dir).convert("RGB")
    #     fixed_image_ori = np.asarray(fixed_image_ori).astype(np.uint8)
    #     moving_image_ori = np.asarray(moving_image_ori).astype(np.uint8)
    #     ## using single channel
    #     fixed_image_ori = Image.fromarray(fixed_image_ori)
    #     fixed_image_ori = transforms_(fixed_image_ori)
    #     moving_image_ori = Image.fromarray(moving_image_ori)
    #     moving_image_ori = transforms_(moving_image_ori)
    #     fixed_image_ori.unsqueeze_(0)
    #     moving_image_ori.unsqueeze_(0)
    #     dice_test_score = dice_score(fixed_image_ori, moving_image_ori)
   
   
    # 4、inference and get the results
    keypoints_txt = "./test_datas/super_gt_generate_fire/sp_align_points/" + txt_dir
   
    print("======", keypoints_txt)
    fixed_point_ = []
    with open(keypoints_txt, "r") as anno_file:
            for line_txt in anno_file:
                point_data = line_txt.strip().split()
                fixed_point_.append([float(point_data[0]) * config["PREDICT"]["model_image_width"] /2912, float(point_data[1]) * config["PREDICT"]["model_image_width"]/2912 ])
                
    fixed_point_ = np.array(fixed_point_)
   
    mask_guide = generate_guide_map(fixed_point_ , config["PREDICT"]["model_image_width"])
    mask_guide = torch.tensor(mask_guide).to(torch.float32)

    registered_image, deformation_matrix = vm_predictor_.inference_on_images(fixed_image, moving_image, mask_guide)
    # 5、可视化最终效果
    # 使用 torch.eq() 比较张量中的元素是否等于零
    zero_mask = torch.eq(registered_image, 0)
    count_zeros = torch.sum(zero_mask).item()
    
    # # 将 PyTorch 张量转换为 NumPy 数组
    # if (is_gray == False):
    #     registered_image = registered_image.to("cpu") 
    #     dice_test_score2 = dice_score(fixed_image , registered_image )
    #     # 把配准后的tensor转成图片进行保存
    #     bgr_array = tensor2images(registered_image)
    # else:
    #     registered_image = vm_predictor_.spatial_trans(moving_image_ori, deformation_matrix)
    #     registered_image = registered_image.to("cpu") 
    #     dice_test_score2 = dice_score(fixed_image_ori , registered_image )
    #     # 把配准后的tensor转成图片进行保存
    #     bgr_array = tensor2images(registered_image)
    
    #print(f"[2、registered image shape:{registered_image.shape}, zero count:{count_zeros}, dice test:{dice_test_score2}]")  
    
    
    # 保存图像为文件
    # return deformation_matrix, dice_test_score.item(), dice_test_score2.item(), bgr_array
    return deformation_matrix
   

def test_fire_imgs(file_dir, save_dir, gt_anno_dir, vm_predictor_, is_gray =False):
    print("[========start process the fire datasets========]")
    """
    读取顺序还是得和super-retinal保持一致，否则前后图片读取顺序不一样，算出来的指标有差距
    """
    # 1、根据gt的txt名字来读取图片
    match_pairs = [x for x in os.listdir(gt_anno_dir) if x.endswith('.txt')
               and not x.endswith('P37_1_2.txt')]

    match_pairs.sort()
    txt_name = save_dir + "./final_results/dice_result.txt"
    dice_score_txt = open(txt_name, "w")
    metrics_predict = record_metrics()
    metrics_baseline = record_metrics()
    plot_deformation_engine = draw_deformation_filed()
    
    # 2 处理图片完成dice分数计算已经变形场图像绘制
    for index in range(len(match_pairs)):
        # 读取moving image和fix image
        metrics_predict.image_num += 1
        metrics_baseline.image_num += 1
        file_name = match_pairs[index].replace('.txt', '')
        category = file_name.split('_')[2][0]
        # 第一张是原来的mov预测的图像，对应点坐标的第二个；第二张图像是原来的fix图像，对应点的坐标的第一个
        refer = file_name.split('_')[2] + '_' + file_name.split('_')[3]
        query = file_name.split('_')[2] + '_' + file_name.split('_')[4]
        moving_path = file_dir + query + '.jpg' # 对应的是query 2
        fixed_path = file_dir + refer + '.jpg' # 对应的是原图的refer 1
        ori_dir = config["PREDICT"]["test_ori_dir"] + query + ".jpg"
        #deformation_matrix, dice_test_score, dice_test_score2, register_img= test_two_images(fixed_path, moving_path, ori_dir, match_pairs[index], vm_predictor_, is_gray)
        
        deformation_matrix= test_two_images(fixed_path, moving_path, ori_dir, match_pairs[index], vm_predictor_, is_gray)
        
        # 计算出来的变形场是n,c,h,w。其中第一个通道对应的是y轴的偏移，第二个通道对应的x轴的偏移，在spatial transform里面会做交换
        # 2.2，计算关键点误差        
        # 2.2.1 关键点读取
        gt_file = os.path.join(gt_anno_dir, match_pairs[index])
        keypoints_process_ = keypoints_process()
        # raw 2代表mov， dst 1代表fix
        mov_p, fix_p = keypoints_process_.keypoint_read(gt_txt_file=gt_file, scale_w=config["PREDICT"]["model_image_width"] / config["PREDICT"]["image_original_size"],
                                        scale_h=config["PREDICT"]["model_image_width"] / config["PREDICT"]["image_original_size"])
        mov_p_ori, fix_p_ori = keypoints_process_.keypoint_read_original(gt_txt_file=gt_file)
        flow_pre = vm_predictor_.spatial_tran.grid + deformation_matrix
        dst_pred = keypoints_process_.points_sample_nearest(raw_point=mov_p, flow=flow_pre, size_scale=config["PREDICT"]["model_image_width"])

        # 2.2.4 关键点的误差计算
        # update predict info
        fix_p = keypoints_process_.keypoint_scale(config, fix_p, "old")
        mov_p = keypoints_process_.keypoint_scale(config, mov_p, "old")
        dst_pred = keypoints_process_.keypoint_scale(config, dst_pred, "old")

        metrics_predict.update_distance(fix_p=fix_p, mov_p=dst_pred)
        # 50,20
        if metrics_predict.mae > 50 or metrics_predict.mee > 20:
            metrics_predict.inaccurate += 1
        print("[info] failed:[%d],image_num:[%d],inaccurate:[%d], mae:[%f], mee:[%f]",metrics_predict.failed, metrics_predict.image_num, metrics_predict.inaccurate, metrics_predict.mae, metrics_predict.mee)
        metrics_predict.update_auc(category, metrics_predict.avg_dist)
        # predict baseline info
        metrics_baseline.update_distance(fix_p=fix_p_ori, mov_p=mov_p_ori)
        if metrics_baseline.mae > 50 or metrics_baseline.mee > 20:
            metrics_baseline.inaccurate += 1
        metrics_baseline.update_auc(category, metrics_baseline.avg_dist)
       
        # 画形变场的图像
        de_save_path = save_dir + file_name.split('_')[2] + ".png"
        #plot_deformation_engine.draw_ne_field(deformation_matrix, de_save_path)
        plot_deformation_engine.process_ori_deformation(deformation_matrix, vm_predictor_.spatial_tran.grid, de_save_path)
       
        print(f"[== id {file_name.split('_')[2]} is processed ok, the rest of {len(match_pairs)-index} images!")
        
        ## 拼接原图以及配准后的图像做可视化
        moving_path_ori = config["PREDICT"]["test_ori_dir"] + query + '.jpg' # 对应的是query 2
        fixed_path_ori = config["PREDICT"]["test_ori_dir"] + refer + '.jpg' # 对应的是原图的refer 1
        
        # mov_img = read_resize(moving_path_ori, config["PREDICT"]["model_image_width"], config["PREDICT"]["model_image_height"])
        # fix_img = read_resize(fixed_path_ori, config["PREDICT"]["model_image_width"], config["PREDICT"]["model_image_height"])
        # 把关键点画到图像上
        # 如果我们比super-retinal好的则画红色点
        if (metrics_predict.mae < metrics_baseline.mae and metrics_predict.mee < metrics_baseline.mee):
            color_type = "good"
        else:
            color_type = "bad"

        fix_p = keypoints_process_.keypoint_scale(config, fix_p, "new")
        mov_p = keypoints_process_.keypoint_scale(config, mov_p, "new")
        dst_pred = keypoints_process_.keypoint_scale(config, dst_pred, "new")

        # mov_img = draw_points(img=mov_img, points=mov_p , color=color_type)
        # fix_img = draw_points(img=fix_img, points=fix_p, color=color_type)
        # register_img = draw_points(img=register_img, points=dst_pred, color=color_type)
        # final_result = show_three_images(img1=mov_img, img2=fix_img, img3=register_img)
        # cv2.imwrite(save_dir + file_name.split('_')[2] + ".jpg", final_result)

        # img_with_pt = draw_compare_points(img=fix_img, fix_points=fix_p, mov_points=mov_p, mov_pred_points=dst_pred, color=color_type)
        # merger_base = merge_two_rgb_with_chessboard(img1=fix_img, img2=mov_img)
        # merger_predict = merge_two_rgb_with_chessboard(img1=fix_img, img2=register_img)
        # #final_result = show_three_images(img1=img_with_pt, img2=merger_base, img3=merger_predict)
        # final_result = show_two_images(merger_base, merger_predict)
        # cv2.imwrite(save_dir + file_name.split('_')[2] + ".jpg", final_result)
        # cv2.imwrite(save_dir + file_name.split('_')[2] + "merge_superetina.jpg", merger_base)
        # merge_base_new = add_chessboard_mask(merger_base)
        # cv2.imwrite(save_dir + file_name.split('_')[2] + "merge_superetina2.jpg", merge_base_new)
        
        # cv2.imwrite(save_dir + file_name.split('_')[2] + "merge_focusretina.jpg", merger_predict)
        # merge_predict_new = add_chessboard_mask(merger_predict)
        # cv2.imwrite(save_dir + file_name.split('_')[2] + "merge_focusretina2.jpg", merge_predict_new)
        
        # 记录dice 的分数提升信息
        # metrics_predict.dice_all += (dice_test_score2 - dice_test_score)
        # txt_line = str(file_name.split('_')[2]) + "   dice score :  " + str(dice_test_score) + "  ,  " + str(dice_test_score2) + " ,key err decrease ," + str(metrics_baseline.mee - metrics_predict.mee) + " " + str(metrics_baseline.mae - metrics_predict.mae) + "\n"
        # dice_score_txt.writelines(txt_line)
        txt_line2= "[info] mae: " + str(metrics_predict.mae) + " mee: " + str(metrics_predict.mee) + "\n"
        dice_score_txt.writelines(txt_line2)

    
    # 打印所有的auc指标结果
    print("==============[predict auc]============")
    metrics_predict.print_log_info()
    print("==============[baseline auc]============")
    metrics_baseline.print_log_info()
    base_all_mre = sum(metrics_baseline.auc_record["S"]) + sum(metrics_baseline.auc_record["P"]) + sum(metrics_baseline.auc_record["A"])
    pre_all_mre = sum(metrics_predict.auc_record["S"]) + sum(metrics_predict.auc_record["P"]) + sum(metrics_predict.auc_record["A"])
    mre_num = len(metrics_predict.auc_record["S"]) + len(metrics_predict.auc_record["P"]) + len(metrics_predict.auc_record["A"])
    
    
    print("base rmse", metrics_baseline.rmse / mre_num)
    print("preict rmse", metrics_predict.rmse / mre_num)
    
    # 2 保存dice 计算的文件
    txt_line = "all images calculate result is : " +  str(metrics_predict.dice_all) + " improved!"
    dice_score_txt.writelines(txt_line)
    dice_score_txt.close()   
    # 3 拼接变形场图像和输入的待配准图像
    for index in range(len(match_pairs)):
        file_name = match_pairs[index].replace('.txt', '')
        id = file_name.split('_')[2]
        fixed_path = save_dir + id + '.jpg'
        moving_path = save_dir + id + '.png'
        # 打开第一幅图像
        image1 = cv2.imread(fixed_path)
        #image2 = cv2.imread(moving_path)
        #stacked_image = vertical_concat(image1, image2)
        # 保存拼接后的图像
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_color = (255, 0, 255)
        font_thickness = 2
        cv2.putText(image1, str(id), (int(1456), 60), font, font_scale, font_color, font_thickness)
        new_file = save_dir + "./final_results/" + id + ".jpg"
        cv2.imwrite(new_file, image1)


def demo_two_images(img1_dir, img2_dir, vm_predictor_, is_gray = False):
    '''
    In this I'll take example of FIRE: Fundus Image Registration Dataset
    to demostrate the working of the API.
    '''
    im_read_engine = image_read()
    # 1、 load image
    transforms_ = transforms.Compose([
            transforms.Resize((config["PREDICT"]["model_image_width"], config["PREDICT"]["model_image_width"])),
            transforms.ToTensor(),
        ])

    if (config["PREDICT"]["data_aug_type"] == 2):
        fixed_image = im_read_engine.read2green(img1_dir)
        moving_image = im_read_engine.read2green(img2_dir)
    elif (config["PREDICT"]["data_aug_type"] == 3):
        fixed_image = im_read_engine.read2gray_np(img1_dir, is_gray= False)
        moving_image = im_read_engine.read2gray_np(img2_dir, is_gray= False)
        fixed_image = im_read_engine.normalize(fixed_image)
        moving_image = im_read_engine.normalize(moving_image)

  
    ## using single channel
    fixed_image = transforms_(Image.fromarray(fixed_image))
    moving_image = transforms_(Image.fromarray(moving_image))
    fixed_image.unsqueeze_(0)
    moving_image.unsqueeze_(0)

    # 2、使用 torch.sum() 统计等于零的元素数量
    # 3、测试dice的相似度得分
    if (is_gray == False):
        dice_test_score = dice_score(fixed_image * 255, moving_image * 255)
    else:
        fixed_image_ori = Image.open(img1_dir).convert("RGB")
        moving_image_ori = Image.open(img2_dir).convert("RGB")
        fixed_image_ori = np.asarray(fixed_image_ori).astype(np.uint8)
        moving_image_ori = np.asarray(moving_image_ori).astype(np.uint8)
        ## using single channel
        fixed_image_ori = Image.fromarray(fixed_image_ori)
        fixed_image_ori = transforms_(fixed_image_ori)
        moving_image_ori = Image.fromarray(moving_image_ori)
        moving_image_ori = transforms_(moving_image_ori)
        fixed_image_ori.unsqueeze_(0)
        moving_image_ori.unsqueeze_(0)
        dice_test_score = dice_score(fixed_image_ori, moving_image_ori)
   
   
    # 4、inference and get the results
    keypoints_txt = "/home/yepeng_liu/code_python/keypoints_benchmark/unofficial-voxelmorph/VoxelMorph-PyTorch-master/train_datas/super_gt_generate_fire/sp_align_points/control_points_A02_1_2.txt"
    print("======", keypoints_txt)
    fixed_point_ = []
    with open(keypoints_txt, "r") as anno_file:
            for line_txt in anno_file:
                point_data = line_txt.strip().split()
                fixed_point_.append([float(point_data[0]) * config["PREDICT"]["model_image_width"] /2912, float(point_data[1]) * config["PREDICT"]["model_image_width"]/2912 ])
                
    fixed_point_ = np.array(fixed_point_)
   
    mask_guide = generate_guide_map(fixed_point_ , config["PREDICT"]["model_image_width"])
    mask_guide = torch.tensor(mask_guide).to(torch.float32)

    registered_image, deformation_matrix = vm_predictor_.inference_on_images(fixed_image, moving_image, mask_guide)
    # 5、可视化最终效果
    # 使用 torch.eq() 比较张量中的元素是否等于零
    zero_mask = torch.eq(registered_image, 0)
    count_zeros = torch.sum(zero_mask).item()
    
    # 将 PyTorch 张量转换为 NumPy 数组
    if (is_gray == False):
        registered_image = registered_image.to("cpu") 
        dice_test_score2 = dice_score(fixed_image , registered_image )
        # 把配准后的tensor转成图片进行保存
        bgr_array = tensor2images(registered_image)
    else:
        registered_image = vm_predictor_.spatial_trans(moving_image_ori, deformation_matrix)
        registered_image = registered_image.to("cpu") 
        dice_test_score2 = dice_score(fixed_image_ori , registered_image )
        # 把配准后的tensor转成图片进行保存
        bgr_array = tensor2images(registered_image)
    
    print(f"[2、registered image shape:{registered_image.shape}, zero count:{count_zeros}, dice test:{dice_test_score2}]")  
    
    
    # 保存图像为文件
    return deformation_matrix, dice_test_score.item(), dice_test_score2.item(), bgr_array






if __name__ == "__main__":
    vm_predict = vm_predictor(config=config)
    
    img1_dir = "/home/yepeng_liu/code_python/keypoints_benchmark/unofficial-voxelmorph/VoxelMorph-PyTorch-master/fire-fundus-image-registration-dataset/S58_1.jpg"
    img2_dir = "/home/yepeng_liu/code_python/keypoints_benchmark/unofficial-voxelmorph/VoxelMorph-PyTorch-master/fire-fundus-image-registration-dataset/S58_2.jpg"
    if (config["PREDICT"]["test_method"] == 1):
        # 1 测试两张图片的匹配效果
        # final_result = test_two_images(img1_dir, img2_dir, vm_predictor=vm_predict)
        # cv2.imwrite("./visulize_files/11.jpg", final_result)

     
        moving_path = "/home/yepeng_liu/code_python/nature_imgs/cloud_3-rgb_sp.jpg" # 对应的是query 2
        fixed_path = "/home/yepeng_liu/code_python/nature_imgs/cloud_2-rgb_sp.jpg" # 对应的是原图的refer 1

        moving_path = "/home/yepeng_liu/code_python/keypoints_benchmark/unofficial-voxelmorph/VoxelMorph-PyTorch-master/train_datas/super_gt_generate_fire/gt_align_jpg/A09_1.jpg" # 对应的是query 2
        fixed_path = "/home/yepeng_liu/code_python/keypoints_benchmark/unofficial-voxelmorph/VoxelMorph-PyTorch-master/train_datas/super_gt_generate_fire/gt_align_jpg/A09_2.jpg" # 对应的是原图的refer 1

        deformation_matrix, dice_test_score, dice_test_score2, register_img= demo_two_images(fixed_path, moving_path, vm_predict, config["PREDICT"]["is_gray"])
        
        import flow_vis
        deformation_matrix_show = deformation_matrix.permute(2,3,1,0).squeeze().cpu().numpy()
        de_max = deformation_matrix_show.max()
        de_min = deformation_matrix_show.min()
        deformation_matrix_show = (deformation_matrix_show-de_min) * 255 / (de_max - de_min)
        print("defor",deformation_matrix_show.shape, deformation_matrix_show.max())
        # one_matrix = np.ones((1456,1456,2)) * 255
        # deformation_matrix_show = one_matrix - deformation_matrix_show
        flow_color = flow_vis.flow_to_color(deformation_matrix_show, convert_to_bgr=False
                                            )
        
        import nibabel as nib
        # 创建一个单位仿射矩阵
        affine = np.eye(4)
        flow_image = nib.Nifti1Image(flow_color, affine)
        nib.save(flow_image, "./flow_color.nii") # fn: output filename

        # print(flow_color.max())
        # # 定义纯白色的阈值
        # # 注意：OpenCV使用的是BGR颜色顺序
        # lower_white = np.array([250, 250, 250])
        # upper_white = np.array([255, 255, 255])
        # # 找到图片中所有纯白色的像素点
        # white_mask = cv2.inRange(flow_color, lower_white, upper_white)
        # # 定义棕色的RGB值（在OpenCV中为BGR）
        # brown = [210, 220, 150]
        # flow_color[white_mask == 255] = brown
        # cv2.imwrite("./color_flow.jpg", flow_color)
        ddd
        print(register_img.shape)
        fix_img = cv2.imread(fixed_path)
        fix_img = cv2.resize(fix_img, (1456,1456))
        mov_img = cv2.imread(moving_path)
        mov_img = cv2.resize(mov_img, (1456,1456))
        
        merger_base = merge_two_rgb_with_chessboard(img1=fix_img, img2=mov_img)
        merger_predict = merge_two_rgb_with_chessboard(img1=fix_img, img2=register_img)

        cv2.imwrite("/home/yepeng_liu/code_python/nature_imgs/merger_predict.jpg", merger_predict)
        cv2.imwrite("/home/yepeng_liu/code_python/nature_imgs/merger_base.jpg", merger_base)
    elif (config["PREDICT"]["test_method"] == 2):  
        # 2 测试fire所有的图片对
        file_dir = config["PREDICT"]["test_fire_dir"]     
        save_files_dir = config["PREDICT"]["save_files_dir"]
        model_name = config["PREDICT"]["model_save_path"].split("/")[-1]
        save_files_dir = save_files_dir + "/" + model_name + "/"
        is_gray = config["PREDICT"]["is_gray"]
        # 检查文件夹是否存在
        if os.path.exists(save_files_dir):
            # 如果存在，清空文件夹内容
            shutil.rmtree(save_files_dir)
        # 创建空文件夹
        os.makedirs(save_files_dir)
        os.makedirs(save_files_dir + "final_results")
        gt_anno_dir = config["PREDICT"]["test_gt_dir"]
        test_fire_imgs(file_dir=file_dir, save_dir=save_files_dir, gt_anno_dir = gt_anno_dir, vm_predictor_=vm_predict, is_gray=is_gray)
    elif (config["PREDICT"]["test_method"] == 3):
        # 3 融合变换完后的两张图像
        file_dir = config["PREDICT"]["test_fire_dir"]     
        save_files_dir = config["PREDICT"]["save_files_dir"] + "blend/"
         # 检查文件夹是否存在
        if os.path.exists(save_files_dir):
            # 如果存在，清空文件夹内容
            shutil.rmtree(save_files_dir)
        # 创建空文件夹
        os.makedirs(save_files_dir)
        blend_two_images(file_dir, save_files_dir)
