import os
import cv2
import numpy as np
import torch
from .image_process import *
"""
show images
"""
# 将三张图像拼接在一幅图像
def show_three_images(img1, img2, img3):
    # 获取图片的高度和宽度
    height, width = img1.shape[:2]
    # 创建一个空白画布，用于拼接图片
    result = np.zeros((height, width * 3, 3), dtype=np.uint8)
    # 拼接图片
    result[:, :width] = img1
    result[:, width:width*2] = img2
    result[:, width*2:] = img3
    # 添加标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    font_thickness = 1
    cv2.putText(result, 'moving', (10, 20), font, font_scale, font_color, font_thickness)
    cv2.putText(result, 'fixed', (width + 10, 20), font, font_scale, font_color, font_thickness)
    cv2.putText(result, 'register', (width * 2 + 10, 20), font, font_scale, font_color, font_thickness)
    return result

# 把仿射变换后的图像融合在一起
def blend_two_images(input_file_dir, save_dir):
    print("[========start blend the fire datasets========]")
    filename = list(set([x.split('_')[0]
                         for x in os.listdir(input_file_dir)]))
    for index in range(len(filename)):
        id = filename[index]
        fixed_path = input_file_dir + id + '_1.jpg'
        moving_path = input_file_dir + id + '_2.jpg'
        # 读取两幅图像
        image1 = cv2.imread(fixed_path)
        image2 = cv2.imread(moving_path)
        # 确保两幅图像的大小相同
        if image1.shape != image2.shape:
            raise ValueError("两幅图像的大小不一致")
        # 设置融合的透明度（alpha值，范围从0到1，0表示完全透明，1表示完全不透明）
        alpha = 0.5
        # 使用 addWeighted 函数融合图像
        blended_image = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
        # 保存融合后的图像
        cv2.imwrite(save_dir + id + ".jpg", blended_image)

# 图像的读取 + resize
def read_resize(img_path, w_size, h_size):
    return read_and_resize(img_path, w_size, h_size)

# pytorch的浮点张量转化成uint8的numpy对象用于图像保存
def tensor2images(tensor, sequence=[0,1,2,3]):
    return tensor_2_images(tensor=tensor, sequence=sequence)
# 垂直方向拼接两幅图像
def vertical_concat(img1, img2):
    return vertical_and_concat(image1=img1, image2=img2)
# 在numpy图像绘制关键点
def draw_points(img, points, radius= 6, color= "blue", type=-1):
    return draw_and_points(img, points, radius, color, type)


"""
计算指标相关
"""
# 计算auc相关的指标
# Compute AUC scores for image registration on the FIRE dataset
def compute_auc(s_error, p_error, a_error):
    assert (len(s_error) == 71)  # Easy pairs
    assert (len(p_error) == 48)  # Hard pairs. Note file control_points_P37_1_2.txt is ignored
    assert (len(a_error) == 14)  # Moderate pairs

    s_error = np.array(s_error)
    p_error = np.array(p_error)
    a_error = np.array(a_error)
    # 2912的尺寸用的是25，现在是768   7
    limit = 25
    gs_error = np.zeros(limit + 1)
    gp_error = np.zeros(limit + 1)
    ga_error = np.zeros(limit + 1)

    accum_s = 0
    accum_p = 0
    accum_a = 0

    for i in range(1, limit + 1):
        gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)
        gp_error[i] = np.sum(p_error < i) * 100 / len(p_error)
        ga_error[i] = np.sum(a_error < i) * 100 / len(a_error)

        accum_s = accum_s + gs_error[i]
        accum_p = accum_p + gp_error[i]
        accum_a = accum_a + ga_error[i]

    auc_s = accum_s / (limit * 100)
    auc_p = accum_p / (limit * 100)
    auc_a = accum_a / (limit * 100)
    mAUC = (auc_s + auc_p + auc_a) / 3.0
    return {'s': auc_s, 'p': auc_p, 'a': auc_a, 'mAUC': mAUC}

def compute_auc_flori(s_error, p_error, a_error):
    

    s_error = np.array(s_error)
    p_error = np.array(p_error)
    a_error = np.array(a_error)

    limit = 100
    gs_error = np.zeros(limit + 1)
    gp_error = np.zeros(limit + 1)
    ga_error = np.zeros(limit + 1)

    accum_s = 0
    accum_p = 0
    accum_a = 0

    for i in range(1, limit + 1):
        gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)
        gp_error[i] = np.sum(p_error < i) * 100 / len(p_error)
        ga_error[i] = np.sum(a_error < i) * 100 / len(a_error)

        accum_s = accum_s + gs_error[i]
        accum_p = accum_p + gp_error[i]
        accum_a = accum_a + ga_error[i]

    auc_s = accum_s / (limit * 100)
    auc_p = accum_p / (limit * 100)
    auc_a = accum_a / (limit * 100)
    mAUC = (auc_s + auc_p + auc_a) / 3.0
    return {'s': auc_s, 'p': auc_p, 'a': auc_a, 'mAUC': mAUC}


class record_metrics():
    def __init__(self) -> None:
        # 需要统计的指标
        self.dice_all = 0.0
        self.image_num = 0
        self.failed = 0
        self.inaccurate = 0
        self.mae = 0
        self.mee = 0
        self.avg_dist = 0
        # category: S, P, A, corresponding to Easy, Hard, Mod in paper
        self.auc_record = dict([(category, []) for category in ['S', 'P', 'A']])
    
    def update_auc(self, category, ave_dis):
        self.auc_record[category].append(ave_dis)

    def update_distance(self, fix_p, mov_p):
        dis = (fix_p - mov_p) ** 2
        dis = np.sqrt(dis[:, 0] + dis[:, 1])
        self.avg_dist = dis.mean()
        self.mae = dis.max()
        self.mee = np.median(dis)
    
    def print_log_info(self, auc_type="flori"):
        # 输出总共的点的误差
        print('-'*40)
        print(f"Failed:{'%.2f' % (100*self.failed/self.image_num)}%, Inaccurate:{'%.2f' % (100*self.inaccurate/self.image_num)}%, "
            f"Acceptable:{'%.2f' % (100*(self.image_num-self.inaccurate-self.failed)/self.image_num)}%")
        print('-'*40)
        auc = compute_auc_flori(self.auc_record['S'], self.auc_record['P'], self.auc_record['A'])
        print('S: %.3f, P: %.3f, A: %.3f, mAUC: %.3f' % (auc['s'], auc['p'], auc['a'], auc['mAUC']))


# 关键点的读取以及处理
class keypoints_process:
    def __init__(self) -> None:
        pass
    
    def keypoint_read(self, gt_txt_file, scale_w = 1.0, scale_h = 1.0):
        points_gd = np.loadtxt(gt_txt_file)
        fix = np.zeros([len(points_gd), 2])
        mov = np.zeros([len(points_gd), 2])
        
        fix[:, 0] = (points_gd[:, 0] * scale_w) #dst对应的是refer 1 fix，raw对应的是query 2 mov
        fix[:, 1] = (points_gd[:, 1] * scale_h)
        mov[:, 0] = (points_gd[:, 2] * scale_w) #对应的格式是10*2, 对应的尺度是原图2912的大小
        mov[:, 1] = (points_gd[:, 3] * scale_h)
        
        mov = mov.astype(int)
        fix = fix.astype(int)
        return mov, fix
    
    def keypoint_read_original(self, gt_txt_file):
        points_gd = np.loadtxt(gt_txt_file)
        fix = np.zeros([len(points_gd), 2])
        mov = np.zeros([len(points_gd), 2])
        fix[:, 0] = points_gd[:, 0] #dst对应的是refer 1 fix，raw对应的是query 2 mov
        fix[:, 1] = points_gd[:, 1]
        mov[:, 0] = points_gd[:, 2] 
        mov[:, 1] = points_gd[:, 3] 


        return mov, fix

    def keypoint_scale(self, config, points1, type= "old"):
        if (type == "old"):
            scale_w = config["PREDICT"]["image_original_size"] / config["PREDICT"]["model_image_width"] 
            scale_h = config["PREDICT"]["image_original_size"] / config["PREDICT"]["model_image_width"]  
        else:
            scale_w = config["PREDICT"]["model_image_width"] / config["PREDICT"]["image_original_size"]
            scale_h = config["PREDICT"]["model_image_width"] / config["PREDICT"]["image_original_size"]
        points1[:, 0] = (points1[:, 0] * scale_w) #dst对应的是refer 1 fix，raw对应的是query 2 mov
        points1[:, 1] = (points1[:, 1] * scale_h)
       
        return points1

    
    def points_sample_bilinear(self, raw_point, flow_image, dst_point):
        dst_pred = np.zeros([raw_point.shape[0], 2])
        for index_p in range(raw_point.shape[0]):
            row_index = (raw_point[index_p, 0] ) #横坐标
            col_index = (raw_point[index_p, 1] ) #纵坐标
            query_value = round(col_index * 0.768 + row_index / 1000, 3)
            print("debug position==========", row_index, col_index, dst_point[index_p,0], dst_point[index_p, 1])
            #import pdb;pdb.set_trace()
            print(query_value)
            indice = torch.where(flow_image[0,0,:,:] == query_value)
            #print(flow_image[0,0,671,:])
            print(indice)
            dst_pred[index_p, 0] = int(indice[1])
            dst_pred[index_p, 1] = int(indice[0])
            #print(dst_pred[index_p, 0])

        return dst_pred
    
    def points_sample_bilinear_new(self, raw_point, flow):
        # 反向计算flow，根据新的图像
        dst_pred = np.zeros([raw_point.shape[0], 2])
        for index_p in range(raw_point.shape[0]):
            row_index = (raw_point[index_p, 0] ) #横坐标
            col_index = (raw_point[index_p, 1] ) #纵坐标
            print("debug position==========", row_index, col_index)
            #print("1 flow value is: ", flow[0,0,row_index,col_index], flow[0,1,row_index,col_index]) #69.8
            #print("2 flow value is: ", flow[0,0,col_index,row_index], flow[0,1,col_index,row_index]) #69.2   #原始的78.4
            dst_pred[index_p, 0] = int(flow[0,1,col_index,row_index])
            dst_pred[index_p, 1] = int(flow[0,0,col_index,row_index]) 
        return dst_pred

    def points_sample_nearest(self, raw_point, flow, size_scale):
        # 正向计算，从grid的附近找最接近的数
        # grid顺序，未做处理是n,c,h,w。方向y偏移方向在前，x偏移方向在后
        dst_pred = np.zeros([raw_point.shape[0], 2])
        for index_p in range(raw_point.shape[0]):
            row_index = (raw_point[index_p, 0]) #横坐标
            col_index = (raw_point[index_p, 1]) #纵坐标
            search_area = 40 # default is 6
            min_distance =10000.0
            for x_search in range(row_index - search_area, row_index + search_area):
                for y_search in range(col_index - search_area, col_index + search_area):
                    # x搜索的范围从原图x坐标附近搜索起
                    if (x_search > size_scale or y_search > size_scale):
                        x_search = size_scale - 1
                        y_search = size_scale - 1
                        distance = np.abs(row_index - int(flow[0,1,y_search,x_search].item())) + np.abs(col_index - int(flow[0,0,y_search,x_search].item()))
                    distance = np.abs(row_index - int(flow[0,1,y_search,x_search].item())) + np.abs(col_index - int(flow[0,0,y_search,x_search].item()))
                   
                    if (distance < min_distance):
                        dst_pred[index_p, 0] = x_search
                        dst_pred[index_p, 1] = y_search
                        min_distance = distance
        return dst_pred
    
    def points_sample_nearest_train(self, raw_point, flow):
        # 正向计算，从grid的附近找最接近的数
        # grid顺序，未做处理是n,c,h,w。方向y偏移方向在前，x偏移方向在后
        dst_pred = np.zeros([raw_point.shape[0], 2])
        for index_p in range(raw_point.shape[0]):
            row_index = (raw_point[index_p, 0]) #横坐标
            col_index = (raw_point[index_p, 1]) #纵坐标
            
            search_area = 25
            min_distance =10000.0
            for x_search in range(row_index - search_area, row_index + search_area):
                for y_search in range(col_index - search_area, col_index + search_area):
                    # x搜索的范围从原图x坐标附近搜索起
                    distance = np.abs(row_index - int(flow[0,1,y_search,x_search].item())) + np.abs(col_index - int(flow[0,0,y_search,x_search].item()))
                   
                    if (distance < min_distance):
                        dst_pred[index_p, 0] = x_search
                        dst_pred[index_p, 1] = y_search
                        min_distance = distance
        return dst_pred



def mask_substract(query_align_first, refer_ori):
    # 生成掩膜
    mask = np.logical_and(query_align_first != 0, refer_ori != 0)
    # 将掩膜应用于两幅图像
    result_a = query_align_first * mask
    result_b = refer_ori * mask
    return result_a, result_b
