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
from skimage.metrics import structural_similarity
import tqdm

def deal_with_floridata(anno_path):
    gt_dir = anno_path
    match_pairs = [x for x in os.listdir(gt_dir) if x.endswith('.txt')
                ]
    match_pairs.sort()
    return match_pairs

def inference_two_imgs(image1, image2, query_p, refer_p):
    # 使用 SIFT 检测特征点和计算描述子
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # 使用 FLANN 匹配器匹配特征点
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 选择良好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 提取匹配点的坐标
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性变换矩阵
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    # 映射图片1到图片2
    result_image = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))

    # 随机选择10个特征点
    selected_keypoints = np.random.choice(len(keypoints1), 10, replace=False)

    # 将图片1上的10个点坐标映射到图片2的坐标系中
    pre_p = cv2.perspectiveTransform(
        query_p.reshape(-1, 1, 2),
        H
    )
    return result_image, pre_p



def generate_val():
    # 1） 构建推理器
   

    # 2） 数据地址，分别读取gt和图片数据
    anno_path = "/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/flori_data/scale_flori/anno/"
    img_path = "/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/flori_data/scale_flori/images/"

    anno_gt_out = "/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/flori_data/sift_infer/align_gt/"
    anno_sp_out = "/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/flori_data/sift_infer/align_sp_txt/"
    img_out = "/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/flori_data/sift_infer/align_img/"

    match_pairs = deal_with_floridata(anno_path)
    print(match_pairs, len(match_pairs))
    
    # 3）循环处理单张数据
    for pair_file in match_pairs:
        gt_file = os.path.join(anno_path, pair_file)
        file_name = pair_file.replace('.txt', '')
        refer = "Montage_Subject_" + file_name.split('_')[-1] + ".jpg"
        query = "Raw_FA_" + file_name.split('_')[3] + "_Subject_" + file_name.split('_')[-1] + ".jpg"

        query_im_path = os.path.join(img_path, query )
        refer_im_path = os.path.join(img_path, refer )
        # 读取的顺序是query对应的是序号2，refer对应的顺序是1
        print("=====1 query image", query_im_path, refer_im_path)
      
        # 这里返回的query_image要是原始的尺寸2912
        query_ori = cv2.imread(query_im_path)
        refer_ori = cv2.imread(refer_im_path)
        
       
        # gt file read
        points_gd = np.loadtxt(gt_file)
        raw = np.zeros([len(points_gd), 2])
        dst = np.zeros([len(points_gd), 2])
        # gt points
        dst[:, 0] = points_gd[:, 0] #dst对应的是refer 1 fix，raw对应的是query 2 mov
        dst[:, 1] = points_gd[:, 1]
        raw[:, 0] = points_gd[:, 2]
        raw[:, 1] = points_gd[:, 3]

        result_image, pre_p = inference_two_imgs(query_ori, refer_ori, raw, dst)
        pre_p = np.squeeze(pre_p)
        label_gt_out = os.path.join(anno_gt_out, pair_file)

        print(dst.shape, pre_p.shape)
        with open(label_gt_out, "w") as label_file:
            for index in range(pre_p.shape[0]):
                label_file.write(f"{dst[index][0]:.3f} {dst[index][1]:.3f} {pre_p[index][0]:.3f} {pre_p[index][1]:.3f}\n")
       
       
        # 5) 保存点和图像
        # 保存align后的图像    
        cv2.imwrite(img_out + query, result_image)
        cv2.imwrite(img_out + refer, refer_ori)


if __name__ == "__main__":
    generate_val()

