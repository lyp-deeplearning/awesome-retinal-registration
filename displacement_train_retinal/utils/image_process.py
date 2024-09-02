import os
import numpy as np
import PIL
import cv2
import neurite as ne

"""图像处理相关"""
def read_and_resize(img_path, w_size, h_size):
    new_img = cv2.imread(img_path)
    new_img = cv2.resize(new_img, (w_size, h_size))
    return new_img

def tensor_2_images(tensor, sequence=[0,1,2,3]):
    numpy_array = tensor.squeeze().numpy() * 255
    uint8_array = numpy_array.astype(np.uint8)
    uint8_array = np.transpose(uint8_array, (1, 2, 0))
    bgr_array = cv2.cvtColor(uint8_array, cv2.COLOR_RGB2BGR)
    return bgr_array

def vertical_and_concat(image1, image2):
    # 获取两幅图像的高度和宽度
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    # 确定新图像的宽度为较大的那个宽度
    new_width = max(width1, width2)
    # 计算需要在较短的一侧添加的白边的宽度
    width_diff1 = new_width - width1
    width_diff2 = new_width - width2
    # 创建白色的填充区域
    padding1 = np.ones((height1, width_diff1, 3), dtype=np.uint8) * 255  # 白色 (255, 255, 255)
    padding2 = np.ones((height2, width_diff2, 3), dtype=np.uint8) * 255  # 白色 (255, 255, 255)
    # 在两幅图像的左侧或右侧添加白边
    image1_padded = np.hstack((image1, padding1))
    image2_padded = np.hstack((image2, padding2))
    # 使用 vconcat 函数垂直拼接两幅图像
    stacked_image = cv2.vconcat([image1_padded, image2_padded])
    return stacked_image

def draw_and_points(img, points, radius= 3, color= "blue", type=-1):
    if color == "blue":
        color_value = (255, 0 , 0)
    else:
        color_value = (0, 0, 255)
    for number_point in range(points.shape[0]):
        cv2.circle(img, (int(points[number_point][0]), int(points[number_point][1])), radius, color_value, type)
    return img
import copy
def draw_compare_points(img, fix_points, mov_points, mov_pred_points, radius= 3, color= "good", type=-1):
    if color == "blue":
        color_value = (255, 0 , 0)
    else:
        color_value = (0, 0, 255)
    new_img = copy.deepcopy(img)
    for number_point in range(fix_points.shape[0]):
        cv2.circle(new_img, (int(fix_points[number_point][0]), int(fix_points[number_point][1])), radius, (0, 0, 255), type) #red
    for number_point in range(mov_points.shape[0]):
        cv2.circle(new_img, (int(mov_points[number_point][0]), int(mov_points[number_point][1])), radius, (255, 0, 0), type) #red
    if (color == "good"):
        for number_point in range(mov_pred_points.shape[0]):
            cv2.circle(new_img, (int(mov_pred_points[number_point][0]), int(mov_pred_points[number_point][1])), radius, (255, 255, 255), type) #red
    else:
        for number_point in range(mov_pred_points.shape[0]):
            cv2.circle(new_img, (int(mov_pred_points[number_point][0]), int(mov_pred_points[number_point][1])), radius, (0, 255, 0), type) #red
    return new_img

def merge_two_rgb(img1, img2):
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2 ,cv2.COLOR_BGR2GRAY)
    # h = img1.shape[1]
    # merged = np.zeros((h, h, 3), dtype=np.uint8)
    # merged[:, :, 2] = img2
    # merged[:, :, 1] = img1
    merged = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    return merged

def merge_two_rgb_with_chessboard(img1, img2):
    # 棋盘格参数
    rows = 20
    cols = 20
    # 生成棋盘格图像
    chessboard = np.zeros_like(img1)
    square_size = min(img1.shape[0] // (1 * rows), img1.shape[1] // (1 * cols))
    for i in range(0, rows):
        for j in range(0, cols):
            x = j * 1 * square_size
            y = i * 1 * square_size
            if (i + j) % 2 == 0:
                chessboard[y:y + square_size, x:x + square_size] = [255, 255, 255]

    # 叠加棋盘格到配准后的图像
    # 创建 mask
    mask_a = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
    mask_b = cv2.bitwise_not(mask_a)
    # 对图像进行 mask 操作
    result_a = cv2.bitwise_and(img1, img1, mask=mask_a)
    result_b = cv2.bitwise_and(img2, img2, mask=mask_b)
    # 将两个结果图像叠加在一起
    final_result = cv2.add(result_a, result_b)
    #final_result = cv2.add(img1, img2)
    print(final_result.shape)
    return final_result





## add chessboard mask
def add_chessboard_mask(register_img):
    # 棋盘格参数
    rows = 20
    cols = 20

    # 生成棋盘格图像
    chessboard = np.zeros_like(register_img)
    square_size = min(register_img.shape[0] // (1 * rows), register_img.shape[1] // (1 * cols))
    for i in range(0, rows):
        for j in range(0, cols):
            x = j * 1 * square_size
            y = i * 1 * square_size
            if (i + j) % 2 == 0:
                chessboard[y:y + square_size, x:x + square_size] = [255, 255, 255]

    # 叠加棋盘格到配准后的图像
    result = cv2.addWeighted(register_img, 0.9, chessboard, 0.1, 0)
    print(result.shape)
    return result








## 求取图像梯度图像
def get_gradient(img1):
    # 使用Sobel滤波器计算梯度
    sobel_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅度和方向
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    # 转换梯度幅度和方向为8位图像（可选）
    magnitude_scaled = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    direction_scaled = ((gradient_direction + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    magnitude_scaled[magnitude_scaled > 230] = 0
    #cv2.imwrite("./mag.jpg", magnitude_scaled)
    return magnitude_scaled, direction_scaled

import numpy as np

def extract_and_sort_points(keypoints1, keypoints2, mag_map):
    # 获取a数组的大小
    num_points = keypoints1.shape[0]

    # 创建一个空的c数组
    c = np.zeros((num_points, 3))
    d = np.zeros((num_points, 3))
    # 遍历a中的每个点，提取图像b上的值
    for i in range(num_points):
        x1, y1 = keypoints1[i]
        x2, y2 = keypoints2[i]
        int_x1 = int(x1)
        int_y1 = int(y1)
        value = mag_map[int_y1, int_x1]  # 注意坐标顺序 (y, x) 以及在NumPy中的索引顺序

        # 将点的坐标和对应的值放入c数组
        c[i] = [x1, y1, value]
        d[i] = [x2, y2, value]

    # 根据值对c数组进行排序, 从大到小排序
    sorted_indices = np.argsort(c[:, 2])[::-1]
    c = c[sorted_indices]
    d = d[sorted_indices]

    return c, d


def visulize_points(points, image, width):
    print("===debug vis",points.size)
    image = cv2.resize(image, (width, width))
    print(points.shape)
    for point in points[:70,:]:
        x, y = point
        x_int = int(x)
        y_int = int(y)
        # 绘制一个圆点（可以根据需要更改颜色和半径）
        image = cv2.circle(image, (x_int, y_int), 5, (255, 255, 255), -1)
    for point in points[70:,:]:
            x, y = point
            x_int = int(x)
            y_int = int(y)
            # 绘制一个圆点（可以根据需要更改颜色和半径）
            image = cv2.circle(image, (x_int, y_int), 5, (255, 0, 0), -1)
      
    return image


def generate_guide_map(points, img_size):
    mask_guide = np.zeros((img_size, img_size))
    
    for point in points:
        x,y = point
        if(int(x) > img_size or int(y)> img_size):
            continue
        mask_guide[int(y), int(x)] = 1.0
    return mask_guide

def generate_edge(image):
    # 使用Canny边缘检测
    edges = cv2.Canny(image, threshold1=50, threshold2=150)

    # 进行二值化
   # _, binary_image = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    return edges

class image_train_process:
    def __init__(self, data_aug_type) -> None:
        self.data_augment_type = data_aug_type

    def process_train_data(self, img):
        if (self.data_augment_type == 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif (self.data_augment_type == 2):
            img = img[:,:,1]
        elif (self.data_augment_type == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self.dataset_normalized(img)
            img = self.clahe_equalized(img)
            img = self.adjust_gamma(img, 1.2)
        return img
    

    def dataset_normalized(self, imgs):
        imgs_normalized = np.empty(imgs.shape)
        imgs_std = np.std(imgs)
        imgs_mean = np.mean(imgs)
        imgs_normalized = (imgs-imgs_mean)/imgs_std
        imgs_normalized = ((imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized)-np.min(imgs_normalized)))*255
        return imgs_normalized
    
    def adjust_gamma(self, imgs, gamma=1.0):
        
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        new_imgs = np.empty(imgs.shape)
        new_imgs = cv2.LUT(np.array(imgs, dtype = np.uint8), table)
        return new_imgs
    
    def clahe_equalized(self, imgs):     
        #create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imgs_equalized = np.empty(imgs.shape)
        imgs_equalized = clahe.apply(np.array(imgs, dtype = np.uint8))
        return imgs_equalized


from PIL import Image
import matplotlib.pyplot as plt

# 绘制形变场
class draw_deformation_filed:
    def __init__(self) -> None:
        pass

    def grid2contour(self, grid, save_path):
        assert grid.ndim == 3
        x = np.arange(-1, 1, 2/grid.shape[1])
        y = np.arange(-1, 1, 2/grid.shape[0])
        X, Y = np.meshgrid(x, y)
        Z1 = grid[:,:,0] + 2#remove the dashed line
        Z1 = Z1[::-1]#vertical flip
        Z2 = grid[:,:,1] + 2
        
        plt.figure()
        plt.contour(X, Y, Z1, 70, colors='k')
        plt.contour(X, Y, Z2, 70, colors='k')
        plt.xticks(()), plt.yticks(())#remove x, y ticks
        plt.title('deform field')
        plt.savefig(save_path, dpi=600)

    def draw_ne_field(self, deformation_matrix, save_dir):
        # 画形变场的图像
        deformation_matrix = deformation_matrix.permute(0,2,3,1) #顺序得是n,h,w,c
        deformation_matrix = deformation_matrix[...,[1,0]] #通道顺序交换一下，变成x偏移，y偏移
        fig, ax = ne.plot.flow([deformation_matrix.squeeze().to("cpu").numpy()[::10,::10]], width=5)
        fig.savefig(save_dir)
    
    def process_ori_deformation(self, deformation_matrix, regular_grid, save_dir):
        
        grid = regular_grid
        new_locs = grid + deformation_matrix  
        shape = deformation_matrix.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            ## 本身的维度范围是1,2,768,768, 这句话把x和y方向的方向场交换了，导致学出来的方向偏差很大
            new_locs = new_locs[..., [1, 0]]
            # 最后new_locs输出的顺序应该就是x方向的偏移在前,y方向的偏移在后
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        self.grid2contour(new_locs.squeeze().cpu().numpy(), save_dir)
        






class image_read:
    def __init__(self) -> None:
        pass
    def read2gray_np(self, img_path, is_gray = False):
        if (is_gray == False):
            image = Image.open(img_path)
            image = np.asarray(image).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = Image.open(img_path)
            image = np.asarray(image).astype(np.uint8)
        return image
    
    def read2rgb_np(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = np.asarray(image).astype(np.uint8)
        return image
    
    def read2green(self, img_path):
        image = Image.open(img_path)
        image = np.asarray(image).astype(np.uint8)
        image = image[:,:,1]
        return image

    def normalize(self, img):
        img = self.dataset_normalized(img)
        img = self.clahe_equalized(img)
        img = self.adjust_gamma(img, 1.2)
        return img
    
    def dataset_normalized(self, imgs):
        imgs_normalized = np.empty(imgs.shape)
        imgs_std = np.std(imgs)
        imgs_mean = np.mean(imgs)
        imgs_normalized = (imgs-imgs_mean)/imgs_std
        imgs_normalized = ((imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized)-np.min(imgs_normalized)))*255
        return imgs_normalized
    
    def adjust_gamma(self, imgs, gamma=1.0):
        
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        new_imgs = np.empty(imgs.shape)
        new_imgs = cv2.LUT(np.array(imgs, dtype = np.uint8), table)
        return new_imgs
    
    def clahe_equalized(self, imgs):     
        #create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imgs_equalized = np.empty(imgs.shape)
        imgs_equalized = clahe.apply(np.array(imgs, dtype = np.uint8))
        return imgs_equalized


if __name__ == "__main__":
    # # test
    # img_path = "/home/yepeng_liu/code_base/unet_pretrain/displacement_train_retinal/train_datas/mix_generate/wrap_file_super-retinal/mix_0_1_1.jpg"
    # img1= cv2.imread(img_path, 0)
    # mag, dir = get_gradient(img1)
    # print(mag.shape, dir.shape, mag[100:120, 200:220])
    # cv2.imwrite("./mag.jpg", mag)
    # cv2.imwrite("./dir.jpg", dir)

    # test绘制变形场图
    draw_defor_test = draw_deformation_filed()
    img_shape = [40, 80]  
    x = np.arange(-1, 1, 2/img_shape[1])
    y = np.arange(-1, 1, 2/img_shape[0])
    X, Y = np.meshgrid(x, y)
    regular_grid = np.stack((X,Y), axis=2)
    draw_defor_test.grid2contour(regular_grid, "./regular.png")
    rand_field = np.random.rand(*img_shape,2)
    rand_field_norm = rand_field.copy()
    rand_field_norm[:,:,0] = rand_field_norm[:,:,0]*2/img_shape[1] 
    rand_field_norm[:,:,1] = rand_field_norm[:,:,1]*2/img_shape[0] 
    
    sampling_grid = regular_grid + rand_field_norm
    print("===grid shape", sampling_grid.shape)
    draw_defor_test.grid2contour(sampling_grid, "deforma.png")
