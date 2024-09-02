import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import math
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# use_gpu = torch.cuda.is_available()

def final_block(in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU()
                )
        return block


class attention_head(nn.Module):
    def __init__(self, in_channels, out_channels, use_gpu=False, device = "cpu"):
        super(attention_head, self).__init__()
        self.down_sample = nn.MaxPool2d(kernel_size=8)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if use_gpu:
            self.down_sample  = self.down_sample.to(device)
            self.conv3x3 = self.conv3x3.to(device)
            
    def forward(self, decode_feature):
        # 拆分特征图a为b和c
        batch_size = decode_feature.shape[0]
        decode_feature2 = self.conv3x3(decode_feature)
        decode_fix, decode_mov = decode_feature2.chunk(2, dim=1)  # 将特征图a在第二维度上分成两半
        fix_feature = self.down_sample(decode_fix).permute(0,2,3,1)
        mov_feature = self.down_sample(decode_mov).permute(0,2,3,1)

        # 压缩b2和c2为b3和c3
        
        fix_feature = fix_feature.view(batch_size, -1, 15)  # 将b2展平为8*(96*96)*15
        mov_feature = mov_feature.view(batch_size, -1, 15)  # 将c2展平为8*(96*96)*15
        
        attention_score_matrix = torch.bmm(fix_feature, mov_feature.permute(0,2,1))
       # attention_score_matrix = torch.sigmoid(attention_score_matrix)
        
        return attention_score_matrix
class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        
        # 使用卷积层来计算注意力权重
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        # 添加3x3卷积层用于保持维度不变
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, feature_map_b, guide_map):
        # 使用softmax函数归一化权重
        guide_map = guide_map.unsqueeze(1)
        attention_weights = self.softmax(guide_map)
        # 将注意力权重应用于特征层b
        attention_output = feature_map_b * attention_weights
        feature_map_c= feature_map_b + attention_output
        feature_map_c = self.conv3x3(feature_map_c)
      
        return feature_map_b


class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU()
                )
        return block
    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
    
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                torch.nn.BatchNorm2d(mid_channel * 2),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
       
        return decode_block1

class UNet_big(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU()
                )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet_big, self).__init__()
        #Encode
        self.conv_encode0 = self.contracting_block(in_channels=in_channel, out_channels=16)
        self.conv_maxpool0 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv_encode1 = self.contracting_block(in_channels=16, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
                                torch.nn.BatchNorm2d(mid_channel * 2),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel*2, out_channels=mid_channel, padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.BatchNorm2d(mid_channel),
                                torch.nn.ReLU(),
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.conv_decode1 = self.expansive_block(64, 32, 16)
        

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block0 = self.conv_encode0(x)
        encode_pool0 = self.conv_maxpool1(encode_block0)
        encode_block1 = self.conv_encode1(encode_pool0)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        cat_layer0 = self.conv_decode1(decode_block1)
        decode_block0 = self.crop_and_concat(cat_layer0, encode_block0)
        return  decode_block0

## voxelmorph版本的spatial transform操作
class vxm_SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, device = "cpu", mode='bilinear'):
        super().__init__()

        self.mode = mode
        # create sampling grid
        self.device = device
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).to(self.device)

        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        #grid = self.grid.repeat(flow.shape[0],1,1,1)
        grid = self.grid
        new_locs = grid + flow  
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            ## 本身的维度范围是1,2,768,768, 这句话把x和y方向的方向场交换了，导致学出来的方向偏差很大
            new_locs = new_locs[..., [1, 0]]
            # 最后new_locs输出的顺序应该就是x方向的偏移在前,y方向的偏移在后
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        #return F.grid_sample(src, new_locs, align_corners=True, mode="nearest")

class VoxelMorph2d(nn.Module):
    def __init__(self, in_channels, config_train, use_gpu=False, device = "cpu"):
        super(VoxelMorph2d, self).__init__()
        ## unet的输出就是displacement field
        if (config_train['train']["big_resolution"] == False):
            self.unet = UNet(in_channels, 2)
            self.final_layer = final_block(64, 32,  2)
        elif (config_train['train']["big_resolution"] == True):
            self.unet = UNet_big(in_channels, 2)
            self.final_layer = final_block(32, 16,  2)

        shape = (config_train['train']["model_image_width"], config_train['train']["model_image_width"])
        self.transformer = vxm_SpatialTransformer(shape, device)
        
        self.use_guide_map = config_train['train']["use_guide_map"]
        self.use_score_map = config_train['train']["use_score_map"]
        if (self.use_score_map):
            self.at_head = attention_head(64,30)
        if ( self.use_guide_map):
            self.guide_attention = AttentionModule(64, 64)
        if use_gpu:
            self.unet = self.unet.to(device)
            self.transformer = self.transformer.to(device)
            self.final_layer = self.final_layer.to(device)
            if ( self.use_guide_map):
                self.guide_attention = self.guide_attention.to(device)
            if (self.use_score_map):
                self.at_head = self.at_head.to(device)

    def forward(self, moving_image, fixed_image, guide_map):
        x = torch.cat([moving_image, fixed_image], dim=1)
        decode_block1 = self.unet(x)
        if (self.use_guide_map): 
            decode_block0 = self.guide_attention(decode_block1, guide_map)
            deformation_matrix = self.final_layer(decode_block0)
            if (self.use_score_map):
                 attention_score_matrix = self.at_head(decode_block0)
        else:
            deformation_matrix = self.final_layer(decode_block1)
            if (self.use_score_map):
                 attention_score_matrix = self.at_head(decode_block1)
        registered_image = self.transformer(moving_image, deformation_matrix)
        if (self.use_score_map):
            return registered_image, deformation_matrix, attention_score_matrix
        else:
            return registered_image, deformation_matrix


def cross_correlation_loss(I, J, n, device):
    I = I.permute(0, 3, 1, 2)
    J = J.permute(0, 3, 1, 2)
    batch_size, channels, xdim, ydim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    sum_filter = torch.ones((1, channels, n, n))
    # cuda
    sum_filter = sum_filter.to(device)
    I_sum = torch.conv2d(I, sum_filter, padding=1, stride=(1,1))
    J_sum = torch.conv2d(J, sum_filter,  padding=1 ,stride=(1,1))
    I2_sum = torch.conv2d(I2, sum_filter, padding=1, stride=(1,1))
    J2_sum = torch.conv2d(J2, sum_filter, padding=1, stride=(1,1))
    IJ_sum = torch.conv2d(IJ, sum_filter, padding=1, stride=(1,1))
    win_size = n**2
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
    cc = cross*cross / (I_var*J_var + np.finfo(float).eps)
    return torch.mean(cc)

## 原版本voxelmorph里面的ncc实现
class ncc_loss_fun:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, device = "cpu", win=None):
        self.win = win
        print("[========= win info]", win)
        self.device =device

    def ncc_loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if (self.win is None):
            win = [9] * ndims
        else:
            win= [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

    
## 原版本voxelmorph里面的mse实现
class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2) * 20 
    
    def loss_sum(self, y_true, y_pred):
        return torch.sum((y_true - y_pred) ** 2)
    
    def loss_l1(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred)) * 5
    
class smooth_loss:
    def smooothing_loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        dy = dy * dy
        dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d/2.0


class Grad(nn.Module):
    """
    N-D gradient loss
    """
    def __init__(self, penalty='l2'):
        super(Grad, self).__init__()
        self.penalty = penalty
    
    def _diffs(self, y):#y shape(bs, nfeat, vol_shape)
        ndims = y.ndimension() - 2
        df = [None] * ndims
        for i in range(ndims):
            d = i + 2#y shape(bs, c, d, h, w)
            # permute dimensions to put the ith dimension first
#            r = [d, *range(d), *range(d + 1, ndims + 2)]
            print("smooth debug2:", i, d, y.shape)
            y = y.permute(d, *range(d), *range(d + 1, ndims + 2))
            print("smooth debug3:", i, d, y.shape)
            dfi = y[1:, ...] - y[:-1, ...]
            print("smooth debug4:", dfi.shape)
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
#            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2))
            print("smooth debug5:", dfi[i].shape)
        
        return df
    
    def forward(self, pred):
        #根据张量的维度，设置ndim为2
        ndims = pred.ndimension() - 2
        
        
        if pred.is_cuda:
            df = Variable(torch.zeros(1).cuda())
        else:
            df = Variable(torch.zeros(1))
        print("smooth debug1:", df)
       
        for f in self._diffs(pred):
            if self.penalty == 'l1':
                df += f.abs().mean() / ndims
            else:
                assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
                df += f.pow(2).mean() / ndims
        return df




def dice_score(y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return dice



if __name__ == "__main__":
    pred_tensor = torch.ones((1, 3, 20, 20))
    target_tensor1 = torch.zeros((1, 3, 20, 20)) + torch.ones((1, 3, 20, 20)) * 0.9
    target_tensor = torch.rand((1, 3, 20, 20))
    import cv2
    ## ideal test
    img1 = "/home/yepeng_liu/code_python/keypoints_benchmark/unofficial-voxelmorph/VoxelMorph-PyTorch-master/fire-fundus-image-registration-dataset/A02_1.jpg"
    img2 = "/home/yepeng_liu/code_python/keypoints_benchmark/unofficial-voxelmorph/VoxelMorph-PyTorch-master/fire-fundus-image-registration-dataset/A02_2.jpg"
    im1 = cv2.imread(img1)
    im2 = cv2.imread(img2)
    # 将图像转换为PyTorch的Tensor
    image_tensor1 = torch.from_numpy(np.transpose(im1, (2, 0, 1))).float()
    # 添加批次维度（假设只有一张图像）
    image_tensor1 = image_tensor1.unsqueeze(0) / 255

    # 将图像转换为PyTorch的Tensor
    image_tensor2 = torch.from_numpy(np.transpose(im2, (2, 0, 1))).float()
    # 添加批次维度（假设只有一张图像）
    image_tensor2 = image_tensor2.unsqueeze(0) / 255

    print(dice_score(image_tensor1, image_tensor2))