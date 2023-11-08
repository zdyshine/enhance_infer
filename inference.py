import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import cv2
from PIL import Image #导入PIL库
import os

def img2tensor(img):
    img_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255.).float()
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)

def check_image_size(x, padder_size):
    _, _, h, w = x.size()
    mod_pad_h = (padder_size - h % padder_size) % padder_size
    mod_pad_w = (padder_size - w % padder_size) % padder_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
    return x, h, w

def tile_process(model, img, scale=4, tile_size = 640, tile_pad = 10):
    """It will first crop input images to tiles, and then process each tile.
    Finally, all the processed tiles are merged into one images.

    Modified from: https://github.com/ata4/esrgan-launcher
    """
    scale = scale
    tile_size = tile_size
    tile_pad = tile_pad
    batch, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (batch, channel, output_height, output_width)

    # start with black image
    output = img.new_zeros(output_shape)
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_size
            ofs_y = y * tile_size
            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            tile_idx = y * tiles_x + x + 1
            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # upscale tile
            try:
                with torch.no_grad():
                    output_tile = model(input_tile)
            except RuntimeError as error:
                print('Error', error)
            print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, output_start_y:output_end_y,
            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                           output_start_x_tile:output_end_x_tile]
    return output

if __name__=='__main__':
    # 开源模型
    # ------------- 模型1-RealESRGAN
    scale = 2
    from archs.rrdbnet_arch import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale).cuda()
    model_path = r'./checkpoints/RealESRGAN/RealESRGAN_x2plus.pth'

    # # ------------- 模型2-SAFMN(潘金山)
    # scale = 2
    # from archs.safmn_arch import SAFMN
    # model = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=2).cuda()
    # model_path = r'./checkpoints/SAFMN/SAFMN_L_Real_LSDIR_x2.pth'

    # # # ------------- 模型3-DiffIR
    # # x1
    # scale = 1
    # model_path = r'./checkpoints/DiffIR/1x/DiffIRS2-GANx1-V2/RealworldSR-DiffIRS2-GANx1-V2.pth'
    #
    # # # x2
    # # scale = 2
    # # model_path = r'./checkpoints/DiffIR/2x/DiffIRS2-GANx2-V2/RealworldSR-DiffIRS2-GANx2-V2.pth'
    #
    # from archs.diffIRS2_arch import DiffIRS2
    # model = DiffIRS2( n_encoder_res= 9, dim= 64, scale=scale,num_blocks= [13,1,1,1],num_refinement_blocks= 13,
    #                           heads= [1,2,4,8], ffn_expansion_factor= 2.2,LayerNorm_type= "BiasFree").cuda()

    # # ------------ load model
    load_checkpoint_basicsr(model, model_path)
    model.eval()
    img_path_list = glob.glob(r'./*.png')
    save_path = r'./part_out/realesrganx2'
    os.makedirs(save_path, exist_ok=True)

    for img_path in img_path_list:
        imgname, extension = os.path.splitext(os.path.basename(img_path))
        extension = '.png'

        img = np.array(Image.open(img_path))
        if img.shape[-1]==4:
            img = img[:,:,:3]
        # h, w, _ = img.shape
        # img = cv2.resize(img, (w//8, h//8))
        print(img_path, img.shape)
        # img = img[2640:2640+640, 2640:2640+640, :]
        # img = np.array(img)

        img_tensor = img2tensor(img[:, :, ::-1])
        img_tensor, H, W = check_image_size(img_tensor, padder_size=16)
        if scale!=1:
            img = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_LANCZOS4)
        # cv2.imshow('img', img)

        with torch.no_grad():
            try:
                out = model(img_tensor.cuda())
            # OOM时切块处理
            except:
                out = tile_process(model, img_tensor.cuda(), scale=scale, tile_size = 640, tile_pad = 10)
                out = out[:, :, :H*scale, :W*scale]

        out = tensor2img(out[:, :, :H*scale, :W*scale])[:, :, ::-1]
        out = Image.fromarray(out)
        out.save(os.path.join(save_path, f'{imgname}{extension}'))

        # cv2.imshow('out', out)
        # cv2.waitKey()
