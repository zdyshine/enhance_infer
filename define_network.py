import torch
from copy import deepcopy

def load_checkpoint_basicsr(model, model_path, strict=True):
    # load checkpoint
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # param_key = 'params'
    param_key = 'params_ema'
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
        load_net = load_net[param_key]

    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=strict)

def MixNet_l():
    # 基础混合退化修复,自训练
    scale = 1
    model = MixNet_l().cuda()
    model_path = r'net_g_858000.pth'
    # # ------------ load model
    load_checkpoint_basicsr(model, model_path)
    model.eval()
    return model, scale

def MixNet_b():
    # 基础混合退化超分
    scale = 2
    model = MixNet_b().cuda()
    model_path = r'net_g_400000.pth'
    load_checkpoint_basicsr(model, model_path)
    model.eval()
    return model, scale

def RealESRGAN():
    # 开源模型：https://github.com/xinntao/Real-ESRGAN
    # ------------- 模型1-RealESRGAN
    scale = 2
    from archs.rrdbnet_arch import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale).cuda()
    model_path = r'./checkpoints/RealESRGAN/RealESRGAN_x2plus.pth'
    load_checkpoint_basicsr(model, model_path)
    model.eval()
    return model, scale

def SAFMN():
    # # ------------- 模型2-SAFMN(潘金山)：https://github.com/sunny2109/SAFMN
    scale = 2
    from archs.safmn_arch import SAFMN
    model = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=scale).cuda()
    model_path = r'./checkpoints/SAFMN/SAFMN_L_Real_LSDIR_x2.pth'
    load_checkpoint_basicsr(model, model_path)
    model.eval()
    return model, scale

def DiffIR_repaire():
    # # # ------------- 模型3-DiffIR：https://github.com/Zj-BinXia/DiffIR
    # x1
    scale = 1
    model_path = r'./checkpoints/DiffIR/1x/DiffIRS2-GANx1-V2/RealworldSR-DiffIRS2-GANx1-V2.pth'

    from archs.diffIRS2_arch import DiffIRS2
    model = DiffIRS2( n_encoder_res= 9, dim= 64, scale=scale,num_blocks= [13,1,1,1],num_refinement_blocks= 13,
                              heads= [1,2,4,8], ffn_expansion_factor= 2.2,LayerNorm_type= "BiasFree").cuda()

    # # ------------ load model
    load_checkpoint_basicsr(model, model_path)
    model.eval()
    return model, scale

def DiffIR_SR():
    # # # ------------- 模型3-DiffIR：https://github.com/Zj-BinXia/DiffIR
    # # x2
    scale = 2
    model_path = r'./checkpoints/DiffIR/2x/DiffIRS2-GANx2-V2/RealworldSR-DiffIRS2-GANx2-V2.pth'
    from archs.diffIRS2_arch import DiffIRS2
    model = DiffIRS2(n_encoder_res=9, dim=64, scale=scale, num_blocks=[13, 1, 1, 1], num_refinement_blocks=13,
                     heads=[1, 2, 4, 8], ffn_expansion_factor=2.2, LayerNorm_type="BiasFree").cuda()
    # # ------------ load model
    load_checkpoint_basicsr(model, model_path)
    model.eval()
    return model, scale
