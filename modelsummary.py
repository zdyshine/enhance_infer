import torch

# from models.repaire.network_scunet import SCUNet as net
# print('SCUNet')
# # model = net(dim=64, config = [1, 1, 1, 1, 1, 1, 1]).cuda()
# model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64).cuda()
# # (1, 3, 1920, 1080), 0.361707 seconds

# from models.repaire.network_restormer import Restormer as net
# print('Restormer')
# model = net().cuda()

# from models.repaire.network_naf import NAFNet as net
# print('NAFNet')
# model = net(img_channel=3, width=16, middle_blk_num=8, enc_blk_nums=[1, 1, 4, 8], dec_blk_nums=[2, 2, 1, 1]).cuda()
# # (1, 3, 1920, 1080), 0.152756 seconds

# from models.repaire.network_pmrid import PMRID as net
# print('PMRID')
# model = net().cuda()
# # (1, 3, 1920, 1080), 0.258285 seconds

# from models.repaire.network_nafa import NAFA as net
# print('NAFA')
# model = net(img_channel=3, width=16, middle_blk_num=8, enc_blk_nums=[1, 1, 4, 4], dec_blk_nums=[2, 2, 1, 1]).cuda()
# # (1, 3, 1920, 1080), 0.187020 seconds

# from models.repaire_540p.network_nafa import NAFA as net
# print('NAFA')
# model = net(img_channel=3, width=16, middle_blk_num=8, enc_blk_nums=[1, 1, 4, 4], dec_blk_nums=[2, 2, 1, 1]).cuda()

from models.sr.nafasr_arch import NAFSR as net
print('NAFSR')
model = net(img_channel=3, width=32, middle_blk_num=4, enc_blk_nums=[2, 2], dec_blk_nums=[2, 2]).cuda()

# from models.modules.ESDSR_arch import ESDSR as net
# print('ESDSR')
# model = net().cuda()
# (1, 3, 1920, 1080), 0.187020 seconds
# import torch.nn as nn
# rescale = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
im_input = torch.rand(1, 3, 1920, 1080).cuda()
# im_input = torch.rand(1, 3, 960, 540).cuda()
runtime = []
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(20):
    with torch.no_grad():
        start.record()
        out = model(im_input)
        print(out.shape)
        end.record()
        torch.cuda.synchronize()
        time_cost = start.elapsed_time(end)  # milliseconds
        runtime.append(time_cost)
ave_runtime = sum(runtime) / len(runtime) / 1000.0
print('------> Average runtime of ({}) is : {:.6f} seconds'.format('model', ave_runtime))

# print(model)
# --------------------------------
# print model summary
# --------------------------------
# print_modelsummary = True
# if print_modelsummary:
#     from utils_ntire.utils_modelsummary import get_model_activation, get_model_flops
#
#     input_dim = (3, 256, 256)  # set the input dimension
#
#     activations, num_conv2d = get_model_activation(model, input_dim)
#     print('{:>16s} : {:<.4f} [M]'.format('#Activations', activations / 10 ** 6))
#     print('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))
#
#     flops = get_model_flops(model, input_dim, False)
#     print('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops / 10 ** 9))
#
#     num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
#     print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters / 10 ** 6))
