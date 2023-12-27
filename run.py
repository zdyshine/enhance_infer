import os, glob
import cv2
import subprocess
'''
docker start -ia 1fd7884f0cdc
docker start -ia ffmpegdnn
copy 模型
cp /data/y/GPU-Server/test/zhangdy/code_zdy/code_basicsr/BasicSR/experiments/Down_001_Netfix_ps512_UPbicubic_Grad/models/net_g_226000.pth /home/zhoujs/mgtvML_FFmpeg/NNResize/checkpoints/DownNetNFv1_Grad_226000.pth
修改代码
vim /home/zhoujs/mgtvML_FFmpeg/NNResize/NNResize_GPU.py

邮件请以团队为单位，按照统一格式命名：颁奖典礼+【赛道】+【团队名】

'''
# ffmpeg -i input.mp4 -t 300 -c:v libx264 -crf 26  -preset slow -an output_crf26.mp4
# NET_NAME = 'DownNetNFv1' # DownNetNFv1_196000.pth
NET_NAME = 'DownNetNFv1_Grad'
out_H= 480 # 540 # 480
out_W= 848 # 960 # 848

NTE_LOAD='DownNetNFv1_Grad_226000.pth'

def nndown_bicubicup():
    # video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源/*')
    video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源_动漫/*')
    output_dir = f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/{NET_NAME}/nndown{out_H}_bicubicup'
    os.makedirs(output_dir, exist_ok=True)
    # video sr
    for video in video_list:
            base_name = os.path.basename(video)
            output_video = os.path.join(output_dir, base_name.split('.')[0] + '.y4m')
            command_NNDownUP = (
                        'CUDA_VISIBLE_DEVICES=1 /home/zhoujs/mgtvML_FFmpeg/ffmpeg'
                        + ' -an'
                        + ' -ss 00:02:00 -to 00:03:00'
                        + ' -i "{}"'.format(video)
                        # + ' -vf "[INPUT]format=rgb24,mgtvpythonisr=path=/home/zhoujs/mgtvML_FFmpeg/NNResize/:model=/home/zhoujs/mgtvML_FFmpeg/NNResize/checkpoints/DownNetNFv1_196000.pth.pth:multiple=1:width=960:height=540:module_name=NNResize_GPU[FID0];[FID0]scale=w=1920:h=1080:sws_flags=lanczos[OUTPUT]"'
                        + f' -vf "[INPUT]format=rgb24,mgtvpythonisr=path=/home/zhoujs/mgtvML_FFmpeg/NNResize/:model=/home/zhoujs/mgtvML_FFmpeg/NNResize/checkpoints/{NTE_LOAD}:multiple=1:width={out_W}:height={out_H}:module_name=NNResize_GPU[FID0];[FID0]scale=w=1920:h=1080:sws_flags=bicubic[OUTPUT]"'
                        + ' -pix_fmt yuv420p -r 25 -y'
                        + ' "{}"'.format(output_video)
                    )
            print(command_NNDownUP)
            subprocess.call(command_NNDownUP, shell=True)

def nndown_lanczosup():
    # video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源/*')
    video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源_动漫/*')
    output_dir = f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/{NET_NAME}/nndown{out_H}_lanczosup'
    os.makedirs(output_dir, exist_ok=True)
    # video sr
    for video in video_list:
            base_name = os.path.basename(video)
            output_video = os.path.join(output_dir, base_name.split('.')[0] + '.y4m')
            command_NNDownUP = (
                        'CUDA_VISIBLE_DEVICES=1 /home/zhoujs/mgtvML_FFmpeg/ffmpeg'
                        + ' -an'
                        + ' -ss 00:02:00 -to 00:03:00'
                        + ' -i "{}"'.format(video)
                        # + ' -vf "[INPUT]format=rgb24,mgtvpythonisr=path=/home/zhoujs/mgtvML_FFmpeg/NNResize/:model=/home/zhoujs/mgtvML_FFmpeg/NNResize/checkpoints/DownNetNFv1_196000.pth.pth:multiple=1:width=960:height=540:module_name=NNResize_GPU[FID0];[FID0]scale=w=1920:h=1080:sws_flags=lanczos[OUTPUT]"'
                        + f' -vf "[INPUT]format=rgb24,mgtvpythonisr=path=/home/zhoujs/mgtvML_FFmpeg/NNResize/:model=/home/zhoujs/mgtvML_FFmpeg/NNResize/checkpoints/{NTE_LOAD}:multiple=1:width={out_W}:height={out_H}:module_name=NNResize_GPU[FID0];[FID0]scale=w=1920:h=1080:sws_flags=lanczos[OUTPUT]"'
                        + ' -pix_fmt yuv420p -r 25 -y'
                        + ' "{}"'.format(output_video)
                    )
            print(command_NNDownUP)
            subprocess.call(command_NNDownUP, shell=True)


def lanczosdown_lanczosup():
    # video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源/*')
    video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源_动漫/*')
    output_dir = f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/lanczosdown{out_H}_lanczosup'
    os.makedirs(output_dir, exist_ok=True)
    # video sr
    for video in video_list:
            base_name = os.path.basename(video)
            output_video = os.path.join(output_dir, base_name.split('.')[0] + '.y4m')
            command_lanczos_down_up = (
                    'CUDA_VISIBLE_DEVICES=1 /home/zhoujs/mgtvML_FFmpeg/ffmpeg'
                    + ' -an'
                    + ' -ss 00:02:00 -to 00:03:00'
                    + ' -i "{}"'.format(video)
                    # + ' -t 60'
                    + ' -pix_fmt yuv420p'
                    + f' -vf "[INPUT]scale=w={out_W}:h={out_H}:sws_flags=lanczos[FID0];[FID0]scale=w=1920:h=1080:sws_flags=lanczos[OUTPUT]"'
                    + ' -r 25 -y '
                    + ' "{}"'.format(output_video)
            )
            print(command_lanczos_down_up)
            subprocess.call(command_lanczos_down_up, shell=True)

def lanczosdowns_bicubicup():
    # video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源/*')
    video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源_动漫/*')
    output_dir = f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/lanczosdown{out_H}_bicubicup_tmp'
    os.makedirs(output_dir, exist_ok=True)
    # video sr
    for video in video_list:
            base_name = os.path.basename(video)
            output_video = os.path.join(output_dir, base_name.split('.')[0] + '.y4m')
            command_lanczos_down_up = (
                    'CUDA_VISIBLE_DEVICES=1 /home/zhoujs/mgtvML_FFmpeg/ffmpeg'
                    + ' -an'
                    + ' -ss 00:02:00 -to 00:03:00'
                    + ' -i "{}"'.format(video)
                    + ' -pix_fmt yuv420p'
                    + f' -vf "[INPUT]scale=w={out_W}:h={out_H}:sws_flags=lanczos[FID0];[FID0]scale=w=1920:h=1080:sws_flags=bicubic[OUTPUT]"'
                    + ' -r 25 -y '
                    + ' "{}"'.format(output_video)
            )
            print(command_lanczos_down_up)
            subprocess.call(command_lanczos_down_up, shell=True)

def cal_psnr():
    # ffmpeg -i N:\zhangdy\workdir\NN降采样\测试视频源\mg_test_0809_ref.y4m
    # -i N:\zhangdy\workdir\NN降采样\DownNet\upx2_crf26\upx2mg_test_0809_DownNet_up2.y4m
    # -lavfi psnr=stats_file=psnr_logfile.txt -f null -

    sorce_dir = '/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源'
    codec_dir = '/data/y/GPU-Server/test/zhangdy/workdir/NNDown/DownNetNFv1_Grad/'
    codec_name = 'nndown540_upbicubic'
    out_txt = os.path.join(codec_dir, codec_name+'_psnr.txt')

    video_list = sorted(os.listdir(sorce_dir))

    # video sr
    for video in video_list:
            command_cal_psnr = (
                    'CUDA_VISIBLE_DEVICES=1 /home/zhoujs/mgtvML_FFmpeg/ffmpeg'
                    + ' -t 60'
                    + ' -i "{}"'.format(os.path.join(sorce_dir, video))
                    + ' -i "{}"'.format(os.path.join(codec_dir, codec_name, video.replace('mp4', 'y4m')))
                    + ' -lavfi psnr=stats_file="{}" -f null -'.format(out_txt)
            )
            # command_cal_psnr = (
            #         'CUDA_VISIBLE_DEVICES=1 /home/zhoujs/mgtvML_FFmpeg/ffmpeg'
            #         + ' -t 60'
            #         + ' -i "{}"'.format(os.path.join(sorce_dir, video))
            #         + ' -i "{}"'.format(os.path.join(codec_dir, codec_name, video.replace('mp4', 'y4m')))
            #         + ' -filter_complex "psnr" -f null /dev/null'
            # )
            # ffmpeg -i 1.mov -i 2.ts -filter_complex "ssim" -f null /dev/null
            print(command_cal_psnr)
            subprocess.call(command_cal_psnr, shell=True)
            exit()
    # # grep -oP 'psnr_avg:\K[0-9.]+' nndown540_upbicubic_psnr.txt | awk '{ total += $1 } END { print "Average PSNR:", total/NR }' > nndown540_upbicubic_psnr_psnr.txt
    #         command_merge_psnr = (
    #             'grep -oP "psnr_avg:\K[0-9.]+"'
    #             + f' {out_txt} | awk '
    #             + ' "{ total += $1 } END { print "Average PSNR:", total/NR }"'
    #             + f' > {out_txt.replace("psnr", "avg_psnr")}'
    #             )
    #         print(command_merge_psnr)
    #         subprocess.call(command_merge_psnr, shell=True)
    #         command_rmrf = (
    #                 f'rm -rf {out_txt}'
    #         )
    #         print(command_rmrf)
    #         exit()
    #         # subprocess.call(command_rmrf, shell=True)


# nndown_bicubicup()
# nndown_lanczosup()
# lanczosdown_lanczosup()
# lanczosdowns_bicubicup()
# # cal_psnr()

def nndown_test():
    video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源/*')
    # video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源_动漫/*')
    output_dir = f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/temp_test/nndown{out_H}'
    os.makedirs(output_dir, exist_ok=True)
    # video sr
    for video in sorted(video_list)[:1]:
            base_name = os.path.basename(video)
            output_video = os.path.join(output_dir, base_name.split('.')[0] + f'_{out_H}.y4m')
            command_NNDownUP = (
                        'CUDA_VISIBLE_DEVICES=1 /home/zhoujs/mgtvML_FFmpeg/ffmpeg'
                        + ' -i "{}"'.format(video)
                        + ' -t 10'
                        # + f' -vf "format=rgb24,mgtvpythonisr=path=/home/zhoujs/mgtvML_FFmpeg/NNResize/:model=/home/zhoujs/mgtvML_FFmpeg/NNResize/checkpoints/{NTE_LOAD}:multiple=1:width={out_W}:height={out_H}:module_name=NNResize_GPU"'
                        + f' -vf "[INPUT]format=rgb24,mgtvpythonisr=path=/home/zhoujs/mgtvML_FFmpeg/NNResize/:model=/home/zhoujs/mgtvML_FFmpeg/NNResize/checkpoints/{NTE_LOAD}:multiple=1:width={out_W}:height={out_H}:module_name=NNResize_GPU[FID0];'
                          f'[FID0]scale=w=1920:h=1080:sws_flags=bicubic[FID1];[FID1]setsar=sar=16/9[FID2];[FID2]setdar=dar=16/9[OUTPUT]"'
                        + ' -pix_fmt yuv420p -r 25 -y'
                        + ' "{}"'.format(output_video)
                    )
            print(command_NNDownUP)
            # subprocess.call(command_NNDownUP, shell=True)


def lanczosdown_test():
    video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源/*')
    # video_list = glob.glob(f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/测试视频源_动漫/*')
    output_dir = f'/data/y/GPU-Server/test/zhangdy/workdir/NNDown/temp_test/lanczosdown{out_H}'
    os.makedirs(output_dir, exist_ok=True)
    # video sr
    for video in sorted(video_list)[:1]:
            base_name = os.path.basename(video)
            output_video = os.path.join(output_dir, base_name.split('.')[0] + '.y4m')
            command_lanczos_down_up = (
                    'CUDA_VISIBLE_DEVICES=1 /home/zhoujs/mgtvML_FFmpeg/ffmpeg'
                    + ' -i "{}"'.format(video)
                    + ' -t 10'
                    + ' -pix_fmt yuv420p'
                    + f' -vf "scale=w={out_W}:h={out_H}:sws_flags=lanczos,setsar=sar=16/9,setdar=dar=16/9"'
                    + ' -r 25 -y '
                    + ' "{}"'.format(output_video)
            )
            print(command_lanczos_down_up)
            # subprocess.call(command_lanczos_down_up, shell=True)

nndown_test()
lanczosdown_test()
