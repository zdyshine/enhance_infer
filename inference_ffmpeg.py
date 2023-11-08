from ffmpeg_utils import *
import torch
import cv2
import numpy as np
import os.path
from define_network import *
import argparse
import os.path as osp

@torch.no_grad()
def inference_video(args, video_save_path, device=None, total_workers=1, worker_idx=0):
    # prepare model
    # model = get_inference_model(args, device)
    # args = get_parser()
    use_cuda = True
    device = torch.device('cuda' if use_cuda else 'cpu')

    model, _ = DiffIR_SR()

    # torch.cuda.empty_cache()

    # prepare reader and writer
    reader = Reader(args, total_workers, worker_idx, device=device)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    height = height - height % args.mod_scale
    width = width - width % args.mod_scale
    fps = reader.get_fps()
    print('=' * 21, f'Video Info width:{width}, height:{height}, fps:{fps}.', '=' * 21)
    writer = Writer(args, audio, height, width, video_save_path, fps)

    # initialize pre/cur/nxt frames, pre sr frame, and pre hidden state for inference
    end_flag = False
    cur = reader.get_frame()

    pbar = tqdm(total=len(reader), unit='frame', desc='inference')
    model_timer = AvgTimer()  # model inference time tracker
    i_timer = AvgTimer()  # I(input read) time tracker
    o_timer = AvgTimer()  # O(output write) time tracker
    while True:
        # inference at current step
        torch.cuda.synchronize(device=device)
        model_timer.start()

        out = model(cur)
        torch.cuda.synchronize(device=device)
        model_timer.record()

        # write current sr frame to video stream
        torch.cuda.synchronize(device=device)
        o_timer.start()
        output_frame = tensor2img(out, rgb2bgr=False)
        writer.write_frame(output_frame)
        torch.cuda.synchronize(device=device)
        o_timer.record()

        # if end of stream, break
        if end_flag:
            break

        # move the sliding window
        torch.cuda.synchronize(device=device)
        i_timer.start()
        # prev = cur
        try:
            cur = reader.get_frame()
        except StopIteration:
            # cur = prev
            end_flag = True
        torch.cuda.synchronize(device=device)
        i_timer.record()

        # update&print infomation
        pbar.update(1)
        pbar.set_description(
            f'I: {i_timer.get_avg_time():.4f} O: {o_timer.get_avg_time():.4f} Model: {model_timer.get_avg_time():.4f}')

    reader.close()
    writer.close()


def run(args):
    if args.suffix is None:
        args.suffix = ''
    else:
        args.suffix = f'_{args.suffix}'
    # video_save_path = osp.join(args.output, f'{args.video_name}{args.suffix}.mp4')
    video_save_path = osp.join(args.output, f'{args.video_name}{args.suffix}.ts')

    # set up multiprocessing
    num_gpus = torch.cuda.device_count()
    num_process = num_gpus * args.num_process_per_gpu
    if num_process == 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_video(args, video_save_path, device=device)
        return

    ctx = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    out_sub_videos_dir = osp.join(args.output, 'out_sub_videos')
    os.makedirs(out_sub_videos_dir, exist_ok=True)
    os.makedirs(osp.join(args.output, 'inp_sub_videos'), exist_ok=True)

    pbar = tqdm(total=num_process, unit='sub_video', desc='inference')
    for i in range(num_process):
        sub_video_save_path = osp.join(out_sub_videos_dir, f'{i:03d}.mp4')
        pool.apply_async(
            inference_video,
            args=(args, sub_video_save_path, torch.device(i % num_gpus), num_process, i),
            callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()

    # combine sub videos
    # prepare vidlist.txt
    with open(f'{args.output}/vidlist.txt', 'w') as f:
        for i in range(num_process):
            f.write(f'file \'out_sub_videos/{i:03d}.mp4\'\n')
    # To avoid video&audio desync as mentioned in https://github.com/xinntao/Real-ESRGAN/issues/388
    # we use the solution provided in https://stackoverflow.com/a/52156277 to solve this issue
    cmd = [
        args.ffmpeg_bin,
        '-f', 'concat',
        '-safe', '0',
        '-i', f'{args.output}/vidlist.txt',
        '-c:v', 'copy',
        '-af', 'aresample=async=1000',
        video_save_path,
        '-y',
    ]  # yapf: disable
    print(' '.join(cmd))
    subprocess.call(cmd)
    shutil.rmtree(out_sub_videos_dir)
    shutil.rmtree(osp.join(args.output, 'inp_sub_videos'))
    os.remove(f'{args.output}/vidlist.txt')


def main():
    """Inference demo for AnimeSR.
    It mainly for restoring anime videos.
    """
    parser = argparse.ArgumentParser()
    # parser = get_parser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='input test image folder or video path')
    parser.add_argument('-o', '--output', type=str, default='results', help='save image/video path')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='AnimeSR_v2',
        help='Model names: AnimeSR_v2 | AnimeSR_v1-PaperModel. Default:AnimeSR_v2')
    parser.add_argument(
        '-s',
        '--outscale',
        type=int,
        default=2,
        help='The netscale is x4, but you can achieve arbitrary output scale (e.g., x2) with the argument outscale'
             'The program will further perform cheap resize operation after the AnimeSR output. '
             'This is useful when you want to save disk space or avoid too large-resolution output')
    parser.add_argument(
        '--expname', type=str, default='animesr', help='A unique name to identify your current inference')
    parser.add_argument(
        '--netscale',
        type=int,
        default=2,
        help='the released models are all x4 models, only change this if you train a x2 or x1 model by yourself')
    parser.add_argument(
        '--mod_scale',
        type=int,
        default=2,
        help='the scale used for mod crop, since AnimeSR use a multi-scale arch, so the edge should be divisible by 4')
    parser.add_argument('--fps', type=int, default=None, help='fps of the sr videos')
    parser.add_argument('--half', action='store_true', help='use half precision to inference')
    parser.add_argument(
        '--extract_frame_first',
        action='store_true',
        help='if input is a video, you can still extract the frames first, other wise AnimeSR will read from stream')
    parser.add_argument(
        '--num_process_per_gpu', type=int, default=1, help='the total process is number_process_per_gpu * num_gpu')
    parser.add_argument(
        '--suffix', type=str, default=None, help='you can add a suffix string to the sr video name, for example, x2')
    args = parser.parse_args()
    # args.ffmpeg_bin = os.environ.get('ffmpeg_exe_path', 'ffmpeg')
    args.ffmpeg_bin = '/test/ffmpeg4.4'

    args.input = args.input.rstrip('/').rstrip('\\')
    print('=================', args.input)
    # if mimetypes.guess_type(args.input)[0] is not None and mimetypes.guess_type(args.input)[0].startswith('video'):
    #     is_video = True
    # else:
    #     is_video = False
    is_video = True
    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    # prepare input and output
    args.video_name = osp.splitext(osp.basename(args.input))[0]
    # args.output = osp.join(args.output, args.expname, 'videos', args.video_name)
    args.output = osp.join(args.output, args.expname)
    os.makedirs(args.output, exist_ok=True)
    if args.extract_frame_first:
        inp_extracted_frames = osp.join(args.output, 'inp_extracted_frames')
        os.makedirs(inp_extracted_frames, exist_ok=True)
        video_util.video2frames(args.input, inp_extracted_frames, force=True, high_quality=True)
        video_meta = get_video_meta_info(args.input)
        args.fps = video_meta['fps']
        args.input = inp_extracted_frames

    run(args)

    if args.extract_frame_first:
        shutil.rmtree(args.input)


# single gpu and single process inference
# CUDA_VISIBLE_DEVICES=0 python scripts/inference_animesr_video.py -i inputs/TheMonkeyKing1965.mp4 -n AnimeSR_v2 -s 2 --expname animesr_v2 --num_process_per_gpu 1 --suffix 1gpu1process
# # single gpu and multi process inference (you can use multi-processing to improve GPU utilization)
# CUDA_VISIBLE_DEVICES=0 python scripts/inference_animesr_video.py -i inputs/TheMonkeyKing1965.mp4 -n AnimeSR_v2 -s 2 --expname animesr_v2 --num_process_per_gpu 3 --suffix 1gpu3process
# # multi gpu and multi process inference
# CUDA_VISIBLE_DEVICES=0,1 python scripts/inference_animesr_video.py -i inputs/TheMonkeyKing1965.mp4 -n AnimeSR_v2 -s 2 --expname animesr_v2 --num_process_per_gpu 3 --suffix 2gpu6process
if __name__ == '__main__':
    main()
    '''
    # SR
    CUDA_VISIBLE_DEVICES=0 python inference_ffmpeg.py -i /test/zhangdy/workdir/20230801_您向我们走来/0705您向我们走来.m4v  -o results -n resultsffmpeg -s 4 --expname textSR --num_process_per_gpu 1 --suffix textSR
    CUDA_VISIBLE_DEVICES=0 python inference_ffmpeg.py -i /test/zhangdy/workdir/temp/姐妹情仇_01-禾兆2023_10min.mpg  -o results -n resultsffmpeg -s 2 --expname DiffIR --num_process_per_gpu 1 --suffix DiffIRx2
    
    CUDA_VISIBLE_DEVICES=0 python inference_ffmpeg.py -i /test/zhangdy/workdir/MZD/video/《毛泽东遗物的故事》之单腿眼镜.mp4 -o results -n resultsDiffIR -s 2 --expname DiffIR --num_process_per_gpu 1 --suffix DiffIRx2
    '''