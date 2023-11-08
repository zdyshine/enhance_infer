import cv2
import ffmpeg
import glob
import mimetypes
import numpy as np
import os
import shutil
import subprocess
import torch
from os import path as osp
from tqdm import tqdm
import time


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def get_video_meta_info(video_path):
    """get the meta info of the video by using ffprobe with python interface"""
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    # ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['audio'] = None
    try:
        ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    except KeyError:  # bilibili transcoder dont have nb_frames
        ret['duration'] = float(probe['format']['duration'])
        ret['nb_frames'] = int(ret['duration'] * ret['fps'])
        print(ret['duration'], ret['nb_frames'])
    return ret


def get_sub_video(args, num_process, process_idx):
    """Cut the whole video into num_process parts, return the process_idx-th part"""
    if num_process == 1:
        return args.input
    meta = get_video_meta_info(args.input)
    duration = int(meta['nb_frames'] / meta['fps'])
    part_time = duration // num_process
    print(f'duration: {duration}, part_time: {part_time}')
    out_path = osp.join(args.output, 'inp_sub_videos', f'{process_idx:03d}.mp4')
    cmd = [
        args.ffmpeg_bin,
        f'-i {args.input}',
        f'-ss {part_time * process_idx}',
        f'-to {part_time * (process_idx + 1)}' if process_idx != num_process - 1 else '',
        '-async 1',
        out_path,
        '-y',
    ]
    print(' '.join(cmd))
    subprocess.call(' '.join(cmd), shell=True)
    return out_path


class Reader:
    """read frames from a video stream or frames list"""

    def __init__(self, args, total_workers=1, worker_idx=0, device=torch.device('cuda')):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        # self.input_type = 'folder' if input_type is None else input_type
        self.input_type = 'video'
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            # read bgr from stream, which is the same format as opencv
            self.stream_reader = (
                ffmpeg
                .input(video_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel='error') # bgr24 -> rgb24
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )  # yapf: disable  # noqa
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']

        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot_frames = len(paths)
                num_frame_per_worker = tot_frames // total_workers + (1 if tot_frames % total_workers else 0)
                self.paths = paths[num_frame_per_worker * worker_idx:num_frame_per_worker * (worker_idx + 1)]

            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            from PIL import Image
            tmp_img = Image.open(self.paths[0])  # lazy load
            self.width, self.height = tmp_img.size
        self.idx = 0
        self.device = device

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        """the fps of sr video is set to the user input fps first, followed by the input fps,
        If the first two values are None, then the commonly used fps 24 is set"""
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        """return the number of frames for this worker, however, this may be not accurate for video stream"""
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            # end of stream
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame_from_list(self):
        if self.idx >= self.nb_frames:
            return None
        img = cv2.imread(self.paths[self.idx])
        self.idx += 1
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            img = self.get_frame_from_stream()
        else:
            img = self.get_frame_from_list()

        if img is None:
            raise StopIteration

        # bgr uint8 numpy -> rgb float32 [0, 1] tensor on device
        img = img.astype(np.float32) / 255.
        # img = mod_crop(img, self.args.mod_scale)
        # img = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(self.device)
        img = img2tensor(img, bgr2rgb=False, float32=True).unsqueeze(0).to(self.device)
        if self.args.half:
            # half precision won't make a big impact on visuals
            img = img.half()
        return img

    def close(self):
        # close the video stream
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()


class Writer:
    """write frames to a video stream"""

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')
        # out_width, out_height = 800*4, 50*4
        # out_width, out_height = 3424, 240 # 3652, 256 # pad -> 3424x240
        # out_width, out_height = 3424, 1920 # 3652, 256 # pad -> 3424x240
        vsp = video_save_path
        if audio is not None:
            self.stream_writer = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{out_width}x{out_height}', framerate=fps)
                .output(audio, vsp, pix_fmt='yuv420p', vcodec='libx264', loglevel='error', acodec='copy')
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )  # yapf: disable  # noqa
        else:
            # self.stream_writer = (
            #     ffmpeg
            #     .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{out_width}x{out_height}', framerate=fps)
            #     .output(vsp, preset='slower', pix_fmt='yuv420p',
            #                 # vcodec='libx264', x264opts=f'force-cfr:fps={fps}:qp=12:colorprim=bt709:transfer=bt709:colormatrix=bt709',
            #                 vcodec='libx264', x264opts=f'force-cfr:fps={fps}:qp=12',
            #                 loglevel='error')
            #     .overwrite_output()
            #     .run_async(pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin)
            # )
         # 针对老片的，添加aspect='4:3'
            self.stream_writer = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                    video_save_path, aspect='4:3', preset='slower',
                    pix_fmt='yuv420p', vcodec='libx264', x264opts='force-cfr:fps=25:qp=10:colorprim=bt709:transfer=bt709:colormatrix=bt709',
                    loglevel='error').overwrite_output().run_async(
                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        self.out_width = out_width
        self.out_height = out_height
        self.args = args

    def write_frame(self, frame):
        if self.args.outscale != self.args.netscale:
            frame = cv2.resize(frame, (self.out_width, self.out_height), interpolation=cv2.INTER_LANCZOS4)
        self.stream_writer.stdin.write(frame.tobytes())

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

initialized_logger = {}


class AvgTimer():
    def __init__(self, window=200):
        self.window = window  # average window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start()

    def start(self):
        self.start_time = self.tic = time.time()

    def record(self):
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count

        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.time()

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time