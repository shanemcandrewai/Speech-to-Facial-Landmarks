""" Replication toolkit """
from pathlib import Path
import shutil
import urllib.request
import bz2
import subprocess as sp
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from scipy import stats
import dlib

class Frames:
    """ Frame file manager """
    root_dir = Path('..', 'replic')
    """ Toolkit working directory """
    frames_dir = Path(root_dir, 'frames')
    """ Location of frames extracted from video """
    suffix = '.jpeg'
    """ Frame file extension """
    num_len = 4
    """ Length of frame file sequence number """

    def __init__(self, frames_dir=None, suffix=None, num_len=None):
        if frames_dir is not None:
            type(self).frames_dir = Path(frames_dir)
        if suffix is not None:
            type(self).suffix = suffix
        if num_len is not None:
            type(self).num_len = num_len

    def get_file_path(self, frame_num=30):
        """ Build file path from frame number """
        return str(Path(self.frames_dir, str(frame_num).zfill(
            self.num_len)).with_suffix(self.suffix))

    def get_frame_file_names(self):
        """ Get list of frame files """
        return sorted(self.frames_dir.glob('*' + self.suffix))

    def get_frame_nums(self):
        """ Get list of frame numbers """
        frames = self.get_frame_file_names()
        return [int(Path(frame).stem) for frame in frames]

class DlibProcess:
    """ Dlib facial landmark extraction manager """
    rgb_image = None
    """ http://dlib.net/python/index.html?highlight=rgb_imag#dlib.load_rgb_image """
    frame_num = None
    """ Frame sequence number """
    shape = None
    """ http://dlib.net/python/index.html?highlight=shape_predictor#dlib.shape_predictor """
    detector = None
    """ http://dlib.net/python/index.html?highlight=get_frontal_face_detector#dlib.get_frontal_face_detector """
    predictor = None
    """ http://dlib.net/python/index.html?highlight=shape_predictor#dlib.shape_predictor """
    frames = None
    """ [Frames](#replication.Frames) """
    lmarks = np.empty((0, 68, 2))
    """ [ndarray](https://numpy.org/doc/1.18/reference/generated/numpy.ndarray.html?highlight=ndarray#numpy.ndarray) containing facial landmarks """
    lmarks_file = None
    """ Landmarks file """
    video_file = None
    """ Input video file """

    def __init__(self, video_file=None, lmarks_file=None, frames=None,
                 model_url='https://raw.github.com/davisking/dlib-models/master/'
                           'shape_predictor_68_face_landmarks.dat.bz2'):
        """ initialize DLib, download model if neccessary """
        if frames is None:
            type(self).frames = Frames()
        else:
            type(self).frames = frames

        if video_file is None:
            type(self).video_file = Path(self.frames.root_dir, 'shared', 'obama2s.mp4')
        else:
            type(self).video_file = Path(video_file)

        if lmarks_file is None:
            type(self).lmarks_file = Path(self.frames.root_dir, 'data',
                                          Path(self.video_file).with_suffix('.npy').name)
        else:
            type(self).lmarks_file = Path(lmarks_file)
        type(self).lmarks_file.parent.mkdir(parents=True, exist_ok=True)

        model_file = Path(Path('..', 'data'), Path(model_url).stem)
        if not model_file.is_file():
            print('Model ' + str(model_file) + ' not found')
            print('Downloading from ' + model_url)
            with urllib.request.urlopen(model_url) as response, open(
                    model_file, 'wb') as model:
                model.write(bz2.decompress(response.read()))
        type(self).detector = dlib.get_frontal_face_detector()
        type(self).predictor = dlib.shape_predictor(str(model_file))

    def get_shape(self, frame_num=30, face_num=0):
        """ load image and attempt to extract shape predictor """
        if frame_num != self.frame_num or self.shape is None:
            image_file_path = self.frames.get_file_path(frame_num)
            type(self).rgb_image = dlib.load_rgb_image(str(image_file_path))
            if self.rgb_image is not None:
                print('Frame ', frame_num, ' extracting faces')
                faces = self.detector(self.rgb_image, 1)
                if len(faces) > 0:
                    print('Frame ', frame_num, ' face ', face_num, ' extracting landmarks')
                    type(self).frame_num = frame_num
                    type(self).shape = self.predictor(self.rgb_image, faces[face_num])
        return self.shape

    def get_lmarks(self, frame_num=30, face_num=0):
        """ load image and attempt to extract landmark """
        shape = self.get_shape(frame_num, face_num)
        if shape is None:
            return np.full((1, 68, 2), np.nan)
        return np.array([(part.x, part.y) for part in shape.parts()]).reshape((1, 68, 2))

    def get_all_lmarks(self, new_extract=False, lmarks_file=None):
        """ Get landmarks from face for all frames as ndarray """
        if lmarks_file is None:
            lmarks_file = self.lmarks_file
        if not new_extract and self.lmarks_file.is_file():
            type(self).lmarks = np.load(self.lmarks_file)
            return self.lmarks
        if not self.frames.get_frame_nums():
            Video(self.frames).extract_frames(self.video_file)
        for frame_num in self.frames.get_frame_nums():
            type(self).lmarks = np.concatenate([self.lmarks, self.get_lmarks(frame_num)])
        np.save(self.lmarks_file, self.lmarks)
        return self.lmarks

    def display_overlay(self, frame_num=30, face_num=0):
        """ Display image overlayed with landmarks """
        win = dlib.image_window()
        win.clear_overlay()
        self.get_lmarks(frame_num, face_num)
        win.set_image(self.rgb_image)
        if self.shape is not None:
            win.add_overlay(self.shape)
        dlib.hit_enter_to_continue()

class DataProcess:
    """ Calculations and supporting methods required for the replication of experiments """
    dlib_proc = None
    """ [DlibProcess](#replication.DlibProcess) """
    video_file = None
    """ Input video file """

    def __init__(self, video_file=None, dlib_proc=None):
        if dlib_proc is None:
            type(self).dlib_proc = DlibProcess(video_file)
        else:
            type(self).dlib_proc = dlib_proc
        type(self).video_file = self.dlib_proc.video_file

    def get_procrustes(self, lmarks=None, mouth_only=False):
        """ Procrustes analysis - return landmarks best fit to mean landmarks """
        if lmarks is None:
            lmarks = self.dlib_proc.get_all_lmarks()
        if mouth_only:
            lmarks = lmarks[:, 48:, :]
        mean_lmarks = np.nanmean(lmarks, 0, keepdims=True)
        proc_lmarks = np.full(lmarks.shape, np.nan)
        for frame_num in range(lmarks.shape[0]):
            if ~np.isnan(lmarks[frame_num, 0, 0]):
                _, proc_lmarks[frame_num], _ = procrustes(
                    mean_lmarks[0], lmarks[frame_num])
        if mouth_only:
            not_lips = np.full((proc_lmarks.shape[0], proc_lmarks.shape[1],
                                48, proc_lmarks.shape[3]), np.nan)
            proc_lmarks = np.concatenate((not_lips, proc_lmarks), 2)
        return proc_lmarks

    def interpolate_lmarks(self, lmarks=None, old_rate=30, new_rate=25):
        """ Change the frame rate of the extracted landmarks using linear
            interpolation """
        if lmarks is None:
            lmarks = self.get_procrustes()
        old_frame_axis = np.arange(lmarks.shape[0])
        new_frame_axis = np.linspace(0, lmarks.shape[0]-1, int(
            lmarks.shape[0]*new_rate/old_rate))
        new_lmarks = np.zeros((len(new_frame_axis),) + (lmarks.shape[1:]))
        for ax1 in range(lmarks.shape[1]):
            for ax2 in range(lmarks.shape[2]):
                new_lmarks[:, ax1, ax2] = np.interp(new_frame_axis, old_frame_axis,
                                                    lmarks[:, ax1, ax2])
        return new_lmarks

    def get_closed_mouth_frame(self, lmarks=None, zscore=1.3):
        """ Determine frame with the minimum distance between the inner lips
            excluding frames where the mouth is unusually wide or narrow """
        if lmarks is None:
            lmarks = self.get_procrustes()
        lip_r = 60
        lip_l = 64
        mouth_width = np.linalg.norm(lmarks[:, lip_r] - lmarks[:, lip_l], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            lmarks_filtered = np.nonzero(np.abs(stats.zscore(
                mouth_width, nan_policy='omit')) < zscore)
        lip_top = slice(61, 64)
        lip_bottom = slice(65, 68)
        lip_dist = np.linalg.norm(lmarks[lmarks_filtered, lip_top] - lmarks[
            lmarks_filtered, lip_bottom], axis=2)
        return lmarks_filtered[0][np.argmin(np.sum(lip_dist, -1)[0])]

    def remove_identity(self, lmarks=None, template_file=None, id_removed_file=None,
                        zscore=0.1):
        """ current frame - the closed mouth frame + template """
        if lmarks is None:
            lmarks = self.get_procrustes()
        if template_file is None:
            template_file = Path('..', 'data', 'mean.npy')
        lmarks = self.interpolate_lmarks().reshape((-1, 68, 2))
        closed_mouth = lmarks[self.get_closed_mouth_frame(lmarks=lmarks, zscore=zscore)]
        template_2d = np.load(str(template_file))[:, :2]
        identity_removed = lmarks - closed_mouth + template_2d
        if id_removed_file is not None:
            Path(id_removed_file).parent.mkdir(parents=True, exist_ok=True)
            np.save(str(Path(id_removed_file)), identity_removed)
        return identity_removed

class Draw:
    """ Draw landmarks with matplotlib """
    data_proc = None
    """ [DataProcess](#replication.DataProcess) """
    plots_dir = None
    """ Generated plots directory """
    frames = None
    """ [Frames](#replication.Frames) """
    dimensions = {'width': 500, 'height': 500}
    axes = None
    bounds = {}

    def __init__(self, plots_dir=None, data_proc=None, dimensions=None):
        if data_proc is None:
            type(self).data_proc = DataProcess()
        else:
            type(self).data_proc = data_proc
        type(self).frames = self.data_proc.dlib_proc.frames
        if plots_dir is None:
            type(self).plots_dir = Path(self.frames.root_dir, 'plots')
        else:
            type(self).plots_dir = Path(plots_dir)
        if dimensions is not None:
            type(self).dimensions = dimensions

    def calc_mean(self, lmarks):
        """ Calculalate mean points in landmark set """
        type(self).bounds = {'mid': np.nanmean(lmarks, 0),
                             'xmid': np.nanmean(lmarks[..., 0]),
                             'ymid': np.nanmean(lmarks[..., 1])}

    def _plot_features(self, lmarks, frame_num=0):
        """ Calculate and plot facial features """
        features = {'jaw': lmarks[frame_num, :17],
                    'eyebrow_r': lmarks[frame_num, 17:22],
                    'eyebrow_l': lmarks[frame_num, 22:27],
                    'nose_top': lmarks[frame_num, 27:31],
                    'nose_side_r': np.concatenate((lmarks[frame_num, 27:28],
                                                   lmarks[frame_num, 31:32])),
                    'nose_side_l': np.concatenate((lmarks[frame_num, 27:28],
                                                   lmarks[frame_num, 35:36])),
                    'nose_mid_r': lmarks[frame_num, 30:32],
                    'nose_mid_l': np.concatenate((lmarks[frame_num, 30:31],
                                                  lmarks[frame_num, 35:36])),
                    'nose_bottom': lmarks[frame_num, 31:36],
                    'eye_r': np.concatenate((lmarks[frame_num, 36:42],
                                             lmarks[frame_num, 36:37])),
                    'eye_l': np.concatenate((lmarks[frame_num, 42:48],
                                             lmarks[frame_num, 42:43])),
                    'lips_out': np.concatenate((lmarks[frame_num, 48:60],
                                                lmarks[frame_num, 48:49])),
                    'lips_in': np.concatenate((lmarks[frame_num, 60:],
                                               lmarks[frame_num, 60:61]))}

        self.axes.plot(features['jaw'][:, 0], features['jaw'][:, 1], 'b-')
        self.axes.plot(features['eyebrow_r'][:, 0], features['eyebrow_r'][:, 1], 'b-')
        self.axes.plot(features['eyebrow_l'][:, 0], features['eyebrow_l'][:, 1], 'b-')
        self.axes.plot(features['nose_top'][:, 0], features['nose_top'][:, 1], 'b-')
        self.axes.plot(features['nose_side_r'][:, 0], features['nose_side_r'][:, 1], 'b-')
        self.axes.plot(features['nose_side_l'][:, 0], features['nose_side_l'][:, 1], 'b-')
        self.axes.plot(features['nose_mid_r'][:, 0], features['nose_mid_r'][:, 1], 'b-')
        self.axes.plot(features['nose_mid_l'][:, 0], features['nose_mid_l'][:, 1], 'b-')
        self.axes.plot(features['nose_bottom'][:, 0], features['nose_bottom'][:, 1], 'b-')
        self.axes.plot(features['eye_r'][:, 0], features['eye_r'][:, 1], 'b-')
        self.axes.plot(features['eye_l'][:, 0], features['eye_l'][:, 1], 'b-')
        self.axes.plot(features['lips_out'][:, 0], features['lips_out'][:, 1], 'b-')
        self.axes.plot(features['lips_in'][:, 0], features['lips_in'][:, 1], 'b-')

    def save_scatter(self, frame_num_sel=None, with_frame=True, dpi=96,
                     annot=False):
        """ Plot landmarks and save """
        _, type(self).axes = plt.subplots(figsize=(self.dimensions['width']/dpi,
                                                   self.dimensions['height']/dpi), dpi=dpi)
        lmarks = self.data_proc.dlib_proc.get_all_lmarks()
        if self.plots_dir.is_dir():
            shutil.rmtree(self.plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        if frame_num_sel is None:
            for frame_num in range(lmarks.shape[0]):
                self.save_scatter_frame(frame_num, lmarks, with_frame, annot=annot)
        else:
            self.save_scatter_frame(frame_num_sel, with_frame=with_frame,
                                    annot=annot)

    def save_scatter_frame(self, frame_num=30, lmarks=None, with_frame=True,
                           annot=False):
        """ Plot landmarks and save frame """
        self.axes.clear()
        if lmarks is None:
            lmarks = self.data_proc.dlib_proc.get_all_lmarks()
        if with_frame:
            image = plt.imread(self.frames.get_file_path(frame_num))
            self.axes.imshow(image)
        self.calc_mean(lmarks)
        frame_left = self.bounds['xmid'] - self.dimensions['width']/2
        frame_right = self.bounds['xmid'] + self.dimensions['width']/2
        frame_bottom = self.bounds['ymid'] - self.dimensions['height']/2
        frame_top = self.bounds['ymid'] + self.dimensions['height']/2
        self.axes.set_xlim(frame_left, frame_right)
        self.axes.set_ylim(frame_bottom, frame_top)
        self.axes.invert_yaxis()
        self.axes.scatter(lmarks[frame_num, :, 0],
                          lmarks[frame_num, :, 1], marker='.')
        if annot:
            self.axes.annotate('Frame: ' + str(frame_num), xy=(
                frame_left + 10, frame_top - 10), color='cyan')
            for lmark_num, (point_x, point_y) in enumerate(
                    lmarks[frame_num]):
                self.axes.annotate(str(lmark_num+1), xy=(point_x, point_y))
        plt.savefig(Path(self.plots_dir, str(frame_num).zfill(self.frames.num_len) + '.png'))

    def save_plots(self, lmarks=None, with_frame=True, annot=False, dpi=96):
        """ Save line plots """
        _, self.axes = plt.subplots(figsize=(self.dimensions['width']/dpi,
                                             self.dimensions['height']/dpi), dpi=dpi)
        if lmarks is None:
            lmarks = self.data_proc.dlib_proc.get_all_lmarks()
        if self.plots_dir.is_dir():
            shutil.rmtree(self.plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        for frame_num in range(lmarks.shape[0]):
            self.axes.clear()
            if with_frame:
                image = plt.imread(self.frames.get_file_path(frame_num))
                self.axes.imshow(image)

            self._plot_features(lmarks, frame_num)
            self.axes.set_xlim(type(self).bounds['xmid'] - (self.dimensions['width']/2),
                               type(self).bounds['xmid'] + (self.dimensions['width']/2))
            self.axes.set_ylim(type(self).bounds['ymid'] - (self.dimensions['height']/2),
                               type(self).bounds['ymid'] + (self.dimensions['height']/2))
            self.axes.invert_yaxis()
            if annot:
                self.annotate(frame_num, lmarks)
            plt.savefig(Path(self.plots_dir, str(frame_num).zfill(
                self.frames.num_len) + '.png'))

    def annotate(self, frame_num, lmarks):
        """ Annote image with landmark and frame numbers """
        self.axes.annotate('Frame: ' + str(frame_num), xy=(
            self.axes.get_xlim()[0] + 0.01, self.axes.get_ylim(
                )[0] - 0.01), color='blue')
        for lmark_num, (point_x, point_y) in enumerate(
                lmarks[frame_num]):
            self.axes.annotate(str(lmark_num+1), xy=(point_x, point_y))

    def save_plots_proc(self, dpi=96, annot=False, mouth_only=False):
        """ Save line plots with Procrustes analysis """
        _, self.axes = plt.subplots(figsize=(
            self.dimensions['width']/dpi, self.dimensions['height']/dpi), dpi=dpi)
        lmarks = self.data_proc.get_procrustes(mouth_only=mouth_only)
        if self.plots_dir.is_dir():
            shutil.rmtree(self.plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        for frame_num in range(lmarks.shape[0]):
            self.axes.clear()
            self.axes.set_aspect(1)
            self._plot_features(lmarks, frame_num)
            self.axes.invert_yaxis()
            if annot:
                self.annotate(frame_num, lmarks)
            plt.savefig(Path(self.plots_dir, str(frame_num).zfill(
                self.frames.num_len) + '.png'))

class Video:
    """ FFmpeg video processing manager """
    frames = None
    """ [Frames](#replication.Frames) """
    root_dir = None
    """ Toolkit working directory """

    def __init__(self, frames=None, root_dir=None):
        if frames is None:
            type(self).frames = Frames()
        else:
            type(self).frames = frames
        if root_dir is None:
            type(self).root_dir = self.frames.root_dir
        else:
            type(self).root_dir = Path(root_dir)

    def extract_audio(self, video_in=None, audio_file=None):
        """ Extract audio from video sample """
        if video_in is None:
            video_in = Path(self.root_dir, 'shared', 'obama2s.mp4')
        if audio_file is None:
            audio_file = Path(self.root_dir, 'audio', Path(video_in).stem,
                              Path(video_in).with_suffix('.wav').name)
        Path(audio_file).parent.mkdir(parents=True, exist_ok=True)
        sp.run(['ffmpeg', '-i', str(video_in), '-y',
                str(audio_file)], check=True)
        return Path(audio_file)

    def extract_frames(self, video_in=None, start_number=0, quality=5):
        """ Extract frames from video using FFmpeg """
        if video_in is None:
            video_in = Path(self.root_dir, 'shared', 'obama2s.mp4')
        frames_dir = self.frames.frames_dir
        if frames_dir.is_dir():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        sp.run(['ffmpeg', '-i', str(video_in),
                '-start_number', str(start_number), '-qscale:v', str(quality),
                str(Path(frames_dir, r'%0' + str(
                    self.frames.num_len) + 'd' + self.frames.suffix))], check=True)

    def create_video(self, video_out=None, plots_dir=None, framerate=25,
                     frame_text='frame %{frame_num} %{pts}'):
        """ Create video from images """
        if video_out is None:
            video_out = Path(self.root_dir, 'video', 'plots.mp4')
        Path(video_out).parent.mkdir(parents=True, exist_ok=True)
        if plots_dir is None:
            plots_dir = Path(self.root_dir, 'plots')
        sp.run(['ffmpeg', '-y', '-f', 'image2', '-framerate', str(framerate), '-i',
                str(Path(plots_dir, r'%0' + str(self.frames.num_len) + 'd.png')), '-vf',
                'drawtext=text=\'' + frame_text + '\':fontsize=20:x=10:y=10',
                str(video_out)], check=True)

    def stack_h(self, video_left=None, video_right=None, video_out=None):
        """ Stack videos horizontally """
        if video_left is None:
            video_left = Path(self.root_dir, 'shared',
                              'obama2s', 'obama2s_painted_t.mp4')
        if video_right is None:
            video_right = Path(self.root_dir, 'shared', 'identity_removed',
                               'obama2s.ir_painted_t.mp4')
        if video_out is None:
            video_out = Path(self.root_dir, 'video',
                             Path(Path(video_left).name + 'comp_h.mp4'))
        Path(video_out).parent.mkdir(parents=True, exist_ok=True)

        sp.run(['ffmpeg', '-i', str(video_left), '-i',
                str(video_right), '-filter_complex',
                'hstack=inputs=2', '-y',
                str(video_out)], check=True)

    def stack_v(self, video_top=None, video_bottom=None, video_out=None):
        """ Stack videos vertically """
        if video_top is None:
            video_top = Path(self.root_dir, 'shared',
                             'obama2s', 'obama2s_painted_t.mp4')
        if video_bottom is None:
            video_bottom = Path(self.root_dir, 'shared', 'identity_removed',
                                'obama2s.ir_painted_t.mp4')
        if video_out is None:
            video_out = Path(self.root_dir, 'video',
                             Path(Path(video_top).name + 'comp_v.mp4'))
        Path(video_out).parent.mkdir(parents=True, exist_ok=True)

        sp.run(['ffmpeg', '-i', str(video_top), '-i',
                str(video_bottom), '-filter_complex',
                'vstack=inputs=2', '-y',
                str(video_out)], check=True)

    def draw_text(self, video_in=None, video_out=None, frame_text='frame %{frame_num} %{pts}'):
        """ Add text to video frames """
        if video_in is None:
            video_in = Path(self.root_dir, 'shared', 'obama2s.mp4')
        if video_out is None:
            video_out = Path(self.root_dir, 'video', Path(Path(video_in).name + 't.mp4'))
        Path(video_out).parent.mkdir(parents=True, exist_ok=True)

        sp.run(['ffmpeg', '-y', '-i', str(video_in), '-vf',
                'drawtext=text=\'' + frame_text + '\':fontsize=20:x=10:y=10',
                str(video_out)], check=True)

    def prepare_ground_truth(self, video_in=None, video_out=None,
                             frame_text='frame %{frame_num} %{pts}'):
        """ Adjust the framerate to 25fps, crop and add text to the source video """
        if video_in is None:
            video_in = Path(self.root_dir, 'shared', '080815_WeeklyAddress.mp4')
        if video_out is None:
            video_out = Path(self.root_dir, 'video',
                             Path(Path(video_in).name + '_25t.mp4'))
        Path(video_out).parent.mkdir(parents=True, exist_ok=True)

        sp.run(['ffmpeg', '-y', '-i', str(video_in), '-vf',
                'fps=25, drawtext=text=\'' + frame_text + '\':fontsize=20'
                ':x=810:y=260,crop=500:500:800:250',
                str(video_out)], check=True)

    def prepare_anims(self, video_in=None, video_out=None, frame_text='frame %{frame_num} %{pts}'):
        """ Scale down, crop and add text to the animations """
        if video_in is None:
            video_in = Path(self.root_dir, 'video', '080815_WeeklyAddress_painted_.mp4')
        if video_out is None:
            video_out = Path(self.root_dir, 'video', Path(Path(video_in).name + 't.mp4'))
        Path(video_out).parent.mkdir(parents=True, exist_ok=True)

        sp.run(['ffmpeg', '-y', '-i', str(video_in), '-vf',
                'scale=500:500,drawtext=text=\'' + frame_text + '\':fontsize=20:x=10:y=10',
                str(video_out)], check=True)

class Analysis:
    """ Data extraction and analysis """
    data_proc = None
    """ [DataProcess](#replication.DataProcess) """
    video = None
    """ [Video](#replication.Video) """
    root_dir = None
    """ Toolkit working directory """

    def __init__(self, video=None):
        if video is None:
            type(self).video = Video()
        else:
            type(self).video = video
        type(self).root_dir = self.video.root_dir

    def calc_rmse(self, video_file=None, python_exe='python', mouth_only=True):
        """ Extract audio from video and use the pre-trained model to predict landmarks
        Extract landmarks from video, preprocess and calculate the root mean square error """
        if video_file is None:
            type(self).data_proc = DataProcess()
        else:
            type(self).data_proc = DataProcess(video_file)
        video_file = self.data_proc.dlib_proc.video_file
        audio_file = self.video.extract_audio(video_file)
        sp.run([python_exe, 'generate.py', '-i', str(audio_file.parent),
                '-m', '../pre_trained/1D_CNN.pt', '-o',
                str(Path(self.root_dir, 'pred_out')), '-s'], check=True)
        pred_lmarks = np.load(str(Path(self.root_dir, 'pred_out', audio_file.name,
                                       'predicted.npy')))
        lmarks = self.data_proc.dlib_proc.get_all_lmarks()
        lmarks_ir = self.data_proc.remove_identity(lmarks)
        pred_lmarks_b = pred_lmarks[:lmarks_ir.shape[0], :, :lmarks_ir.shape[2]]
        if mouth_only:
            lmarks_ir = lmarks_ir[:, 48:]
            pred_lmarks_b = pred_lmarks_b[:, 48:]
        return np.mean(((pred_lmarks_b - lmarks_ir)**2)[..., 0] + ((
            pred_lmarks_b - lmarks_ir)**2)[..., 1], 1)**0.5
