""" Replication toolkit """
import glob
import os
from pathlib import Path
import shutil
import urllib.request
import bz2
import subprocess as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from scipy import stats
import dlib

class Frames:
    """ Frame file manager """
    def __init__(self, frame_dir=os.path.join('..', 'replic', 'frames'),
                 extension='jpeg', num_len=4):
        self.frame_dir = frame_dir
        Path(frame_dir).mkdir(parents=True, exist_ok=True)
        self.extension = extension
        self.num_len = num_len
        self.frame_num = None
        self.file_name = None

    def get_file_path(self, frame_num=30):
        """ Build file path from frame number """
        self.frame_num = frame_num
        self.file_name = (str(frame_num).zfill(self.num_len) + '.' + self.extension)
        return os.path.join(self.frame_dir, self.file_name)

    def get_frame_num(self, file_name=os.path.join('..', 'replic', 'frames', '0030.jpeg')):
        """ Derive frame number from image file name """
        self.file_name = file_name
        self.frame_num = int(os.path.splitext(os.path.split(file_name)[1])[0])
        return self.frame_num

    def get_frame_file_names(self):
        """ Get list of frame files """
        return sorted(glob.glob(os.path.join(self.frame_dir, '*.' +
                                             self.extension)))
    def get_frame_nums(self):
        """ Get list of frame numbers """
        frames = self.get_frame_file_names()
        return [self.get_frame_num(frame) for frame in frames]

    def count_frames(self):
        """ count number of frames in directory """
        return len(self.get_frame_nums())

class DlibProcess:
    """ Extract landmarks from frames using Dlib """
    def __init__(self, rgb_image=None, model_dir=os.path.join('..', 'data'),
                 model_url='https://raw.github.com/davisking/dlib-models/master/'
                 'shape_predictor_68_face_landmarks.dat.bz2'):
        self.detector = dlib.get_frontal_face_detector()
        self.rgb_image = rgb_image
        self.frame_num = None
        self.faces = None
        self.shape = None
        self.model_file = os.path.join(model_dir, os.path.splitext(
            os.path.split(model_url)[1])[0])
        if not os.path.isfile(self.model_file):
            print('Model ' + self.model_file + ' not found')
            print('Downloading from ' + model_url)
            with urllib.request.urlopen(model_url) as response, open(
                    self.model_file, 'wb') as model:
                model.write(bz2.decompress(response.read()))
        self.predictor = dlib.shape_predictor(self.model_file)

    def load_image(self, frame_num=30, frames=Frames()):
        """ load image and attempt to extract faces """
        self.faces = None
        self.shape = None
        image_file_path = frames.get_file_path(frame_num)
        self.rgb_image = dlib.load_rgb_image(image_file_path)
        if self.rgb_image is not None:
            self.frame_num = frame_num
            print('Frame ', frame_num, ' extracting faces')
            self.faces = self.detector(self.rgb_image, 1)

    def get_shape(self, frame_num=30):
        """ Retrieve or extract landmarks from face as dlib.points """
        if self.shape is None or frame_num != self.frame_num:
            self.extract_shape(frame_num)
        return self.shape

    def extract_shape(self, frame_num=30):
        """ Extract landmarks from face as dlib.points """
        if self.faces is None or frame_num != self.frame_num:
            self.load_image(frame_num)
        if len(self.faces) > 0:
            print('Frame ', frame_num, ' face ', 0, ' extracting landmarks')
            self.shape = self.predictor(self.rgb_image, self.faces[0])

    def get_lmarks(self, frame_num=30):
        """ Get landmarks from face as ndarray """
        if self.get_shape(frame_num) is None:
            return np.full((1, 68, 2), np.nan)
        return np.array([(part.x, part.y) for part in self.shape.parts()]).reshape((1, 68, 2))

    def display_overlay(self, image_file=None, frame_num=30):
        """ Display image overlayed with landmarks """
        win = dlib.image_window()
        win.clear_overlay()
        if image_file is None:
            self.load_image(frame_num)
        else:
            self.rgb_image = dlib.load_rgb_image(image_file)
        win.set_image(self.rgb_image)
        if self.get_shape(frame_num) is not None:
            win.add_overlay(self.shape)
        dlib.hit_enter_to_continue()

class DataProcess:
    """ Plots for landmarks """
    def __init__(self, extract_file=os.path.join('..', 'replic', 'data', 'obama2s.npy')):
        self.extract_file = extract_file
        self.axes = None
        self.all_lmarks = np.empty((0, 68, 2))

    def get_all_lmarks(self, new_extract=False,
                       extract_file=None,
                       dlib_proc=DlibProcess(), frames=Frames()):
        """ Get landmarks from face for all frames as ndarray """
        if extract_file is None:
            extract_file = self.extract_file
        if new_extract:
            self.all_lmarks = None
        elif os.path.exists(extract_file):
            self.all_lmarks = np.load(extract_file)
        if self.all_lmarks.size == 0:
            if not frames.get_frame_nums():
                Video().extract_frames(os.path.splitext(os.path.split(extract_file)[1])[0] + '.mp4')
            for frame_num in frames.get_frame_nums():
                self.all_lmarks = np.concatenate([self.all_lmarks, dlib_proc.get_lmarks(frame_num)])
            Path(os.path.split(extract_file)[0]).mkdir(parents=True, exist_ok=True)
            np.save(extract_file, self.all_lmarks)
        return self.all_lmarks

    def filter_outliers(self, zscore=4, extract_file=None):
        """ replace outliers greater than specified zscore with np.nan) """
        if extract_file is None:
            extract_file = self.extract_file
        lmarks = self.get_all_lmarks(extract_file=extract_file)
        lmarks_zscore = stats.zscore(lmarks, nan_policy='omit')
        with np.errstate(invalid='ignore'):
            lmarks[np.any(lmarks_zscore > zscore, (1, 2, 3))] = np.nan
        return lmarks

    def get_procrustes(self, extract_file=None, lips_only=False):
        """ Procrustes analysis - return landmarks best fit to mean landmarks """
        if extract_file is None:
            extract_file = self.extract_file
        lmarks = self.get_all_lmarks(extract_file=extract_file)
        if lips_only:
            lmarks = lmarks[:, 48:, :]
        mean_lmarks = np.nanmean(lmarks, 0, keepdims=True)
        proc_lmarks = np.full(lmarks.shape, np.nan)
        for frame_num in range(lmarks.shape[0]):
            if ~np.isnan(lmarks[frame_num, 0, 0]):
                _, proc_lmarks[frame_num], _ = procrustes(
                    mean_lmarks[0], lmarks[frame_num])
        if lips_only:
            not_lips = np.full((proc_lmarks.shape[0], proc_lmarks.shape[1],
                                48, proc_lmarks.shape[3]), np.nan)
            proc_lmarks = np.concatenate((not_lips, proc_lmarks), 2)
        return proc_lmarks

    def interpolate_lmarks(self, extract_file=None, old_rate=30, new_rate=25):
        """ Change the frame rate of the extracted landmarks using linear interpolation """
        if extract_file is None:
            extract_file = self.extract_file
        lmarks = self.get_procrustes(extract_file=extract_file)
        old_frame_axis = np.arange(lmarks.shape[0])
        new_frame_axis = np.linspace(0, lmarks.shape[0]-1, int(lmarks.shape[0]*new_rate/old_rate))
        new_lmarks = np.zeros((len(new_frame_axis),) + (lmarks.shape[1:]))
        for ax1 in range(lmarks.shape[1]):
            for ax2 in range(lmarks.shape[2]):
                new_lmarks[:, ax1, ax2] = np.interp(new_frame_axis, old_frame_axis,
                                                    lmarks[:, ax1, ax2])
        return new_lmarks

    def get_closed_mouth_frame(self, extract_file=None, lmarks=None):
        """ Determine frame with the minimum distance between the inner lips where
            the horizonal distance is no more the one standard deviation from the mean """
        if extract_file is None:
            extract_file = self.extract_file
        if lmarks is None:
            lmarks = self.get_procrustes(extract_file=extract_file)
        lip_top = lmarks[:, 61:64]
        lip_bottom = lmarks[:, 65:68]
        lip_left = lmarks[:, 60:61]
        lip_right = lmarks[:, 64:65]
        diff_squared_vert = (lip_top - lip_bottom)**2
        diff_squared_hori = (lip_left - lip_right)**2
        diff_vert_total = (diff_squared_vert[:, 0] + diff_squared_vert[:, 1])
        return np.nanargmin(np.sum(diff_vert_total, -1))

    def remove_identity(self, extract_file=None,
                        template=os.path.join('..', 'data', 'mean.npy')):
        """ current frame - the closed mouth frame + template """
        if extract_file is None:
            extract_file = self.extract_file
        lmarks = self.interpolate_lmarks(extract_file=extract_file).reshape((-1, 68, 2))
        closed_mouth = lmarks[self.get_closed_mouth_frame(lmarks=lmarks)]
        template_2d = np.load(template)[:, :2]
        return lmarks - closed_mouth + template_2d

class Draw:
    """ Draw landmarks with matplotlib """
    def __init__(self, plots_dir=os.path.join('..', 'replic', 'plots'),
                 data_proc=DataProcess(), width=500, height=500):
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        self.plots_dir = plots_dir
        self.data_proc = data_proc
        lmarks = data_proc.get_all_lmarks()
        self.axes = None
        self.bounds = {'width': width, 'height': height,
                       'mid': np.nanmean(lmarks, 0),
                       'xmid': np.nanmean(lmarks[..., 0]),
                       'ymid': np.nanmean(lmarks[..., 1])}

    def _plot_features(self, lmarks, frame_num=0):
        """ calculate and plot facial features """
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

    def save_scatter(self, frame_num_sel=None, with_frame=True, dpi=96, annot=False):
        """ Plot landmarks and save """
        _, self.axes = plt.subplots(figsize=(self.bounds['width']/dpi,
                                             self.bounds['height']/dpi), dpi=dpi)
        lmarks = self.data_proc.get_all_lmarks()
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
            lmarks = self.data_proc.get_all_lmarks()
        if with_frame:
            image = plt.imread(Frames().get_file_path(frame_num))
            self.axes.imshow(image)
        frame_left = self.bounds['xmid'] - self.bounds['width']/2
        frame_right = self.bounds['xmid'] + self.bounds['width']/2
        frame_bottom = self.bounds['ymid'] - self.bounds['height']/2
        frame_top = self.bounds['ymid'] + self.bounds['height']/2
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
        plt.savefig(os.path.join(self.plots_dir, str(frame_num) + '.png'))

    def save_plots(self, with_frame=True, annot=False, dpi=96):
        """ save line plots """
        _, self.axes = plt.subplots(figsize=(self.bounds['width']/dpi,
                                             self.bounds['height']/dpi), dpi=dpi)
        lmarks = self.data_proc.get_all_lmarks()
        for frame_num in range(lmarks.shape[0]):
            self.axes.clear()
            if with_frame:
                image = plt.imread(Frames().get_file_path(frame_num))
                self.axes.imshow(image)

            self._plot_features(lmarks, frame_num)
            self.axes.set_xlim(self.bounds['xmid'] - (self.bounds['width']/2),
                               self.bounds['xmid'] + (self.bounds['width']/2))
            self.axes.set_ylim(self.bounds['ymid'] - (self.bounds['height']/2),
                               self.bounds['ymid'] + (self.bounds['height']/2))
            self.axes.invert_yaxis()
            if annot:
                self.annotate(frame_num, lmarks)
            plt.savefig(os.path.join(self.plots_dir, str(frame_num) + '.png'))

    def annotate(self, frame_num, lmarks):
        """ Annote image with landmark and frame numbers """
        self.axes.annotate('Frame: ' + str(frame_num), xy=(
            self.axes.get_xlim()[0] + 0.01, self.axes.get_ylim(
                )[0] - 0.01), color='blue')
        for lmark_num, (point_x, point_y) in enumerate(
                lmarks[frame_num]):
            self.axes.annotate(str(lmark_num+1), xy=(point_x, point_y))

    def save_plots_proc(self, dpi=96, annot=False,
                        extract_file=os.path.join('..', 'replic', 'data',
                                                  'obama2s.npy'), lips_only=False):
        """ save line plots with Procrustes analysis """
        _, self.axes = plt.subplots(figsize=(self.bounds['width']/dpi,
                                             self.bounds['height']/dpi), dpi=dpi)
        lmarks = self.data_proc.get_procrustes(
            extract_file=extract_file, lips_only=lips_only)
        if os.path.isdir(self.plots_dir):
            shutil.rmtree(self.plots_dir)
        Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
        for frame_num in range(lmarks.shape[0]):
            self.axes.clear()
            self.axes.set_aspect(1)
            self._plot_features(lmarks, frame_num)
            self.axes.invert_yaxis()
            if annot:
                self.annotate(frame_num, lmarks)
            plt.savefig(os.path.join(self.plots_dir, str(frame_num) + '.png'))

class Video:
    """ Video processing """
    def __init__(self, video_dir=os.path.join('..', 'replic', 'video'),
                 audio_dir=os.path.join('..', 'replic', 'audio'),
                 frames=Frames()):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.frames_dir = frames.frame_dir

    def extract_audio(self, video_in='obama2s.mp4',
                      audio_out=None):
        """ Extract audio from video sample """
        if audio_out is None:
            audio_out = os.path.join(self.audio_dir, os.path.splitext(video_in)[0] + '.wav')
        Path(self.audio_dir).mkdir(parents=True, exist_ok=True)
        sp.run(['ffmpeg', '-i', os.path.join(self.video_dir, video_in), '-y',
                audio_out], check=True)

    def extract_frames(self, video_in='obama2s.mp4', start_number=0, quality=5):
        """ Extract frames from video using FFmpeg """
        if os.path.isdir(self.frames_dir):
            shutil.rmtree(self.frames_dir)
        Path(self.frames_dir).mkdir(parents=True, exist_ok=True)
        sp.run(['ffmpeg', '-i', os.path.join(self.video_dir, video_in), '-start_number',
                str(start_number), '-qscale:v', str(quality),
                os.path.join(self.frames_dir, '%04d.jpeg')], check=True)

    def create_video(self, video_out='plots.mp4', plots_dir=os.path.join('..', 'replic', 'plots'),
                     framerate=30):
        """ create video from images """
        sp.run(['ffmpeg', '-f', 'image2', '-framerate', str(framerate), '-i',
                os.path.join(os.path.join(plots_dir, '%d.png')),
                '-y', os.path.join(os.path.join(self.video_dir, video_out))],
               check=True)

    def combine_h(self, video_left='ob25_painted_.mp4',
                  video_right='obama2s_painted_.mp4',
                  video_out='obama2s_comparison.mp4'):
        """ stack videos horizontally """
        sp.run(['ffmpeg', '-i', os.path.join(self.video_dir, video_left), '-i',
                os.path.join(self.video_dir, video_right), '-filter_complex',
                'hstack=inputs=2', '-y',
                os.path.join(self.video_dir, video_out)], check=True)

    def combine_v(self, video_top='obamac.mp4', video_bottom='combined_h.mp4',
                  video_out='obama_v.mp4'):
        """ stack videos vertically """
        sp.run(['ffmpeg', '-i', os.path.join(self.video_dir, video_top), '-i',
                os.path.join(self.video_dir, video_bottom), '-filter_complex',
                'vstack=inputs=2', '-y',
                os.path.join(self.video_dir, video_out)], check=True)

    def scale(self, video_in='obama2s.mp4', video_out='scale.mp4', width=500,
              height=500):
        """ scale video """
        sp.run(['ffmpeg', '-i', video_in, '-s', str(width) + 'x' + str(height),
                '-c:a', 'copy', '-y',
                os.path.join(self.video_dir, video_out)], check=True)
