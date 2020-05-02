""" Replication toolkit """
import glob
import os
import urllib.request
import bz2
import subprocess as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from scipy import stats
import dlib

class FrameFile:
    """ Frame file manager """
    def __init__(self, frame_dir='../replic/frames',
                 prefix='frame', extension='jpeg', num_begin=-9, num_end=-5):
        self.frame_dir = frame_dir
        self.prefix = prefix
        self.extension = extension
        self.num_begin = num_begin
        self.num_end = num_end
        self.frame_num = None
        self.file_name = None

    def get_file_path(self, frame_num=30):
        """ Build file path from frame number """
        self.frame_num = frame_num
        self.file_name = (self.prefix + str(frame_num).zfill(
            self.num_end - self.num_begin) + '.' + self.extension)
        return os.path.join(self.frame_dir, self.file_name)

    def get_frame_num(self, file_name='frame0030.jpeg'):
        """ Derive frame number from image file name """
        self.file_name = file_name
        self.frame_num = file_name[self.num_begin:self.num_end]
        return self.frame_num

class Frames:
    """ Frames directory manager """
    def __init__(self, frame_dir='../replic/frames', frame_file=None):
        self.frame_dir = frame_dir
        if frame_file is None:
            self.frame_file = FrameFile()

    def get_frames(self):
        """ Get list of frame files """
        return sorted(glob.glob(os.path.join(self.frame_dir, '*.' +
                                             self.frame_file.extension)))

    def get_frame_nums(self):
        """ Get list of frame numbers """
        frames = self.get_frames()
        return {int(frame[self.frame_file.num_begin:self.frame_file.num_end]) for frame in frames}

class Dlib():
    """ Extract landmarks from frames using Dlib """
    def __init__(self, rgb_image=None, model_dir='../data',
                 model_url='https://raw.github.com/davisking/dlib-models/master/'
                 'shape_predictor_68_face_landmarks.dat.bz2'):
        self.detector = dlib.get_frontal_face_detector()
        self.rgb_image = rgb_image
        self.frame_num = None
        self.faces = None
        self.shape = None
        self.model_file = os.path.join(model_dir, os.path.splitext(os.path.split(model_url)[1])[0])
        if not os.path.isfile(self.model_file):
            print('Model ' + self.model_file + ' not found')
            print('Downloading from ' + model_url)
            with urllib.request.urlopen(model_url) as response, open(
                    self.model_file, 'wb') as model:
                model.write(bz2.decompress(response.read()))
        self.predictor = dlib.shape_predictor(self.model_file)

    def load_image(self, frame_num=30):
        """ load image and attempt to extract faces """
        self.faces = None
        self.shape = None
        image_file_path = FrameFile().get_file_path(frame_num)
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

class Plots(Dlib):
    """ Plots for landmarks """
    def __init__(self, plots_dir='plots', width=500, height=500):
        super().__init__(None)
        self.plots_dir = plots_dir
        self.axes = None
        self.all_lmarks = None
        self.bounds = {'width': width, 'height': height,
                       'mid': None, 'xmid': None,
                       'ymid': None}

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

    def get_all_lmarks(self, new_extract=False, extract_file='obama2s.npy'):
        """ Get landmarks from face for all frames as ndarray """
        if not new_extract and os.path.exists(extract_file):
            self.all_lmarks = np.load(extract_file)
        if self.all_lmarks is None:
            for frame_num in Frames().get_frame_nums():
                if self.all_lmarks is None:
                    self.all_lmarks = self.get_lmarks(frame_num)
                else:
                    self.all_lmarks = np.concatenate([self.all_lmarks,
                                                      self.get_lmarks(frame_num)])
            np.save(extract_file, self.all_lmarks)
        self.bounds['mid'] = np.nanmean(self.all_lmarks, 0)
        self.bounds['xmid'] = np.nanmean(self.all_lmarks[:, :, 0])
        self.bounds['ymid'] = np.nanmean(self.all_lmarks[:, :, 1])
        return self.all_lmarks

    def filter_outliers(self, zscore=4, extract_file='obama2s.npy'):
        """ replace outliers greater than specified zscore with np.nan) """
        lmarks = self.get_all_lmarks(extract_file=extract_file)
        lmarks_zscore = stats.zscore(lmarks, nan_policy='omit')
        with np.errstate(invalid='ignore'):
            lmarks[np.any(lmarks_zscore > zscore, (1, 2, 3))] = np.nan
        return lmarks

    def get_procrustes(self, extract_file='obama2s.npy', lips_only=False):
        """ Procrustes analysis - return landmarks best fit to mean landmarks """
        lmarks = self.get_all_lmarks(extract_file=extract_file)
        if lips_only:
            lmarks = lmarks[..., 48:, :]
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

    def interpolate_lmarks(self, extract_file='obama2s.npy', old_rate=30, new_rate=25):
        """ Change the frame rate of the extracted landmarks using linear interpolation """
        lmarks = self.get_procrustes(extract_file=extract_file)
        old_frame_axis = np.arange(lmarks.shape[0])
        new_frame_axis = np.linspace(0, lmarks.shape[0]-1, int(lmarks.shape[0]*new_rate/old_rate))
        new_lmarks = np.zeros((len(new_frame_axis),) + (lmarks.shape[1:]))
        for ax1 in range(lmarks.shape[1]):
            for ax2 in range(lmarks.shape[2]):
                for ax3 in range(lmarks.shape[3]):
                    new_lmarks[:, ax1, ax2, ax3] = np.interp(new_frame_axis, old_frame_axis,
                                                             lmarks[:, ax1, ax2, ax3])
        return new_lmarks

    def get_closed_mouth_frame(self, extract_file='obama2s.npy', lmarks=None):
        """ Determine frame with the minimum distance between the inner lips """
        if lmarks is None:
            lmarks = self.get_procrustes(extract_file=extract_file)
        top_lip = lmarks[..., 61:64, :]
        bottom_lip = lmarks[..., 65:68, :]
        diffsq = (top_lip - bottom_lip)**2
        dist = (diffsq[..., 0] + diffsq[..., 1])**0.5
        return np.nanargmin(np.sum(dist, -1))

    def remove_identity(self, extract_file='obama2s.npy', template='../data/mean.npy'):
        """ current frame - the closed mouth frame + template """
        lmarks = self.interpolate_lmarks(extract_file=extract_file).reshape((-1, 68, 2))
        closed_mouth = lmarks[self.get_closed_mouth_frame(lmarks=lmarks)]
        template_2d = np.load(template)[:, :2]
        return lmarks - closed_mouth + template_2d

    def save_scatter(self, frame_num_sel=None, with_frame=True, dpi=96, annot=False):
        """ Plot landmarks and save """
        _, self.axes = plt.subplots(figsize=(self.bounds['width']/dpi,
                                             self.bounds['height']/dpi), dpi=dpi)
        if frame_num_sel is None:
            lmarks = self.get_all_lmarks()
            for frame_num in range(lmarks.shape[0]):
                self.save_scatter_frame(frame_num, lmarks, with_frame, annot=annot)
        else:
            lmarks = self.get_lmarks(frame_num_sel)
            self.save_scatter_frame(frame_num_sel, lmarks, with_frame, annot=annot)

    def save_scatter_frame(self, frame_num=30, lmarks=None, with_frame=True, annot=False):
        """ Plot landmarks and save frame """
        self.axes.clear()
        if lmarks is None:
            lmarks = self.get_all_lmarks()
        if with_frame:
            image = plt.imread(FrameFile().get_file_path(frame_num))
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

    def save_plots(self, with_frame=True, dpi=96):
        """ save line plots """
        _, self.axes = plt.subplots(figsize=(self.bounds['width']/dpi,
                                             self.bounds['height']/dpi), dpi=dpi)
        lmarks = self.get_all_lmarks()
        for frame_num in range(lmarks.shape[0]):
            self.axes.clear()
            if with_frame:
                image = plt.imread(FrameFile().get_file_path(frame_num))
                self.axes.imshow(image)

            self._plot_features(lmarks, frame_num)
            self.axes.set_xlim(self.bounds['xmid'] - (self.bounds['width']/2),
                               self.bounds['xmid'] + (self.bounds['width']/2))
            self.axes.set_ylim(self.bounds['ymid'] - (self.bounds['height']/2),
                               self.bounds['ymid'] + (self.bounds['height']/2))
            self.axes.invert_yaxis()
            plt.savefig(os.path.join(self.plots_dir, str(frame_num) + '.png'))

    def save_plots_proc(self, dpi=96, annot=False, extract_file='obama2s.npy',
                        lips_only=False):
        """ save line plots with Procrustes analysis """
        _, self.axes = plt.subplots(figsize=(self.bounds['width']/dpi,
                                             self.bounds['height']/dpi), dpi=dpi)
        lmarks = self.get_procrustes(
            extract_file=extract_file, lips_only=lips_only)
        for frame_num in range(lmarks.shape[0]):
            self.axes.clear()
            self.axes.set_aspect(1)
            self._plot_features(lmarks, frame_num)
            self.axes.invert_yaxis()
            if annot:
                self.axes.annotate('Frame: ' + str(frame_num), xy=(
                    self.axes.get_xlim()[0] + 0.01, self.axes.get_ylim(
                        )[0] - 0.01), color='blue')
                for lmark_num, (point_x, point_y) in enumerate(
                        lmarks[frame_num]):
                    self.axes.annotate(str(lmark_num+1), xy=(point_x, point_y))
            plt.savefig(os.path.join(self.plots_dir, str(frame_num) + '.png'))

class Video(Plots):
    """ Video processing """
    def __init__(self, video_dir='../replic/video_in', audio_dir='../replic/audio_in',
                 frames_dir='../replic/frames'):
        super().__init__()
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.frame_dir = frames_dir

    def extract_audio(self, video_in='obama2s.mp4',
                      audio_out=None):
        """ Extract audio from video sample """
        if audio_out is None:
            audio_out = os.path.join(self.audio_dir, os.path.splitext(video_in)[0] + '.wav')
        sp.run(['ffmpeg', '-i', os.path.join(self.video_dir, video_in), '-y',
                audio_out], check=True)

    def extract_frames(self, video_in='obama2s.mp4', start_number=0, quality=5):
        """ Extract frames from video using FFmpeg """
        sp.run(['ffmpeg', '-i', os.path.join(self.video_dir, video_in), '-start_number',
                str(start_number), '-qscale:v', str(quality),
                os.path.join(self.frame_dir, 'frame%04d.jpeg')], check=True)

    def crop(self, video_in='obama2s.mp4', video_out='obama_crop.mp4',
             crop_param=None):
        """ crop video """
        if crop_param is None:
            plots = Plots()
            plots.get_all_lmarks()
            crop_param = str(plots.bounds['width']) + ':' + str(
                plots.bounds['height']) + ':' +  str(
                    round(plots.bounds['xmid'] - plots.bounds['width']/2)) + ':' +  str(
                        round(plots.bounds['ymid'] - plots.bounds['height']/2))
        sp.run(['ffmpeg', '-i', video_in, '-filter:v',
                'crop=' + crop_param, '-y',
                os.path.join(self.video_dir, video_out)], check=True)

    def create_video(self, video_out='plots.mp4', framerate=30):
        """ create video from images """
        sp.run(['ffmpeg', '-f', 'image2', '-framerate', str(framerate), '-i',
                os.path.join(os.path.join(self.plots_dir, '%d.png')),
                '-y', os.path.join(os.path.join(self.video_dir, video_out))],
               check=True)

    def combine_h(self, video_left='obama_paint.mp4', video_right='plots.mp4',
                  video_out='obama_h.mp4'):
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
