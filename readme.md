# Landmark Generation From Speech

> Replicability and reproducibility, after all, are central to the creation and testing of theories and their acceptance by scientific communities (Berg, 2004)

This project is an attempt to replicate some of the results from Eskimez et al's paper [Noise-Resilient Training Method for Face Landmark Generation From Speech](https://ieeexplore.ieee.org/document/8871109).

The majority of the files are identical to [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface) from which this project is forked. Apart from this readme, the two most important enhancements and some testing utilities are described below -

## code/generate.py

The [original generate script](https://github.com/eeskimez/noise_resilient_3dtface/blob/master/code/generate.py) reads audio files, infers the facial landmarks and generates animated faces. The [enhanced generate script](code/generate.py) allows the user to save the predicted landmarks to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file. It also extends the animation functionality to accept an arbitrary externally created file of landmarks. These changes are limited to enhancements which could not be put into a separate script and were carefully inserted in order to minimise the possibility of disturbing the original functionality. In addition, some redundant code was removed as a result of running [Pylint](https://www.pylint.org/) on the script. To view the precise changes, execute `git diff 3c804caff3e4e0cabd7259ccb97c4038b509d630 code/generate.py`

### Extended Functionality

In addition to the command-line options specified in [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), the [enhanced generate script](code/generate.py) includes the following -

* `-s --save_prediction` save the predicted landmarks and speech array in the folder specified by the `-o` option and disable generation of animation
* `-l --load_prediction` load predictions from the folder specified by the `-i` option and generate a painted face animation in the folder specified by the `-o` option. This option expects the input folder to contain pairs of files with the same name but different extensions - `.wav` and `.npy`

#### Examples

Save the landmarks predicted and speech vector using the [ID_CNN](pre_trained/1D_CNN.pt) model from audio in `replic/samples/obama2s/` to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file in `replic/pred_out/`

    python generate.py -i ../replic/samples/obama2s/ -m ../pre_trained/1D_CNN.pt -o ../replic/pred_out/ -s  

Load landmarks from an external files in `replic/samples/identity_removed/` and generate animation in `replic/anim_out/`

    python generate.py -i ../replic/samples/identity_removed/ -m ../pre_trained/1D_CNN.pt -o ../replic/anim_out/ -l

## code/replication.py

Eskimez et al. released pre-trained models and the generation script with the express aim of promoting scientific reproducibility; however the tools for achieving this were not included. The [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py) is an attempt to fill this gap.
### Differences and Interpretations of the original paper
#### Face Landmark Identity Removal
Eskimez et al. noted that even after applying [procrustes analysis](https://link.springer.com/article/10.1007/BF02291478), there was still a significant correlation between the aligned landmarks and the facial characteristics of the individual speaker. The goal is to minimise these so that the neural network can learn the relationship between speech and facial landmarks regardless of the speaker's identity. Eskimez et al. select a reference frame with a closed mouth and then calculate the differences between this and the corresponding landmarks resulting from procrustes analysis. This reference is calculated based on the distance between the upper and lower lip coordinates.  However this simple calculation can result in reference frames which are far from average such as a frame where the speaker has [pursed lips](replic/samples/obpursed.jpg)

In order to avoid this problem, the calculation was [enhanced](code/replication.py#L179) to exclude frames where the mouth is unusually wide or narrow.

Next a template face is calculated based the average of all the closed mouth references measured across all identities. The differences between each frame and the selected reference frame for the particular speaker are calculated and added to the template face.

### class Video(video_dir, audio_dir, frames):
Manages frame extraction and video manipulation using [FFmpeg](https://www.ffmpeg.org/)
#### Example usage : extract_frames(video_in, start_number, quality)
Extract frames from `replic/samples/obama2s.mp4` into `replic/frames/`

    python -c "from replication import *; Video('../replic/samples/', frames=Frames('../replic/frames/')).extract_frames('obama2s.mp4')"
#### Example usage : extract_audio(video_in, audio_out)
Extract audio in WAV format from video `replic/samples/obama2s.mp4` to `replic/audio_out/obama2s.wav`

    python -c "from replication import *; Video('../replic/samples/', '../replic/audio_out/').extract_audio('obama2s.mp4')"
#### Example usage : draw_text(video_in, video_out, frame_text)
Add the frame number and timestamp to video `replic/identity_removed/obama2s.ir_painted_.mp4` and save as `replic/anim_out/obama2s.ir_painted_t.mp4`

    python -c "from replication import *; Video('../replic/').draw_text('samples/identity_removed/obama2s.ir_painted_.mp4', 'anim_out/obama2s.ir_painted_t.mp4')"
#### Example usage : stack_h(video_left, video_right, video_out)
Stack input videos `replic/samples/obama2s/obama2s_painted_t.mp4` and `replic/samples/identity_removed/obama2s.ir_painted_t.mp4` horizontally and for easier visual comparision into `replic/anim_out/obama2s_comparison.mp4`

    python -c "from replication import *; Video('../replic/samples/').stack_h('obama2s/obama2s_painted_t.mp4', 'identity_removed/obama2s.ir_painted_t.mp4', '../anim_out/obama2s_comparison.mp4')"
### class Frames(frames_dir, video, suffix, num_len):
Helper class used to manage a folder of frames extracted from source video. Each frame is jpeg file named according to the frame number.
#### Example usage : get_frame_nums()
Get frame numbers from `replic/frames/`

    python -c "from replication import *; print(Frames('../replic/frames').get_frame_nums())"
### class DlibProcess(model_dir, model_url)
Manages the extraction of landmarks from individual frames using the [Dlib toolkit](http://dlib.net/)
#### Example usage : display_overlay(frame_num)
Extract landmarks from Frame 30 and overlay the frame image with corresponding line plots

    python -c "from replication import *; DlibProcess().display_overlay(frame_num=30)"
### class DataProcess(data_dir, extract_file, frames):
Calculations and supporting methods required for the replication of experiments
#### Example usage : get_closed_mouth_frame(lmarks, zscore)
First calculate the width of the lips in each frame and filter out outliers. From those remaining, select the one with the lowest distance between the upper and low lips.

    python -c "from replication import *; print(DataProcess('../replic/data/', 'obama2s.npy').get_closed_mouth_frame())"
#### Example usage : remove_identity(lmarks, template, file_out)
Following the specification described in the orginal paper, apply Procrustes analysis to the extracted landmarks `replic/samples/obama2s.npy`, reduce the frame rate to 25 fps, subtract the closed mouth frame and add to the template face. Save the resulting landmarks to `replic/data/obama2s.ir.npy`

    python -c "from replication import *; DataProcess('../replic/', 'samples/obama2s.npy').remove_identity(file_out = 'data/obama2s.ir.npy')"
### class Draw(plots_dir, data_proc, dimensions):
Manages plotting, annoting, saving of landmarks using [Matplotlib](https://matplotlib.org/)
#### Example usage(dpi, annot, lips_only)
Use procrustes analysis to align and normalise landmarks, plot and save them with annotations in `replic/plots/`

    python -c "from replication import *; Draw('../replic/plots/', DataProcess('../replic/data/', 'obama2s.npy')).save_plots_proc(annot=True)"
## code/test_utils.py
### function readme_test
Extract examples from this readme and execute them sequentially
#### Example usage

    python -c "from test_utils import *; readme_tests()"
## Potential adaptation to other models
The [replication script](code/replication.py) could be adapted to other models besides those created by Eskimez at al. The model's inferred landmarks must be saved in NPY format file with three axes - frame number, landmark number, and coordinates such as [this example](replic/samples/obama2s.npy).

## Prerequisites
In addition to the dependencies of [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), [FFmpeg](https://www.ffmpeg.org/) must be executable from the folder of the [replication script](code/replication.py)
