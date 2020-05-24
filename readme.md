# Landmark Generation From Speech

> Replicability and reproducibility, after all, are central to the creation and testing of theories and their acceptance by scientific communities (Berg, 2004)

This project is an attempt to replicate some of the results from Eskimez et al's paper [Noise-Resilient Training Method for Face Landmark Generation From Speech](https://ieeexplore.ieee.org/document/8871109).

In this [comparison video](replic/shared/080815_WeeklyAddress_25t_compare_v.mp4), we see an original video of Barack Obama, a plot of the facial landmarks extracted with the [Dlib toolkit](http://dlib.net/) on the top-right, the inferred animation created with Easkimez et al's script on the bottom-left and finally an animation produced by my attempt to replicate the method in the original paper on the bottom-right.

The majority of the files are identical to [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface) from which this project is forked. Apart from this readme, the two most important enhancements and some testing utilities are described below -

## code/generate.py

The [original generate script](https://github.com/eeskimez/noise_resilient_3dtface/blob/master/code/generate.py) reads audio files, infers the facial landmarks and generates animated faces. The [enhanced generate script](code/generate.py) allows the user to save the predicted landmarks to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file. It also extends the animation functionality to accept an arbitrary externally created file of landmarks. These changes are limited to enhancements which could not be put into a separate script and were carefully inserted in order to minimise the possibility of disturbing the original functionality. In addition, some redundant code was removed as a result of running [Pylint](https://www.pylint.org/) on the script. To view the precise changes, execute `git diff 3c804caff3e4e0cabd7259ccb97c4038b509d630 code/generate.py`

### Extended Functionality

In addition to the command-line options specified in [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), the [enhanced generate script](code/generate.py) includes the following -

* `-s --save_prediction` save the predicted landmarks and speech array in the folder specified by the `-o` option and disable generation of animation
* `-l --load_prediction` load predictions from the folder specified by the `-i` option and generate a painted face animation in the folder specified by the `-o` option. This option expects the input folder to contain pairs of files with the same name but different extensions - `.wav` and `.npy`

These enhancements were subsequently merged into the [upstream repository](https://github.com/eeskimez/noise_resilient_3dtface/commit/59536f4ebe43bcabd0b2f90a93974552e87dc553)

#### Examples

Save the landmarks predicted and speech vector using the [ID_CNN](pre_trained/1D_CNN.pt) model from audio in `replic/shared/obama2s/` to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file in `replic/pred_out/`

    python generate.py -i ../replic/shared/obama2s/ -m ../pre_trained/1D_CNN.pt -o ../replic/pred_out/ -s  

Load landmarks from an external files in `replic/shared/identity_removed/` and generate animation in `replic/anim_out/`

    python generate.py -i ../replic/shared/identity_removed/ -m ../pre_trained/1D_CNN.pt -o ../replic/anim_out/ -l

## code/replication.py

Eskimez et al. released pre-trained models and the generation script with the express aim of promoting scientific reproducibility; however the tools for achieving this were not included. The [replication toolkit](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py) is an attempt to fill this gap. The [API documentation](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/) is generated automatically by [pdoc](https://pdoc3.github.io/pdoc/)
### Differences and Interpretations of the original paper
#### Face Landmark Identity Removal
Eskimez et al. noted that even after applying [procrustes analysis](https://link.springer.com/article/10.1007/BF02291478), there was still a significant correlation between the aligned landmarks and the facial characteristics of the individual speaker. The goal is to minimise these so that the neural network can learn the relationship between speech and facial landmarks regardless of the speaker's identity. Eskimez et al. select a reference frame with a closed mouth and then calculate the differences between this and the corresponding landmarks resulting from procrustes analysis. This reference is calculated based on the distance between the upper and lower lip coordinates.  However this simple calculation can result in reference frames which are far from average such as a frame where the speaker has [pursed lips](replic/shared/obpursed.jpg)

In order to avoid this problem, the calculation was [enhanced](code/replication.py#L183) to exclude frames where the mouth is unusually wide or narrow.

Next a template face is calculated based the average of all the closed mouth references measured across all identities. The differences between each frame and the selected reference frame for the particular speaker are calculated and added to the template face.

### [class Video](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.Video):
Manages frame extraction and video manipulation using [FFmpeg](https://www.ffmpeg.org/)
#### [method extract_frames](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.Video.extract_frames) example usage : 
Extract frames from `replic/shared/obama2s.mp4` into `replic/frames/`

    python -c "from replication import *; Video().extract_frames('../replic/shared/obama2s.mp4')"
#### [method extract_audio](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.Video.extract_audio) example usage :
Extract audio in WAV format from video `replic/shared/obama2s.mp4` to `replic/audio/obama2s.wav`

    python -c "from replication import *; Video().extract_audio('../replic/shared/obama2s.mp4')"
#### [method draw_text](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.Video.draw_text) example usage : 
Add the frame number and timestamp to video `replic/identity_removed/obama2s.ir_painted_.mp4` and save as `replic/anim_out/obama2s.ir_painted_t.mp4`

    python -c "from replication import *; Video().draw_text('../replic/shared/identity_removed/obama2s.ir_painted_.mp4', '../replic/anim_out/obama2s.ir_painted_t.mp4')"
#### [method stack_h](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.Video.stack_h) example usage : 
Stack input videos `replic/shared/obama2s/obama2s_painted_t.mp4` and `replic/shared/identity_removed/obama2s.ir_painted_t.mp4` horizontally and for easier visual comparision into `replic/anim_out/obama2s_comparison.mp4`

    python -c "from replication import *; Video().stack_h('../replic/shared/obama2s/obama2s_painted_t.mp4', '../replic/shared/identity_removed/obama2s.ir_painted_t.mp4', '../replic/anim_out/obama2s_comp_h.mp4')"
### [class Frames](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.Frames):
Helper class used to manage a folder of frames extracted from source video. Each frame is jpeg file named according to the frame number.
#### [method get_frame_nums](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.Frames.get_frame_nums) example usage :
Get frame numbers from `replic/frames/`

    python -c "from replication import *; print(Frames().get_frame_nums())"
### [class DlibProcess](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.DlibProcess):
Manages the extraction of landmarks from individual frames using the [Dlib toolkit](http://dlib.net/)
#### [method display_overlay](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.DlibProcess.display_overlay) example usage :
Extract landmarks from Frame 30 and overlay the frame image with corresponding line plots

    python -c "from replication import *; DlibProcess().display_overlay(frame_num=30)"
### [class DataProcess](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.DataProcess):
Calculations and supporting methods required for the replication of experiments
#### [method get_closed_mouth_frame](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.DataProcess.get_closed_mouth_frame) example usage :
First calculate the width of the lips in each frame and filter out outliers. From those remaining, select the one with the lowest distance between the upper and low lips.

    python -c "from replication import *; print(DataProcess('../replic/shared/obama2s.mp4').get_closed_mouth_frame())"
#### [method remove_identity](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.DataProcess.remove_identity) example usage :
Following the specification described in the orginal paper, apply Procrustes analysis to the extracted landmarks `replic/shared/obama2s.npy`, reduce the frame rate to 25 fps, subtract the closed mouth frame and add to the template face. Save the resulting landmarks to `replic/data/obama2s.ir.npy`

    python -c "from replication import *; print(DataProcess(dlib_proc=DlibProcess(lmarks_file='../replic/shared/obama2s.npy')).remove_identity(id_removed_file = '../replic/data/obama2s.ir.npy'))"
### [class Draw](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.Draw):
Manages plotting, annoting, saving of landmarks using [Matplotlib](https://matplotlib.org/)
#### [method save_plots_proc](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.Draw.save_plots_proc) example usage :
Use procrustes analysis to align and normalise landmarks, plot and save them with annotations in `replic/plots/`

    python -c "from replication import *; Draw('../replic/plots/', DataProcess(dlib_proc=DlibProcess(lmarks_file='../replic/shared/obama2s.npy'))).save_plots_proc(annot=True)"
### [class Analysis](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.Analysis):
Data extraction and analysis 
#### [method calc_rmse](https://shanemcandrewai.github.io/Speech-to-Facial-Landmarks/#replication.Analysis.calc_rmse) example usage :
Extract audio from `replic/shared/obama2s.mp4` and use the pre-trained model to predict landmarks. Extract landmarks from the video, preprocess and calculate the root mean square error """

    python -c "from replication import *; Analysis().calc_rmse('../replic/shared/obama2s.mp4')"
## [code/test_utils.py](code/test_utils.py)
### function readme_test
Extract examples from this readme and execute them sequentially
#### Example usage

    python -c "from test_utils import *; readme_tests()"
## Potential adaptation to other models
The [replication toolkit](code/replication.py) could be adapted to other models besides those created by Eskimez at al. The model's inferred landmarks must be saved in NPY format file with three axes - frame number, landmark number, and coordinates such as [this example](replic/shared/obama2s.npy).

## Prerequisites
In addition to the dependencies of [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), [FFmpeg](https://www.ffmpeg.org/) must be executable from the folder of the [replication toolkit](code/replication.py)
