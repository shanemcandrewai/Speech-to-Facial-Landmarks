# Landmark Generation From Speech

> Replicability and reproducibility, after all, are central to the creation and testing of theories and their acceptance by scientific communities (Berg, 2004)

This project is an attempt to replicate some of the results from Eskimez et al's paper [Noise-Resilient Training Method for Face Landmark Generation From Speech](https://ieeexplore.ieee.org/document/8871109).

The majority of the files are identical to [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface) from which this project is forked. Apart from this readme and some testing utilities, the two most important enhancements are described below -

## code/generate.py

The [original generate script](https://github.com/eeskimez/noise_resilient_3dtface/blob/master/code/generate.py) reads audio files, infers the facial landmarks and generates animated faces. The [enhanced generate script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/generate.py) allows the user to save the predicted landmarks to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file. It also extends the animation functionality to accept an arbitrary externally created file of landmarks. These changes are limited to enhancements which could not put into a separate script and were carefully inserted in order to minimize the possibility of disturbing the original functionality. In addition, some redundant code was removed as a result of running [Pylint](https://www.pylint.org/) on the script. To view the precise changes, execute `git diff 3c804caff3e4e0cabd7259ccb97c4038b509d630 code/generate.py`

### Extended Functionality

In addition to the command-line options specified in [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), the enhanced [generate script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/generate.py) includes the following -

* `-s --save_prediction` save the predicted landmarks and speech array in the folder specified by the `-o` option and disable generation of animation
* `-l --load_prediction` load predictions from the folder specified by the `-i` option and generate a painted face animation in the folder specified by the `-o` option. This option expects the input folder to contain pairs of files with the same name but different extensions - `.wav` and `.npy`

#### Examples

Save the landmarks predicted and speech vector using the [ID_CNN](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/tree/master/pre_trained) model from `../speech_samples/` to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file in `replic/pred_out`

    python generate.py -i ../speech_samples/ -m ../pre_trained/1D_CNN.pt -o ../replic/pred_out -s  

Load landmarks from an external file in `../replic/samples/files_in/` and generate animation in `../replic/pred_out`

    python generate.py -i ../replic/samples/files_in/ -m ../pre_trained/1D_CNN.pt -o ../replic/anim_out/ -l

## code/replication.py

Eskimez et al. released pre-trained models and the generation script with the express aim of promoting scientific reproducibility; however the tools for achieving this were not included. The [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py) is an attempt to fill this gap.
### Differences and Interpretations of the original paper
#### Face Landmark Identity Removal
Eskimez et al. noted that even after applying [procrustes analysis](https://link.springer.com/article/10.1007/BF02291478), there was still a significant correlation between the aligned landmarks and the facial characteristics of the individual speaker. The goal is to minimize these so that the neural network can learn the relationship between speech and facial landmarks regardless of the speaker's identity. Eskimez et al. select a reference frame with a closed mouth and then calculate the differences between this and the corresponding landmarks in the resulting from procrustes analysis. This reference is calculated based on the distance between the upper and lower lip coordinates.  However this simple calculation can result in reference frames which are far from average such as a frame where the speak has pursed lips.

In order to avoid this problem, the calculation was [enhanced](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py#L183) to exclude frames where the mouth is unusually wide or narrow.

Next a template face is calculated based the average of all the closed mouth references measured across all identities. The differences between each frame and the selected reference frame for the particular speaker are calculated and added to the template face.

### class Video:
manages frame extraction and video manipulation using [FFmpeg](https://www.ffmpeg.org/)
#### Example usage
Extract frames from `replic/samples/obama2s.mp4` into `replic/frames`

    python -c "from replication import *; Video('../replic/samples/', frames=Frames('../replic/frames')).extract_frames('obama2s.mp4')"
### class Frames:
Helper class used to manage a folder of frames extracted from source video. Each frame is jpeg file named according to the frame number.
#### Example usage
Get frame numbers from `../replic/frames`

    python -c "from replication import *; print(Frames('../replic/frames').get_frame_nums())"
### class DlibProcess:
Manages the extraction of landmarks from individual frames using the [Dlib toolkit](http://dlib.net/)
#### Example usage
Extract landmarks from Frame 30 and overlay the frame image with corresponding line plots

    python -c "from replication import *; DlibProcess().display_overlay(frame_num=30)"
### class DataProcess:
Calculations and supporting methods required for the replication of experiments
#### Example usage : get_closed_mouthframe:
First calculate the width of the lips in each frame and filter out outliers. From these remaining, the one with the lowest distance between the upper and low lips.

    python -c "from replication import *; print(DataProcess('../replic/data').get_closed_mouth_frame('obama2s.npy'))"
### class Draw:
Manages plotting, annoting, saving of landmarks using [Matplotlib](https://matplotlib.org/)
#### Example usage
Use procrustes analysis to align and normalise landmarks, plot and save them in `replic/plots`

    python -c "from replication import *; Draw('../replic/plots').save_plots_proc(annot=True, extract_file='../replic/samples/obama2s.npy')"

## code/test_utils.py
### function readme_test
Extract examples from this readme and execute automatically sequencially
#### Example usage

    python -c "from test_utils import *; readme_tests()"
## Potential adaptation to other models
The [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py) could be adapted to other models besides those created by Eskimez at al. The model's inferred landmarks must be saved in NPY format file with three axes - frame number, landmark number, and coordinates such as [this example](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/replic/samples/obama2s.npy).

## Prerequisites
In addition to the dependencies of [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), [FFmpeg](https://www.ffmpeg.org/) must be executable from the directory of the [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py)
