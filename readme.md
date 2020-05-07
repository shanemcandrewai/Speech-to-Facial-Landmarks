# Landmark Generation From Speech

> Replicability and reproducibility, after all, are central to the creation and testing of theories and their acceptance by scientific communities (Berg, 2004)

This project is an attempt to replicate some of the results from Eskimez et al's paper [Noise-Resilient Training Method for Face Landmark Generation From Speech](https://ieeexplore.ieee.org/document/8871109).

The majority of the files are identical to [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface) from which this project is forked. The two most important enhancements are described below -

## code/generate.py

The [original generate script](https://github.com/eeskimez/noise_resilient_3dtface/blob/master/code/generate.py) reads audio files, infers the facial landmarks and generates animated faces. The [enhanced generate script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/generate.py) allows the user to save the predicted landmarks to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file. It also extends the animation functionality to accept an arbitrary externally created file of landmarks. These changes are limited to enhancements which could not put into a separate script and were carefully inserted in order to minimize the possibility of disturbing the original functionality. To view the changes, execute `git diff 3c804caff3e4e0cabd7259ccb97c4038b509d630 code/generate.py`

### Extended Functionality

In addition to the command-line options specified in [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), the enhanced [generate script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/generate.py) includes the following -

* `-s --save_prediction` save the predicted landmarks and speech array in the in the folder specified by the `-o` option and disable generation of animation
* `-l --load_prediction_and_paint` load predictions from the folder specified by the `-i` option and generate a painted face animation in the folder specified by the `-o` option. This option expects the input folder to contain pairs of files with the same name but different extensions - `.wav` and `.npy`

#### Examples

Save the landmarks predicted and speech vector using the [ID_CNN](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/tree/master/pre_trained) model from `../speech_samples/` to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file in `replic/pred_out`

    python generate.py -i ../speech_samples/ -m ../pre_trained/1D_CNN.pt -o ../replic/pred_out -s  

Load landmarks from an external file in `../replic/samples/files_in/` and generate animation in `../replic/pred_out`

    python generate.py -i ../replic/samples/files_in/ -m ../pre_trained/1D_CNN.pt -o ../replic/anim_out/ -l

## code/replication.py

Eskimez et al. released pre-trained models and the generation script with the express aim of promoting scientific reproducibility; however the tools for achieving this were not included. The [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py) is an attempt to fill this gap.

### Classes
#### Video
manages frame extraction and video manipulation using [FFmpeg](https://www.ffmpeg.org/)
##### Example usage
Extract frames from `replic/samples/obama2s.mp4` into `replic/frames`
    python -c "from replication import *; Video('../replic/samples/', frames=Frames('../replic/frames')).extract_frames('obama2s.mp4')"
#### Frames
Helper class used to manage a folder of frames extracted from source video. Each frame is jpeg file named according to the frame number.
#### DlibProcess
manages the extraction of landmarks from individual frames using the [Dlib toolkit](http://dlib.net/)
#### DataProcess
calculations and supporting methods required for the replication of experiments
#### Draw
manages this plotting, annoting, saving of landmarks using [Matplotlib](https://matplotlib.org/)

## Potential adaptation to other models
The [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py) could be adapted to other models besides those created by Eskimez at al. The model's inferred landmarks must be saved in NPY format file with axes - frame number, landmark number, and coordinates such as [this example](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/replic/samples/obama2s.npy).

## Prerequisites

In addition to the dependencies of [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), [FFmpeg](https://www.ffmpeg.org/) must be executable from the directory of the [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py)


