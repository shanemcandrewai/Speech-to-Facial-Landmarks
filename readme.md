# Landmark Generation From Speech

> Replicability and reproducibility, after all, are central to the creation and testing of theories and their acceptance by scientific communities (Berg, 2004)

This project is an attempt to replicate some of the results from Eskimez et al's paper [Noise-Resilient Training Method for Face Landmark Generation From Speech](https://ieeexplore.ieee.org/document/8871109).

The majority of the files are identical to [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface) from which this project is forked. The two most important enhancements are described below -

## code/generate.py

The [original generate script](https://github.com/eeskimez/noise_resilient_3dtface/blob/master/code/generate.py) reads audio files, infers the facial landmarks and generates animated faces. The [enhanced generate script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/generate.py) allows the user to save the predicted landmarks to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file. It also extends the animation functionality to accept an arbitrary externally created file of landmarks. These changes are limited to enhancements which could not put into a separate script and were carefully inserted in order to minimize the possibility of disturbing the original functionality.

### Functionality

In addition to the command-line options specified in [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), the enhanced [generate script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/generate.py) includes the following options -

* `-rd --replication_data_folder` Replication study data folder for predicted landmarks, by default `replic/data`
* `-rv --replication_video_folder` Replication study video folder for animation generated from loaded predictions, by default `replic/video_out`
* `-lf --load_prediction_file` File to load if `-l` load prediction flag is set, by default `../replic/data/ob25.npy`
* `-s --save_prediction` Disable creation of animiatons and instead save the predicted landmarks and speech array in the in the folder specified by the `-rd` command-line option
* `-l --load_prediction_and_paint` Load predictions from the file specified by the `-lf` command-line option and generate an animation in the folder specified by the `-rv` command-line option.

#### Examples

Save the landmarks predicted from the [ID_CNN](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/tree/master/pre_trained) model to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file in the default folders

    python generate.py -i ../speech_samples/ -m ../pre_trained/1D_CNN.pt -o ../results/1D_CNN/ -s

Load landmarks from an external file with the default path and create an animation in the default location

    python generate.py -i ../speech_samples/ -m ../pre_trained/1D_CNN.pt -o ../results/1D_CNN/ -l

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
The [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py) could be adapt to other besides those created by Eskimez at al.
=======
The [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py) could be adapted to other models besides those created by Eskimez at al. The model's inferred landmarks must be saved in NPY format file with axes - frame number, landmark number, and coordinates such as [this example](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/replic/samples/obama2s.npy).

## Prerequisites

In addition to the dependencies of [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), [FFmpeg](https://www.ffmpeg.org/) must be executable from the directory of the [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py)


