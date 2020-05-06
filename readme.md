# Landmark Generation From Speech

> Replicability and reproducibility, after all, are central to the creation and testing of theories and their acceptance by scientific communities (Berg, 2004)

This project is an attempt to replicate some of the results from Eskimez et al's paper [Noise-Resilient Training Method for Face Landmark Generation From Speech](https://ieeexplore.ieee.org/document/8871109).

The majority of the files are identical to [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface) from which this project is forked. My extensions are described below -

## code/generate.py

The original script reads audio files, infers the facial landmarks and created animated faces. The [enhanced generate script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/generate.py) allows the user to save the predicted landmarks to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file. It also extends the animation functionality to accept an externally created file of landmarks

## Prerequisites

In addition to the dependencies of [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), [FFmpeg](https://www.ffmpeg.org/) must be executable from the directory of the [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py)

## Functionality

In addition to those specified in [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), the enhanced [generate script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/generate.py) includes the following parameters -

* `-rd --replication_data_folder` Replication study data folder for predicted landmarks, by default `replic/data`
* `-rv --replication_video_folder` Replication study video folder for video generated from loaded predictions, by default `replic/video_out`
* `-lf --load_prediction_file` File to load if `-l` load prediction flag is set, by default `../replic/data/ob25.npy`
* `-s --save_prediction` Disable creation of video and save the predicted landmarks and speech array in the replication data study folder
* `-l --load_prediction_and_paint` Load prediction from the specified by the `-lf` command-line option and generate painted video

### Examples

Save the predicted landmarks from a the [ID_CNN](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/tree/master/pre_trained) model from the sample audio to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file 

    python generate.py -i ../speech_samples/ -m ../pre_trained/1D_CNN.pt -o ../results/1D_CNN/ -s

## Potential adaptation to other models
