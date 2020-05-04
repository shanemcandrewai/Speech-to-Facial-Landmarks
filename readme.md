# Landmark Generation From Speech

> Replicability and reproducibility, after all, are central to the creation and testing of theories and their acceptance by scientific communities (Berg, 2004)

This project is an attempt to replicate some of the results from Eskimez et al's paper [Noise-Resilient Training Method for Face Landmark Generation From Speech](https://ieeexplore.ieee.org/document/8871109).

## Prerequisites

In addition to the dependencies of [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface) from which this project is forked, [FFmpeg](https://www.ffmpeg.org/) must be executable from the directory of the [replication script](https://github.com/shanemcandrewai/Speech-to-Facial-Landmarks/blob/master/code/replication.py)

## Code Example



In addition to those specified in [Noise-Resilient Training Method](https://github.com/eeskimez/noise_resilient_3dtface), the generation script includes the following parameters -

* -in --- Replicaton study Directory of predicted landmarks, by default `replic` 
* -p --- Disable creation of plots and video

You can run the following code to test the system:

```
python generate.py -i ../speech_samples/ -m ../pre_trained/1D_CNN.pt -o ../results/1D_CNN/ -p
```

