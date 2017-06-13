## Scene Generation based on PPGN

This repository contains source code necessary to generate scene based on PPGN.
["Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space"](http://arxiv.org/abs/1612.00005v1). Computer Vision and Pattern Recognition.


## Setup

### Installing software
This code is built on top of Caffe. You'll need to install the following:
* Install Caffe; follow the official [installation instructions](http://caffe.berkeleyvision.org/installation.html).
* Build the Python bindings for Caffe
* You can optionally build Caffe with the GPU option to make it run faster (recommended)
* Make sure the path to your `caffe/python` folder in [settings.py](settings.py#L2) is correct

### Downloading models
You will need to download a few models to run the examples below. There are `download.sh` scripts provided for your convenience.
* The generator network (Noiseless Joint PPGN-h) can be downloaded via: `cd nets/generator/noiseless && ./download.sh`
* The encoder network (here [BVLC reference CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)): 
`cd nets/caffenet && ./download.sh`
* Scene model, download [AlexNet CNN trained on MIT Places dataset](http://places.csail.mit.edu/): `cd nets/placesCNN && python gdrive_download.py`

### Settings:
* Paths to the downloaded models are in [settings.py](settings.py). They are relative and should work if the `download.sh` scripts run correctly.


## Examples
[See original examples](https://github.com/Evolving-AI-Lab/ppgn)

[scene_generation.ipynb](scene_generation.ipynb): 
Sampling conditioning on the class "lagoon" (output unit #204 of the places365). 
* Open [scene_generation.ipynb](scene_generation.ipynb) and run it to produces this result:<p 

<p align="center">
    <img src="http://i.imgur.com/jVp4krY.png" width=700px>
</p>
