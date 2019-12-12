# projetoGurupi
## Project Summary and Results
A tool for image restoration from video frames using Bayesian Super-resolution.
The method is described in the paper by [Tipping and Bishop](http://www.cs.cmu.edu/~kangli/doc/papers/tipping03.pdf).
The work was done in partial fulfillment to obtain the Barchelor's degree in computer engineering.
A full report on the project is found [here](https://github.com/danchieta/projetoGurupi_texto/blob/master/projetoGurupi.pdf) (in Portuguese).

This tool is able to obtain a high resolution image from a set of low resolution
images from the same scene.
Tests were made using only synthetic data as input, i.e. the low resolution 
images were obtained by downsampling, warping an rotating a higher resolution
image.

The HR image has 152x104 and is shown below.

![original](https://github.com/danchieta/projetoGurupi/blob/master/testIMG/imtestes.png)

A set of LR images with 38x26 pixels were generated from it.

![degraded image](https://github.com/danchieta/projetoGurupi/blob/master/readme_img/resized_nearest.png)

The image below was obtained from the set of LR images using Bayesian super resolution.

![restored w sr](https://github.com/danchieta/projetoGurupi/blob/master/readme_img/restored_sr.png)
