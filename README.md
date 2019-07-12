# projetoGurupi
## Project Summary and Results
Uma ferramenta de restauração de imagens a partir de frames de vídeo utilizando super-resolução Bayesiana.
O método utilizado  descrito no artigo de [Michael tipping e Chrstopher Bishop](http://www.cs.cmu.edu/~kangli/doc/papers/tipping03.pdf).
Este projeto foi feito como parte do Trabaho de Conclusão de curso do autor para a obtenço do grau de Bacharel em Engenharia da Computaço pela Universidade Federal do Pará, o texto do trabalho está dispinível [aqui](https://github.com/danchieta/projetoGurupi_texto/blob/master/projetoGurupi.pdf).

This tool is able to obtain a high resolution image from a set of low resolution
images from the same sccene.
The tool was only tested with synthetic data, the low resolution images are degraded versions of
a higher resolution image.

The HR image has 152x92 and is shown below.

![original](https://github.com/danchieta/projetoGurupi/blob/master/readme_img/imteste2s.png)

A set of LR images with 38x23 pixels were generated from it.

![degraded image](https://github.com/danchieta/projetoGurupi/blob/master/readme_img/resized_nearest2.png)

The image below was obtained from the set of LR images using bayesian super resolution.

![restored w sr](https://github.com/danchieta/projetoGurupi/blob/master/readme_img/restored_image_sr2.png)
