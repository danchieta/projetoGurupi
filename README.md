# projetoGurupi
## Resumo do projeto e resultados
Uma ferramenta de restauração de imagens a partir de frames de vídeo utilizando super-resolução Bayesiana.
O método utilizado  descrito no artigo de [Michael tipping e Chrstopher Bishop](http://www.cs.cmu.edu/~kangli/doc/papers/tipping03.pdf).
Este projeto foi feito como parte do Trabaho de Conclusão de curso do autor para a obtenço do grau de Bacharel em Engenharia da Computaço pela Universidade Federal do Pará, o texto do trabalho está dispinível [aqui](https://github.com/danchieta/projetoGurupi_texto/blob/master/projetoGurupi.pdf).

Em resumo, o projeto é capaz de obter uma imagem de _alta resolução_ a partir de um conjunto de várias imagens de
baixa resolução de uma mesma cena.
No momento, a ferramenta só trabalha com um conjunto sintético de imagens,
ou seja, os quadros de vídeo são simulados e gerados a partir de uma imagem de teste.

A ferramenta parte de uma imagem de 152x92 pixels como a mostrada abaixo.

![original](https://github.com/danchieta/projetoGurupi/blob/master/readme_img/imteste2s.png)

A partir dessa imagem, foi gerado um conjunto de imagens degradadas de 38x23 pixels.

![degraded image](https://github.com/danchieta/projetoGurupi/blob/master/readme_img/resized_nearest2.png)

Essas imagens foram utilizadas para obter a seguinte imagem restaurada.

![restored w sr](https://github.com/danchieta/projetoGurupi/blob/master/readme_img/restored_image_sr2.png)
