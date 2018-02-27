# Scripts Python
Esta pasta contém os módulos e scripts desenvolvidos na execução do projeto.

## Módulos
* **genModel**: Este módulo contém os métodos que implementam o modelo de observação, utilizado tanto para gerar as imagens simuladas quanto na restauração.
* **srModel**: Este módulo contém as funções e classes usadas na restauração das imagens.
* **vismodule**: Neste módulo estão as funções usadas para visualizar os dados e resultados usando o módulo matplotlib. Eles estão separados para reduzir o tamanho dos scripts.

## Scripts
* **geraImagemLR.py**: Este script gera as imagens a partir de uma imagem de teste localizada em testIMG
* **parameterOptimization**: Este script realiza a etapa de otimização do parâmetros de degradação, a qual é a primeira do processo de restauração.
* **imageOptimization**: Este script realiza a otimização da imagem usando as imagens de teste e os valores obtidos na etapa de otimização dos parâmetros.
