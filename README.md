# Projeto Final de Mineração de Dados
Código do projeto final da matéria de mineração de dados UFMG

1 - Criar máscaras com o labelme
    pip install pyqt5 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install labelme -i https://pypi.tuna.tsinghua.edu.cn/simple
    labelme

2 - Converter para png
    labelme_json_to_dataset <imagem_para_converter>.json -o <imagem_para_converter>_json

3 - Criar as classes (labels) em .csv

4 - Algoritmo de pré-processamento

5 - Na pasta machine_learning consta o arquivo trainingImages.py. Neste arquivo, é feita a fase de treinamento com as imagens originais e suas máscaras semânticas (obtidas em 4). Em seguida, o algoritmo para classificação é feito colocando como input uma imagem original qualquer e retornando o resultado do aprendizado em out.png. O aprendizado de máquina da segmentação semântica é feito utilizando a biblioteca vgg_unet do Keras e TensorFlow
    pip install keras_segmentation.models.unet
    pip install tensorflow
