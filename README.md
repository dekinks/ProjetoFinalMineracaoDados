# Projeto Final de Mineração de Dados</br>
Código do projeto final da matéria de mineração de dados UFMG</br>

1 - Criar máscaras com o labelme</br>
    >pip install pyqt5 -i https://pypi.tuna.tsinghua.edu.cn/simple</br>
    >pip install labelme -i https://pypi.tuna.tsinghua.edu.cn/simple</br>
    labelme

2 - Converter para png</br>
    >labelme_json_to_dataset <imagem_para_converter>.json -o <imagem_para_converter>_json</br>

3 - Criar as classes (labels) em .csv</br>

4 - Algoritmo de pré-processamento</br>

5 - Na pasta machine_learning consta o arquivo trainingImages.py. Neste arquivo, é feita a fase de treinamento com as imagens originais e suas máscaras semânticas (obtidas em 4). Em seguida, o algoritmo para classificação é feito colocando como input uma imagem original qualquer e retornando o resultado do aprendizado em out.png. O aprendizado de máquina da segmentação semântica é feito utilizando a biblioteca vgg_unet do Keras e TensorFlow</br>
    >pip install keras_segmentation.models.unet</br>
    >pip install tensorflow</br>
