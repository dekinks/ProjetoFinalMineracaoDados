from PIL import Image
import time
from keras_segmentation.models.unet import vgg_unet
import tensorflow as tf

def main():
    learning()

def learning():

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')

    n_classes = 24
    epochs = 30
    model = vgg_unet(n_classes=n_classes, input_height=416, input_width=608)
    model.train( 
        train_images =  "original_images/",
        train_annotations = "masks_images_semantic_result/",
        checkpoints_path = "vgg_unet" , epochs=epochs)

    for i in range(60):
        input_image = "prev_images/" + str(i) + ".jpg"
        out = model.predict_segmentation(
            inp=input_image,
            out_fname="out" + str(i) + ".png"
        )

if __name__ == '__main__':
    main() 