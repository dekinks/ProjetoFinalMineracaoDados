from PIL import Image
import time
from keras_segmentation.models.unet import vgg_unet

def main():
    learning()

def learning():
    n_classes = 24
    epochs = 5
    model = vgg_unet(n_classes=n_classes, input_height=416, input_width=608)
    model.train( 
        train_images =  "original_images/",
        train_annotations = "masks_images_semantic_result/",
        checkpoints_path = "vgg_unet" , epochs=epochs)

    begin = time.time()
    for i in range(120):
        input_image = "original_images/" + str(i) + ".jpg"
        out = model.predict_segmentation(
            inp=input_image,
            out_fname="out" + str(i) + ".png"
        )
    end = time.time()
    print(begin - end) 

if __name__ == '__main__':
    main() 