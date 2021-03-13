from PIL import Image
import matplotlib.pyplot as plt
from keras_segmentation.models.unet import vgg_unet

def main():
    learning()

def learning():
    n_classes = 24
    epochs = 20
    model = vgg_unet(n_classes=n_classes, input_height=416, input_width=608)
    model.train( 
        train_images =  "original_images/",
        train_annotations = "masks_images_semantic_result/",
        checkpoints_path = "vgg_unet" , epochs=epochs)

    input_image = "original_images/001.jpg"
    out = model.predict_segmentation(
        inp=input_image,
        out_fname="out.png"
    )

    fig, axs = plt.subplots(1, 2, figsize=(20, 20), constrained_layout=True)

    img_orig = Image.open(input_image)
    axs[0].imshow(img_orig)
    axs[0].set_title('original image-002.jpg')
    axs[0].grid(False)

    axs[1].imshow(out)
    axs[1].set_title('prediction image-out.png')
    axs[1].grid(False)

    print(out)
    print(out.shape)

if __name__ == '__main__':
    main() 