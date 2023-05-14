from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras import preprocessing
import tensorflow as tf
from argparse import ArgumentParser
import numpy as np
from vit.model import ViT, ViTBase, ViTHuge, ViTLarge

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--test-image", default='./test.png', type=str, required=True)
    parser.add_argument(
        "--model-folder", default='./model/', type=str)

    args = parser.parse_args()

    # Loading Model
    # model = load_model(args.model_folder)
    model = load_model(args.model_folder, compile=False)

    test_image = image.load_img(args.test_image, target_size = (150, 150)) 
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    predictions = model.predict(test_image)
    if np.argmax(predictions) == 0:
        ans = 'CAT'
    else:
        ans = 'DOG'
    print(f'Result: {ans}')