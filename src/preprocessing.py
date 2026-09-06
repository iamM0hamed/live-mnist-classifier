import numpy as np


def normalize_image(image):
    return image.astype("float32") / 255.0


def expanding_channels(image):
    return np.expand_dims(image, -1)


def preprocess_images(images):
    images = normalize_image(images)
    images = expanding_channels(images)
    return images


if __name__ == "__main__":
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    print(x_train.shape)
    print(x_train.dtype)
    print(x_train.min(), x_train.max())
