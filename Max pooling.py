import numpy as np
from keras.layers import MaxPooling2D
from keras.models import Sequential

def max_pooling():
    image = np.array([[2, 3, 4, 2],
                      [8, 5, 5, 1],
                      [6, 7, 9, 4],
                      [3, 1, 4, 5]])

    # reshaping
    arr = image.reshape(1, 4, 4, 1)

    # define a max pooling layer
    max_pool = MaxPooling2D(pool_size=2, strides=2)

    # define a sequential model with just one pooling layer
    model = Sequential(
        [max_pool])

    # get the output
    output = model.predict(arr)
    output = np.squeeze(output)
    return output


if __name__ == '__main__':
    print(max_pooling())
