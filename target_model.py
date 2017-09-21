from keras import initializers
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import optimizers
from keras import losses


def target_model(LEARNING_RATE=1e-7):
    target_model = Sequential()
    target_model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same', input_shape=(64, 64, 4),
                            kernel_initializer=initializers.random_normal(stddev=0.01), bias_initializer=initializers.Constant(value=0.01), activation='relu'))
    target_model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=initializers.random_normal(
        stddev=0.01), bias_initializer=initializers.Constant(value=0.01), activation='relu'))
    target_model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=initializers.random_normal(
        stddev=0.01), bias_initializer=initializers.Constant(value=0.01), activation='relu'))
    target_model.add(Flatten())
    target_model.add(Dense(512, kernel_initializer=initializers.random_normal(
        stddev=0.01), bias_initializer=initializers.Constant(value=0.01), activation='relu'))
    target_model.add(Dense(2))
    adam = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9,
                           beta_2=0.999, epsilon=1e-08)
    target_model.compile(loss=losses.mean_squared_error, optimizer='adam')
    return target_model

if __name__ == '__main__':
    target_model()
