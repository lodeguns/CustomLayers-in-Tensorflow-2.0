
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras import layers
from tensorflow.keras import backend as K
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D

batch_size = 128
num_classes = 10
epochs = 12
# input image dimensions
img_rows, img_cols = 28, 28


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(2, (3, 3), activation='relu'))
    model.add(CustomLayer())
    model.add(Conv2D(1, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


#extract the i-th element from the tensor
def ith_element(x, i, j, n_chan):
    return tf.reshape( x[0, i:i + 1, j:j + 1, n_chan ], (-1,))


@tf.function
def dyn_assignment(z, row, col, sel_chan, T ):
    # some operatiions
    out_val0 = ith_element(z, row, col, sel_chan)*T
    out_val1 = ith_element(z, row, col, sel_chan)*T
    value_to_assign = tf.math.add_n([out_val0, out_val1, out_val1])
    z = assign_op_tensor(z, value_to_assign, row, col, sel_chan)
    return z

@tf.function
def static_assignment(x, row, col, sel_chan, T):
    up_val =  tf.constant([T], dtype=tf.float32)
    z = assign_op_tensor(x, up_val, row, col, sel_chan)
    return z


@tf.function
def assign_op_tensor(x, updates, cord_i, cord_j, n_chan):
    indices = tf.constant([[0, cord_i, cord_j, n_chan]])
    updated = tf.tensor_scatter_nd_update(x, indices, updates)
    return(updated)

@tf.function
def out_res(x):
    dim = x.shape
    h = dim[1]
    w = dim[2]
    n_chan = dim[3]
    row = 6
    col = 6
    sel_chan = 1
    z = x
    z = dyn_assignment(z, row, col, sel_chan,  1/3)
    z = static_assignment(z, row, col, sel_chan, 1.)

    return z

@tf.custom_gradient
def custom_op(x):
    result = out_res(x) # do forward computation
    def custom_grad(dy):
        print(dy, [dy])
        grad = dy # compute gradient
        return grad
    return result, custom_grad

class CustomLayer(layers.Layer):
    def __init__(self):
        super(CustomLayer, self).__init__()

    def call(self, x):
        return custom_op(x)





# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
print(x_train.shape)
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# build the model

model = create_model()
# compile the model

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# train the model
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
