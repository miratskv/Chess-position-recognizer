import numpy as np 
import tensorflow as tf
from chessrec.models import DetectorBase


class BoardDetector(DetectorBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.resize_input = tf.keras.layers.Resizing(128, 128)
        self.conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides = (2,2), activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))
        self.conv_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides = (2,2), activation='relu')
        self.conv_6 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides = (2,2), activation='relu')

        self.dense_1 = tf.keras.layers.Dense(64, activation = tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_out = tf.keras.layers.Dense(4, activation = None)
        

        input_layer = tf.keras.Input(shape=(256,256,1,))
        output_layer = self.call(input_layer)

        super().__init__(inputs=input_layer,
                        outputs=output_layer,
                        **kwargs)

    def call(self, inputs, training=False):
        hidden = inputs/255.
        hidden = self.resize_input(hidden)
        hidden = self.conv_1(hidden)
        hidden = self.conv_2(hidden)
        hidden = self.conv_3(hidden)
        hidden = self.conv_4(hidden) 
        hidden = self.pool_1(hidden)
        hidden = self.conv_5(hidden) 
        hidden = self.conv_6(hidden)
        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = self.dense_1(hidden)
        hidden = self.dropout(hidden)
        hidden = self.dense_out(hidden)
        return hidden





