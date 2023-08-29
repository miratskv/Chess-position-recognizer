import numpy as np 
import tensorflow as tf
from chessrec.models import RecognizerBase


class PositionRecognizer(RecognizerBase):
    """
    Default network architecture for recognizing the pieces on board.
    Input image is processed by convolutional networks, then resized to (8,8,n_features).
    Dense layer and softmax is then used to classify the (8,8) representation.
    Note:
    The model is as simple as possible (while the performance turned out to be still quite reasonable):
        Flattening is not used before dense part on purpose - in all application, the board fills 
            almost the whole image and the orientation is always the same (no tilting etc.).
        No BatchNorm used or residual connections used, as it seems the task is so simple it 
            does not improve accuracy significantly
        ...

        Overall it looks like the training data quality plays a bigger role in this case than the actuall network
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.resize_input = tf.keras.layers.Resizing(128, 128)
        self.conv_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation='relu')
        self.conv_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation='relu')
        self.resize_hidden = tf.keras.layers.Resizing(8, 8)
        self.dense_1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_out = tf.keras.layers.Dense(13, activation=tf.nn.softmax)

        # TODO: This is just a cheat to make self.summary() work properly.
        # Double check if it does not interfere with anything
        input_layer = tf.keras.Input(shape=(256,256,1,))
        output_layer = self.call(input_layer)
        super().__init__(
            inputs=input_layer,
            outputs=output_layer,
            **kwargs)

    def call(self, inputs, training=False):
        hidden = inputs/255.
        hidden = self.resize_input(hidden)
        hidden = self.conv_1(hidden)
        hidden = self.conv_2(hidden)
        hidden = self.conv_3(hidden)
        hidden = self.conv_4(hidden) 
        hidden = self.conv_5(hidden)
        hidden = self.resize_hidden(hidden)
        hidden = self.dense_1(hidden)
        hidden = self.dropout(hidden)
        hidden = self.dense_out(hidden)
        return hidden