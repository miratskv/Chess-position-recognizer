#!/usr/bin/env python3

import os
import pkg_resources
import argparse

import numpy as np 
import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers
import cv2

from chessrec.data_generator import ChessBoardGenerator as CBgen
from chessrec.models.metrics import BoardAccuracy, IoU
from chessrec.models.position_recognizer_v0 import PositionRecognizer
import chessrec.constants as consts


app_path = pkg_resources.resource_filename('chessrec', "")
example_assets = os.path.join(app_path, "cmds", "generator_assets_example")


parser = argparse.ArgumentParser()
parser.add_argument("--piece_sets_path", default=os.path.join(example_assets, "chess_pieces"), type=str, help="Path to piece sets")
parser.add_argument("--boards_imgs_path", default=os.path.join(example_assets, "chess_boards"), type=str, help="Path to boards")
parser.add_argument("--background_im_path", default=os.path.join(example_assets, "backgrounds"), type=str, help="Path to background images")

parser.add_argument("--threads", default=1, type=int, help="NUmber of threads to use during training")
parser.add_argument("--load_recognizer", default='', type=str, help="Path to pretrained weights")
parser.add_argument("--save_recognizer", default="trained_recognizer", type=str, help="Path to save the weights")
parser.add_argument("--load_val_dataset", default="", type=str, 
    help="Load some fixed validation dataset; if empty, validation dataset will not be fixed during training")
parser.add_argument("--val_dataset_size", default=2000, type=int, help="")
parser.add_argument("--train_batches_per_epoch", default=100, type=int, help="")
parser.add_argument("--train_data_update_period", default=20, type=int, 
    help="Every _th epoch, generator generates new train dataset")
parser.add_argument("--batch_size", default=128, type=int, help="")
parser.add_argument("--epochs", default=1000, type=int, help="")

# preprocessing constants
MAX_BRIGHTNESS_DELTA = 0.4
MIN_CONTRAST = 0.5
MAX_CONTRAST = 1.5
GAUSS_NOISE_PROB = 0.1
GAUSS_NOISE_VAR = 0.6

def train_preprocess(img, label, bbox):
    label = tf.one_hot(label, consts.SQUARE_CLASSES)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.random_contrast(img, MIN_CONTRAST, MAX_CONTRAST)
    img = tf.image.random_brightness(img, max_delta=MAX_BRIGHTNESS_DELTA)
    img = tf.cast(img, tf.float32)
    if tf.random.uniform([], minval=0, maxval=1) > 1-GAUSS_NOISE_PROB:
        img = tf.keras.layers.GaussianNoise(GAUSS_NOISE_VAR)(img, training=True)
    return img, label, bbox

def validation_preprocess(img, label, bbox):
    label = tf.one_hot(label, consts.SQUARE_CLASSES)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.cast(img, tf.float32)
    return img, label, bbox

def fit_recognizer(args, recognizer):
    data_generator = CBgen(
            args.boards_imgs_path, 
            args.piece_sets_path,
            args.background_im_path).tfGenerator(only_boards=True)
    train_generator = data_generator.map(train_preprocess).map(lambda x,y,z: (x,y))
    val_generator = data_generator.map(validation_preprocess).map(lambda x,y,z: (x,y))

    if args.load_val_dataset:
        dataset_val = tf.data.Dataset.load(args.load_val_dataset)
    else:
        dataset_val = val_generator.take(args.val_dataset_size).batch(args.batch_size) 
    
    for epoch in range(args.epochs):
        print(f'EPOCH: {epoch}')
        if epoch % args.train_data_update_period == 0:
            dataset_train = train_generator.batch(args.batch_size)
            dataset_train = dataset_train.take(args.train_batches_per_epoch)
        
        recognizer.fit(dataset_train,
            epochs=1,
            verbose='auto',
            validation_data = dataset_val,
            steps_per_epoch = args.train_batches_per_epoch,
        )
        if args.save_recognizer: 
            recognizer.save_model(f'{args.save_recognizer}_{epoch}.h5')

def main():
    args = parser.parse_args([] if "__file__" not in globals() else None)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    pos_recognizer = PositionRecognizer()
    if args.load_recognizer != "": 
        pos_recognizer.load_model(args.load_recognizer)
        print(f'Recognizer loaded')


    pos_recognizer.compile(
        optimizer=optimizers.AdamW(),
        loss=losses.CategoricalCrossentropy(),
        metrics=[
            metrics.CategoricalAccuracy(name="Square accuracy"), 
            BoardAccuracy(name="Boad accuracy")
        ]
    )
    pos_recognizer.summary()
    fit_recognizer(args, pos_recognizer)

if __name__ == '__main__':
  main()
