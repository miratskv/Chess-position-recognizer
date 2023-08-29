import tensorflow as tf
import numpy as np 


def bbox_ts_to_yxyx(boxes):
    t_y, t_x, t_h, t_w = tf.unstack(boxes, axis=-1)
    h = tf.exp(t_h) 
    w = tf.exp(t_w) 
    y = t_y + 1/2
    x = t_x + 1/2
    boxes = tf.stack([y, x, y+h, x+w], axis=-1)
    return boxes

def bbox_yxyx_to_ts(boxes):
    y, x, y2, x2 = tf.unstack(boxes, axis=-1)
    t_h = tf.math.log(y2-y)
    t_w = tf.math.log(x2-x)
    t_y = y - 1/2
    t_x = x - 1/2
    boxes = tf.stack([t_y, t_x, t_h, t_w], axis=-1)
    return boxes


class RecognizerBase(tf.keras.Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def save_model(self, save_path):
        self.save_weights(save_path, save_format="h5")

    def load_model(self, load_path):
        self.load_weights(load_path) 

    def predict(self, inputs):
        predicitons = self(inputs, training = False)
        predicitons = tf.math.argmax(predicitons, axis=-1)
        return predicitons


### TODO: This is basically just a playground at this point and can be fully ignored
class DetectorBase(tf.keras.Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def save_model(self, save_path):
        self.save_weights(save_path, save_format="h5")

    def load_model(self, load_path):
        self.load_weights(load_path)

    def add_rel_coordinates(self, coordinates_outer, coordinates_inner):
        y_ = coordinates_outer[...,0] + (coordinates_outer[...,2] - coordinates_outer[...,0])*coordinates_inner[..., 0]
        x_ = coordinates_outer[...,1] + (coordinates_outer[...,3] - coordinates_outer[...,1])*coordinates_inner[..., 1]

        y_2 = y_ + (coordinates_outer[...,2] - coordinates_outer[...,0])*(coordinates_inner[...,2] - coordinates_inner[...,0])
        x_2 = x_ + (coordinates_outer[...,3] - coordinates_outer[...,1])*(coordinates_inner[...,3] - coordinates_inner[...,1])

        return tf.stack([y_, x_, y_2, x_2], axis=-1)

    def crop_with_tolerance(self, images, y_tol, x_tol, H, W):
        boxes = self(images)
        boxes = bbox_ts_to_yxyx(boxes)

        y_tol = y_tol * (boxes[...,2] - boxes[...,0])
        x_tol = x_tol * (boxes[...,3] - boxes[...,1])
        y_ = boxes[...,0] - y_tol
        x_ = boxes[...,1] - x_tol
        y_2 = boxes[...,2] + y_tol
        x_2 = boxes[...,3] + x_tol

        boxes = tf.stack([y_, x_, y_2, x_2], axis=-1)
        cropped_images = tf.image.crop_and_resize(images, boxes, box_indices=tf.range(tf.shape(images)[0]), crop_size=[H, W])
        return cropped_images, boxes

    def predict(self, images):
        # Get bounding boxes from the model
        H, W = tf.shape(images)[1], tf.shape(images)[2]
        H = tf.cast(H, dtype=tf.float32)
        W = tf.cast(W, dtype=tf.float32)
        
        cropped_images, coordinates_1 = self.crop_with_tolerance(images, 1.05/8, 1.05/8, H, W)
        cropped_images, coordinates_2 = self.crop_with_tolerance(cropped_images, 1/8, 1/8, H, W)
        cropped_images, coordinates_3 = self.crop_with_tolerance(cropped_images, 0.5/8, 0.5/8, H, W)
        cropped_images, coordinates_4 = self.crop_with_tolerance(cropped_images, 0*0.2/8, 0*0.2/8, H, W)

        final_coordinates = coordinates_2
        final_coordinates = self.add_rel_coordinates(coordinates_3, final_coordinates)
        final_coordinates = self.add_rel_coordinates(coordinates_2, final_coordinates)
        final_coordinates = self.add_rel_coordinates(coordinates_1, final_coordinates)

        cropped_images = tf.image.crop_and_resize(images, final_coordinates, box_indices=tf.range(tf.shape(images)[0]), crop_size=[H, W])

        return cropped_images, final_coordinates  
