import tensorflow as tf
import numpy as np
import chessrec.constants as const
from chessrec.models import bbox_ts_to_yxyx, bbox_yxyx_to_ts


class BoardAccuracy(tf.keras.metrics.Accuracy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis = -1)
        y_pred = tf.one_hot(y_pred, const.SQUARE_CLASSES, axis=-1)
        boards_equal = tf.math.equal(y_pred, y_true)
        boards_equal = tf.math.reduce_all(boards_equal, axis = (1,2,3))
        return super().update_state(tf.ones_like(boards_equal), boards_equal)


class IoU(tf.keras.metrics.Metric):
    def __init__(self, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert the bounding boxes from [y,x,h,w] format to [xmin, ymin, xmax, ymax] format
        box_pred = bbox_ts_to_yxyx(y_pred)
        box_true = bbox_ts_to_yxyx(y_true)

        # Calculate the intersection rectangle
        xmin_inter = tf.maximum(box_true[:,1], box_pred[:,1])
        ymin_inter = tf.maximum(box_true[:,0], box_pred[:,0])
        xmax_inter = tf.minimum(box_true[:,3], box_pred[:,3])
        ymax_inter = tf.minimum(box_true[:,2], box_pred[:,2])
        width_inter = tf.maximum(xmax_inter - xmin_inter + 1, 0)
        height_inter = tf.maximum(ymax_inter - ymin_inter + 1, 0)

        # Calculate the area of the intersection rectangle, and the area of each bounding box
        area_inter = width_inter * height_inter
        area_true = (box_true[:,2] - box_true[:,0] + 1) * (box_true[:,3] - box_true[:,1] + 1)
        area_pred = (box_pred[:,2] - box_pred[:,0] + 1) * (box_pred[:,3] - box_pred[:,1] + 1)

        # Calculate the union area
        area_union = area_true + area_pred - area_inter

        # Calculate the IoU
        iou = tf.where(tf.equal(area_union, 0), tf.zeros_like(area_union), area_inter / area_union)

        # Update the IoU and total counts
        self.iou.assign_add(tf.reduce_sum(iou))
        self.total.assign_add(tf.cast(tf.shape(iou)[0], self.dtype))

    def result(self):
        # Calculate the mean IoU
        return self.iou / self.total

    def reset_states(self):
        # Reset the accumulated counts
        self.iou.assign(0)
        self.total.assign(0) 
