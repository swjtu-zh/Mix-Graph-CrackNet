
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


smooth = 1e-15

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def loss(y_true, y_pred):
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    binary_loss = bce_loss(y_true, y_pred)
    dice_loss = 1.0 - dice_coef(y_true, y_pred)
    total_loss = binary_loss + 0.5*dice_loss
    return total_loss


def iou_eval(y_true,y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    val = (intersection + 1e-15) / (union + 1e-15)
    val = val.astype(np.float32)
    return val


def element_eval(y_true, y_pred):
    TP = (y_true * y_pred).sum()
    y_true_inverse = 1 - y_true
    FP = (y_true_inverse * y_pred).sum()
    y_pred_inverse = 1 - y_pred
    FN = (y_true * y_pred_inverse).sum()
    return TP, FP, FN



