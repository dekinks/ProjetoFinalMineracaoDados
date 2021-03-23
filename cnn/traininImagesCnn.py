import tensorflow as tf
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255
  input_mask = tf.cast(input_mask, tf.float32) / 255
  return input_image, input_mask



