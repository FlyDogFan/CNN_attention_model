# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import os
import tensorflow as tf


def get_num_model_params():
	# Compute number of parameters in the model.
	num_params = 0
	from operator import mul
	for var in tf.trainable_variables():
		num_params += reduce(mul, var.get_shape().as_list(), 1)
	return num_params


def read_mask(paths, read_size):
	"""
	Read gif/png masks as 0/1 np.uint8 array [num_masks, height, width, 1].
	"""
	masks = np.zeros((len(paths), read_size[0], read_size[1], 1),
    	dtype=np.uint8)
	for i, p in enumerate(paths):
		img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
		# squeeze the third dimension if present.

    	if len(img.shape) > 2:
			img = img[:,:,0]

    	img = cv2.resize(img, dsize=read_size, interpolation=cv2.INTER_CUBIC)
    	img = mask_conversion(img, 1)
		masks[i,:,:,0] = img
		
	return masks


def mask_conversion(img, target):
	return img[np.nonzero(img)] == target


def standardize(imgs):
	"""
  	Normalize each img using the mean/std of each img.
  	imgs: [batch, height, width, 1] np.uint8 or np.float32
  	"""
	imgs = imgs.astype(np.float32)
	imgs -= np.mean(imgs)
	# imgs /= np.std(imgs)

	return imgs
