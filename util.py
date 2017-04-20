# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import time
from env import *


kernel = np.array([[0,1,0], [1,1,1], [0,1,0]])


def load_data():
	_file = os.path.join(proc_path, 'shenzhen.dev.csv')
	df = pd.read_csv(_file)
	print('read', _file)

	_paths = df['path'].values
	_labels = df['label'].values
	_y = df['tb'].values

	num_png = len(_paths)
	_index = range(num_png)
	random.seed(2017)
	random.shuffle(_index)
	train_index = _index[:int(num_png * 0.8)]
	test_index = _index[int(num_png * 0.8):]

	train_paths = _paths[train_index]
	test_paths = _paths[test_index]
	train_labels = _labels[train_index]
	test_labels = _labels[test_index]
	y_train = _y[train_index]
	y_test = _y[test_index]

	return train_paths, test_paths, train_labels, test_labels, y_train, y_test


def get_num_model_params():
	# Compute number of parameters in the model.
	num_params = 0
	from operator import mul
	for var in tf.trainable_variables():
		num_params += reduce(mul, var.get_shape().as_list(), 1)
	return num_params


def read_png(paths, read_size):
    """
    paths: list of path.
    return: np array [num_imgs, read_size[0], read_size[1], 1] of type uint8
        [0, 256).
    """
    imgs = np.zeros((len(paths), read_size[0], read_size[1], 1), dtype=np.uint8)
    for i, p in enumerate(paths):
        assert os.path.isfile(p), 'cannot open ' + p
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        res = cv2.resize(img, dsize=read_size, interpolation=cv2.INTER_CUBIC)
        imgs[i,:,:,0] = res
    return imgs


def read_mask(paths, read_size):
	"""
	Read gif/png masks as 0/1 np.uint8 array [num_masks, height, width, 1].
	"""
	masks = np.zeros((len(paths), read_size[0], read_size[1], 1),
    	dtype=np.uint8)
	for i, p in enumerate(paths):
		mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
		# squeeze the third dimension if present.

		if len(mask.shape) > 2:
			mask = mask[:,:,0]

		mask = cv2.resize(mask, dsize=read_size, interpolation=cv2.INTER_CUBIC)
		mask = mask_conversion(mask, 1)
		masks[i,:,:,0] = mask
		
	return masks


def mask_conversion(img, target):
	img[np.nonzero(img)] = target
	return img


def standardize(imgs):
	"""
  	Normalize each img using the mean/std of each img.
  	imgs: [batch, height, width, 1] np.uint8 or np.float32
  	"""
	imgs = imgs.astype(np.float32)
	imgs -= np.mean(imgs)
	# imgs /= np.std(imgs)
	return imgs


def blackout(imgs, right_masks, left_masks, margin):
	masks = right_masks + left_masks
	for i in xrange(masks.shape[0]):
		masks[i,:,:,0] = cv2.dilate(masks[i,:,:,0], kernel, iterations = margin)
		imgs[i][np.where(masks[i] == 0)] = 0

	return imgs


def add_margin(mask, margin):
	return cv2.dilate(mask, kernel, iterations = margin)


class BatchNorm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    """
    Normalize over all dim except the last.
    """
    def __init__(self, name, epsilon=1e-5, momentum=0.9):
        self.__dict__.update(locals())
        with tf.variable_scope(name):
            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                    initializer=tf.constant_initializer(0.), dtype=tf.float32)
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                    initializer=tf.random_normal_initializer(1., 0.02), dtype=tf.float32)

                norm_dims = range(len(shape) - 1)  # [0, 1, 2] for 2d convolution
                batch_mean, batch_var = tf.nn.moments(x, norm_dims, name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean = self.ema.average(batch_mean)
                self.ema_var = self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                        mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        #return tf.nn.batch_norm_with_global_normalization(
        #                x, mean, var, self.beta, self.gamma, self.epsilon,
        #                scale_after_normalization=True)
        return tf.nn.batch_normalization(x, mean, var, self.beta, self.gamma,
                self.epsilon)


class SmartLogger():
    """
    Simple wrapper around logger to allow print() style formatting (Unpack
    Argument List for Format String).

    logger = get_logger()
    logger.info('Output is', 3)
    """
    def __init__(self, logger):
        self.logger = logger

    def error(self, *args):
        return self.logger.error(' '.join([str(x) for x in args]))

    def error_if(self, cond, *args):
        if cond:
            return self.logger.error(' '.join([str(x) for x in args]))

    def info(self, *args):
        return self.logger.info(' '.join([str(x) for x in args]))

    def info_if(self, cond, *args):
        if cond:
            return self.logger.info(' '.join([str(x) for x in args]))

    def debug(self, *args):
        return self.logger.debug(' '.join([str(x) for x in args]))

    def debug_if(self, cond, *args):
        if cond:
            return self.logger.debug(' '.join([str(x) for x in args]))


def get_logger(filename=None):
    """
    Log to console and to file log/logging.INFO, which will be created /
    overwritten.

    Comment(wdai): Cannot let multiple logger log to the same file due to race.
    Thus each file needs to have its own output.

    Usage:
    >>> logger = get_logger()
    >>> logger.info('The answer is', 42)
    """
    import logging
    import inspect

    # Get caller's filename to use as log file name.
    frame = inspect.stack()[1]      # caller frame
    filename_with_extension = os.path.basename(inspect.getfile(frame[0]))
    filename = filename_with_extension.split('.')[0]
    log_dir = os.path.join(proj_path, "log")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # set up logging to file - see previous section for more details
    format_str = '[%(asctime)s %(levelname)s %(name)s] %(message)s'
    logging.basicConfig(level=logging.INFO, # Set logging level here!
                                            format=format_str,
                                            datefmt='%Y%m%d %H:%M:%S',
                                            filename=os.path.join(log_dir, filename + '.INFO'),
                                            filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # set a format which is simpler for console_handler use
    formatter = logging.Formatter(format_str, '%Y%m%d %H:%M:%S')
    # tell the handler to use this format
    console_handler.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger(filename).addHandler(console_handler)

    return SmartLogger(logging.getLogger(filename))


class Timer():
    def __init__(self):
        self.restart()

    def restart(self):
        self.start = time.time()

    def elapsed(self):
        return round(time.time() - self.start, 2)