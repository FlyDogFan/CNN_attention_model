# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import tflearn.initializations as tfi
import tflearn
import util
import numpy
from base_model import BaseModel

logger = util.get_logger()

class Model(BaseModel):
	def __init__(self, x_, y_):
		"""
		x_: tf.placeholder(tf.float32, [None, image-dim for train])
		y_: [None, num_fg_classes, 4] np.float array. num_fg_classes = 3 for
		num foreground classes
		"""
		super(Model, self).__init__()
		self.__dict__.update(locals())
		self.num_fg_classes = self.y_.get_shape().as_list()[1]
		print('num_fg_classes:', self.num_fg_classes)

	def build_train_model(self):
		logit = self._build_model(is_train=True)
		return self.loss(logit)

	def build_test_model(self):
		logit = self._build_model(is_train=False)
		return self.loss(logit)

	def _build_model(self, is_train):
		"""
		Return logit of shape [None, self.num_fg_classes, 4].
		"""
		if is_train:
			print('Building train model')
		else:
			print('Building test model')

		l2 = 0.0001
		nf = 8
		with tf.variable_scope('net', reuse=not is_train):
			self._zero_counts()
			x = self.x_
			for i in xrange(4):
				x = self._resblock(x, {'num_filters': (2**i)*nf, 'filter_size': 3,
						'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})
				# x = self._conv_relu_bn(x, {'num_filters': nf, 'filter_size': 3,
				#			'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})
				# x = self._conv_relu_bn(x, {'num_filters': nf, 'filter_size': 3,
				#			'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})
				x = self._pool(x, {'filter_size': 2, 'mode': 'avg'})

			# Conv FC layer
			x = self._resblock(x, {'num_filters': (2**3)*nf, 'filter_size': 1,
					'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})

			for i in range(2):
				x = self._resblock(x, {'num_filters': (2**3)*nf, 'filter_size': 3,
						'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})
				x = self._resblock(x, {'num_filters': (2**3)*nf, 'filter_size': 1,
						'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})

			x = self._conv(x, {'num_filters': self.num_fg_classes, 'filter_size': 1,
					'stride': 1, 'padding': 'same', 'l2': l2})
			x = self._flatten(x, {})
			# 4 coordinates (x_center, y_center, height, width) for each class.
			num_outputs = self.num_fg_classes * 4
			x = self._fully_connected(x, {'num_outputs': num_outputs, 'l2': l2})
			x = tf.reshape(x, [-1, self.num_fg_classes, 4])
			print('after conv FC:', x.get_shape().as_list())
		return x

	def loss(self, logits):
		"""Calculate the loss from the logits and the labels.
		Args:
			logits: tensor, float - [batch_size, self.num_fg_classes, 4].
					Use vgg_fcn.up as logits.
			bboxes: same shape as logits. Ground truth
		Returns:
		"""
		l2_loss = tf.nn.l2_loss(logits - self.y_)

		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		loss = tf.add_n(reg_losses) + l2_loss

		return loss, logits
