# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tflearn
import tflearn.initializations as tfi
from util import *

"""
tensorflow version: 1.0.1
tflearn version: 0.3
"""
logger = get_logger()

class BaseModel(object):
    def __init__(self, x_, y_, nf_):
        """
        x_: tf.placeholder(tf.float32, [None, h, w, 1])
        y_: tf.placeholder(tf.int32, [None,]) 0/1 (no_tb, tb)
        nf_: number of filters
        """
        self.x_ = x_
        self.y_ = y_
        self.nf_ = nf_
        self.bns = []

    def build_train_model(self):
        logit = self._build_model(is_train=True)
        return self.loss(logit)

    def build_test_model(self):
        logit = self._build_model(is_train=False)
        return self.loss(logit)

    def _zero_counts(self):
        self.bn_count = 0
        self.conv_count = 0
        self.dropout_count = 0
        self.fc_count = 0
        self.resblock_count = 0
        self.resbottle_count = 0

    def _conv_relu_bn(self, x, c):
        x = self._conv(x, c)
        x = tf.nn.relu(x)
        return self._bn(x, c)

    def _conv_relu(self, x, c):
        x = self._conv(x, c)
        return tf.nn.relu(x)

    def _conv(self, x, c):
        if 'padding' not in c:
            c['padding'] = 'same'
        x = tflearn.layers.conv.conv_2d(x, c['num_filters'], c['filter_size'],
            c['stride'], activation='linear', regularizer='L2', weight_decay=c['l2'], weights_init=tfi.xavier(), padding=c['padding'], name='conv%d' %(self.conv_count))
        logger.info('conv%d output' %(self.conv_count), x.get_shape().as_list())
        self.conv_count += 1
        return tf.nn.relu(x)

    def _bn_relu_conv(self, x, c):
        x = self._bn(x, c)
        x = tf.nn.relu(x)
        return self._conv(x, c)

    def _shortcut(self, input, x, c):
        # Expand channels of shortcut to match residual. Stride appropriately to
        # match residual (width, height), which should be int if network
        # architecture is correctly configured.
        in_channels = input.get_shape().as_list()[-1]
        stride_width = input.get_shape().as_list()[1] // x.get_shape().as_list()[1]
        stride_height = input.get_shape().as_list()[2] // x.get_shape().as_list()[2]
        assert(stride_width == stride_height)

        # 1 X 1 conv if shape is different. Else identity.
        if stride_width != 1 or stride_height != 1 \
            or in_channels != c['num_filters']:
            c_1 = {'num_filters': c['num_filters'], 'filter_size': 1, \
                'stride': stride_width, 'l2': c['l2'], 'padding': c['padding']}
            input = self._conv(input, c_1)
        return x + input

    def _resblock(self, x, c):
        # This is an improved scheme proposed in
        # http://arxiv.org/pdf/1603.05027v2.pdf
        # c can contain an optional 'subsample' integer parameter to subsample in
        # the first conv layer.
        # Need same resolution for shortcut connection.
        assert 'padding' not in c or c['padding'] == 'same'
        c_subsample = c
        if 'subsample' in c:
            c_subsample['stride'] = c['subsample']
        x = self._bn_relu_conv(x, c_subsample)
        x = self._bn_relu_conv(x, c)
        x = self._shortcut(input, x, c)

        logger.info('resblock%d output' % self.resblock_count,
                x.get_shape().as_list())
        self.resblock_count += 1
        return x

    def _global_pool(self, x, c):
        assert c['mode'] in {'avg', 'max'}
        if c['mode'] == 'avg':
            return tflearn.layers.conv.global_avg_pool(x)
        else:
            return tflearn.layers.conv.global_max_pool(x)

    def _pool(self, x, c):
        assert c['mode'] in {'avg', 'max'}
        func = tflearn.layers.conv.max_pool_2d if c['mode'] == 'max' else \
            tflearn.layers.conv.avg_pool_2d
        return func(x, c['filter_size'])

    def _flatten(self, x):
        x = tflearn.layers.core.flatten(x)
        logger.info('flatten output', x.get_shape().as_list())
        return x

    def _bn(self, x, c):
        if c['is_train']:
            # Create the BN node the first time.
            self.bns.append(BatchNorm('bn%d' % self.bn_count))
        x = self.bns[self.bn_count](x, train=c['is_train'])
        self.bn_count += 1
        return x

        # Don't use tflearn's train/test control. They aren't reliable.
        #return tflearn.layers.normalization.batch_normalization(x)

    def _fully_connected(self, x, c):
        if 'activation' not in c:
            c['activation'] = 'linear'
        x = tflearn.layers.core.fully_connected(x, c['num_outputs'], activation=c['activation'], weight_decay=c['l2'], name='fully_connncted%d' %(self.fc_count))
        logger.info('fully_connected output', x.get_shape().as_list())
        self.fc_count += 1
        return x

    def _dropout(self, x, c):
        if c['is_train']:
            x = tf.nn.dropout(x, 1 - c['dropout_rate'])
            self.dropout_count += 1
        # Don't use tflearn's train/test control. They aren't reliable.
        #x = tflearn.layers.core.dropout(x, 1-c['dropout_rate'],
        #  name='dropout%d' % self.dropout_count)
        return x

    def _conv_transpose(self, x, c):
        strides = [1, c['stride'], c['stride'], 1]
        # output shape
        #new_shape = [c['shape'][0], c['shape'][1], c['shape'][2], c['num_classes']]
        #deconv = tf.nn.conv2d_transpose(x, weights, output_shape,
        #                                                                strides=strides, padding='SAME')
        # TODO(wdai): l2 regularization
        return tflearn.layers.conv.upscore_layer(x, c['num_classes'],
                shape=c['shape'], kernel_size=c['filter_size'], strides=strides)

    def _build_model(self, is_train):
        """
        Return logit of the model.
        """
        # Avoid creating new loss nodes in repeated _build_model() calls.
        #if is_train and hasattr(self, 'train_ops'):
        #  return self.train_ops
        #if not is_train and hasattr(self, 'test_ops'):
        #  return self.test_ops
        if is_train:
            print('Building train model')
        else:
            print('Building test model')

        l2 = 0.0001
        nf = self.nf_

        with tf.variable_scope('cnn', reuse=not is_train):
            self._zero_counts()
            x = self.x_
            x = self._conv_relu_bn(x, {'num_filters': nf, 'filter_size': 3,
                'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})
            x = self._conv_relu_bn(x, {'num_filters': nf, 'filter_size': 3,
                'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})
            x = self._pool(x, {'filter_size': 2, 'mode': 'max'})

            x = self._conv_relu_bn(x, {'num_filters': 2*nf, 'filter_size': 3,
                'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})
            x = self._conv_relu_bn(x, {'num_filters': 2*nf, 'filter_size': 3,
                'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})
            x = self._pool(x, {'filter_size': 2, 'mode': 'max'})

            x = self._conv_relu_bn(x, {'num_filters': 4*nf, 'filter_size': 3,
                'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})
            x = self._conv_relu_bn(x, {'num_filters': 4*nf, 'filter_size': 3,
                'stride': 1, 'padding': 'same', 'l2': l2, 'is_train': is_train})
            x = self._flatten(x)
            x = self._fully_connected(x, {'num_outputs': 1, 'l2': l2})
        return x

    def loss(self, logits):
        """Calculate the loss from the logits and the labels.

        Args:
            logits: tensor, float - [batch_size, 1].

        Returns:
            loss: Loss tensor of type float.
        """
        # Define loss
        print('logits shape', logits.get_shape().as_list())
        y = tf.cast(tf.expand_dims(self.y_, 1), tf.float32)
        print('y shape:', y.get_shape().as_list())
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits))
        tf.add_to_collection('losses', cross_entropy)
        c1 = tf.sigmoid(logits)
        c0 = 1 - c1
        prob = tf.concat([c0, c1], 1)
        #print('prob shape', prob.get_shape().as_list())

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        correct_pred = tf.nn.in_top_k(prob, self.y_, 1)
        total_reg_losses = tf.add_n(reg_losses)
        total_loss = total_reg_losses + cross_entropy

        # Evaluate model
        accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
        return cross_entropy, total_reg_losses, total_loss, accuracy, c1
