# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import cv2
import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import util
from env import *

# feature params
read_size = (400, 400)  # height, width
mid_ratio = 0.5 # ratio for the mid for right lung
margin = 0

# network params
batch_size = 32
num_epochs = 500
nf = 8 # number of filters
learning_rate = 0.001 # initial learning rate

exp_name = 'rpn_exp_01'
load_exp_name = ''


def load_data():
	_file = os.path.join(proc_path, 'shenzhen.dev.csv')
	df = pd.read_csv(_file)
	print('read', _file)

	_paths = df['path'].values
	_labels = df['label'].values

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

	return train_paths, test_paths, train_labels, test_labels


def load_lung_masks(paths):
    right_mask_paths = [os.path.join(right_mask_dir, p) for p in paths]
    left_mask_paths = [os.path.join(left_mask_dir, p) for p in paths]

    return right_mask_paths, left_mask_paths



def get_mid(contour, top_margin=mid_ratio):
    """
    Return the y value for the horizontal line that's mid_ratio between top and bottom
    (closer to top).
    """
    top = np.amax(contour, axis=0)[1]
    bot = np.amin(contour, axis=0)[1]
    return int((top + bot) / 2)


def get_mask(mask_paths):
    masks = util.read_mask(mask_paths, read_size)
    return masks


def get_contour(mask):
	_, contours, _ = cv2.findContours(np.copy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	assert len(contours) == 1, 'more than one contour'
	return np.squeeze(contours[0])


def add_margin(mask, margin):
	kernel = np.array([[0,1,0], [1,1,1], [0,1,0]])
	return cv2.dilate(mask, kernel, iterations = margin)


def get_submask(mask, region = 'lower')

def get_ground_truth(r_mask, l_mask, labels, margin = 0):
	gt = np.zeros(r_mask.shape)
	for i, label in enumerate(labels):
		if label in {'bilateral', 'normal', 'tb'}:
			mask = r_mask + l_mask
		elif label == 'l':
			mask = np.copy(l_mask)
		elif label == 'r':
			mask = np.copy(r_mask)
		elif label == 'll':
			mask = np.copy(l_mask)
			mask = get_submask(mask, 'lower')
		elif label == 'rl':
			mask = np.copy(r_mask)
			mask = get_submask(mask, 'lower')
		

		if margin > 0:
			mask = add_margin(mask, margin)
		gt[i,:,:,0] = mask

	return gt
			



def main():
	train_paths, test_paths, train_labels, test_labels = load_data()
	print('num_train:', len(train_paths), 'num_test:', len(test_paths))

    img_paths_train = [os.path.join(png_dir, p) for p in train_paths]
    img_paths_test = [os.path.join(png_dir, p) for p in test_paths]

    r_mask_paths_train, l_mask_paths_train = load_lung_masks(train_paths)
    r_mask_paths_test, l_mask_paths_test = load_lung_masks(test_paths)
    
    X_train = util.read_png(img_paths_train, read_size)
    X_test = util.read_png(img_paths_test, read_size)
    X_train = util.standardize(X_train)
    X_test = util.standardize(X_test)

    r_mask_train = get_mask(r_mask_paths_train)
    l_mask_train = get_mask(l_mask_paths_train)
    r_mask_test = get_mask(r_mask_paths_test)
    l_mask_test = get_mask(l_mask_paths_test)

    # ground truth
    ground_truth_train = get_ground_truth(r_mask_train, l_mask_train, train_labels)
    ground_test_train = get_ground_truth(r_mask_test, l_mask_test, test_labels)

    


    # test_builder = build_patches(X_test, test_labels, l_mask_test, 
    #     r_mask_test, test_paths)
    # x_patch_test, x_lung_test, location_info_test, y_test = \
    #     test_builder.get_patches()
    # test_manager = test_builder.get_patient_manager()
    # print('test patch ratio', float(np.sum(y_test))/ len(y_test))

    # print('num_test_patches:', x_patch_test.shape[0])
    # print('y_test guess:', util.guess_acc(y_test))

    # x_patch_ = tf.placeholder(tf.float32, [None, crop_size[0], crop_size[1], 1])
    # x_lung_ = tf.placeholder(tf.int32, [None, 3])
    # x_location_ = tf.placeholder(tf.float32, [None, 2])
    # y_ = tf.placeholder(tf.int32, [None, ])
    # model = Model(x_patch_, x_lung_, x_location_, y_, nf)

    # train_ce_var, train_rl_var, train_loss_var, train_acc_var, train_prob_var \
    #     = model.build_train_model()
    # test_ce_var, test_rl_var, test_loss_var, test_acc_var, test_prob_var \
    #     = model.build_test_model()

    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_loss_var)
    # print('num_params total:', util.get_num_model_params())

    # timer = util.Timer()
    # init = tf.global_variables_initializer()

    # saver = tf.train.Saver(max_to_keep=3)
    # model_dir = os.path.join(k_model_path, exp_name)
    # print('model_dir:', model_dir)
    # resume = load_exp_name != ''
    # load_dir= os.path.join(k_model_path, load_exp_name)
    # print('-'*10, exp_name, '-'*10)

    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)

    # with tf.Session() as sess:
    #     sess.run(init)

    #     if resume:
    #         checkpoint = tf.train.get_checkpoint_state(load_dir)
    #         if checkpoint and checkpoint.model_checkpoint_path:
    #             print('loading model from', load_dir)
    #             saver.restore(sess, checkpoint.model_checkpoint_path)

    #     for e in range(num_epochs):
    #     # for e in range(1):
    #         print('epoch: %d' %(e))
    #         train_builder = build_patches(X_train, train_labels, l_mask_train, r_mask_train, train_paths)
    #         x_patch_train, x_lung_train, location_info_train, y_train = train_builder.get_patches()

    #         # print('y labels: ', y)
    #         num_patches = x_patch_train.shape[0]
    #         train_manager = train_builder.get_patient_manager()

    #         num_batches = int(math.ceil(float(num_patches) / batch_size))

    #         train_loss_total = 0
    #         train_acc_total = 0
    #         train_prob_all = 0

    #         _idx = range(num_patches)

    #         for b in range(num_batches):
    #             start = b * batch_size
    #             end = min(start + batch_size, num_patches)
    #             # print('start-end', start, end)
    #             batch_idx = _idx[start:end]

    #             _, train_ce, train_rl, train_loss, train_acc, train_prob =  sess.run([optimizer, train_ce_var, train_rl_var, train_loss_var, train_acc_var, train_prob_var], 
    #                 feed_dict={x_patch_: x_patch_train[batch_idx],
    #                 x_lung_: x_lung_train[batch_idx], 
    #                 x_location_: location_info_train[batch_idx],
    #                 y_: y_train[batch_idx]})
                
                
    #             train_prob = np.squeeze(train_prob)

    #             if b == 0:
    #                 train_prob_all = train_prob
    #             else:
    #                 train_prob_all = np.concatenate((train_prob_all, train_prob))
                
    #             train_loss_total += train_loss
    #             train_acc_total += train_acc

    #         train_manager.get_pred_patch_prob(train_prob_all,e)
    #         train_patient_acc = train_manager.get_patient_acc()

    #         test_ce, test_rl, test_loss, test_acc, test_prob = sess.run(
    #             [test_ce_var, test_rl_var, test_loss_var, test_acc_var, 
    #             test_prob_var], feed_dict={x_patch_: x_patch_test,
    #             x_lung_: x_lung_test, x_location_: location_info_test,
    #             y_: y_test})

    #         test_manager.get_pred_patch_prob(test_prob,e)
    #         test_patient_acc = test_manager.get_patient_acc()
    #         test_patient_confusion_matrix = test_manager.get_patient_confusion_matrix()
    #         f1 = test_manager.get_f1_score()

    #         print('train loss:', round(train_loss_total/num_batches, 6),
    #             'test loss:', round(test_loss, 6),
    #             '\ntrain_acc:', round(float(train_acc_total)/num_patches , 6),
    #             'test_acc:', round(float(test_acc)/x_patch_test.shape[0], 6),
    #             '\ntrain_patient_acc:', train_patient_acc,
    #             'test_patient_acc:', test_patient_acc)
    #         print('test confusion matrix\n',
    #             test_patient_confusion_matrix)
    #         print('f1', f1)


if __name__ == '__main__':
    main()