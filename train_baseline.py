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
import time
import util
from env import *
from models.model_baseline_02 import *
from sklearn.metrics import roc_auc_score

# feature params
read_size = (400, 400)  # height, width
margin = 0

# network params
batch_size = 32
num_epochs = 500
nf = 8 # number of filters
learning_rate = 0.001 # initial learning rate

threshold = 0. 5 # 0.5 default

exp_name = raw_input('exp_name: ')
exp_name = 'exp_baseline_' + exp_name
load_exp_name = raw_input('load_exp_name: ')


def load_lung_masks(paths):
    right_mask_paths = [os.path.join(right_mask_dir, p) for p in paths]
    left_mask_paths = [os.path.join(left_mask_dir, p) for p in paths]
    return right_mask_paths, left_mask_paths


def get_mask(mask_paths):
    masks = util.read_mask(mask_paths, read_size)
    return masks


def get_confusion(preds, labels):
    confusion = np.zeros((2,2))
    confusion[0][0] = np.sum((preds == 1) & (labels == 1))
    confusion[1][0] = np.sum((preds == 1) & (labels == 0))
    confusion[0][1] = np.sum((preds == 0) & (labels == 1))
    confusion[1][1] = np.sum((preds == 0) & (labels == 0))
    return confusion

def get_precision(confusion):
    return confusion[0][0]/(confusion[0][0] + confusion[1][0])


def get_sensitivity(confusion):
    return confusion[0][0]/(confusion[0][0] + confusion[0][1])


def get_f1_score(prec, recall):
    return 2 * (prec * recall) / (prec + recall)


def get_AUC(labels, scores):
    return roc_auc_score(labels, scores)


def prob2pred(probs):
    probs = np.squeeze(probs)
    preds = probs > threshold
    preds = preds.astype(np.int32)
    return preds


def get_ACC(preds, labels):
    return np.mean(preds == labels)


def main():
    train_paths, test_paths, train_labels, test_labels, y_train, y_test = util.load_data()
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

    X_train = util.blackout(X_train, r_mask_train, l_mask_train, margin)
    X_test = util.blackout(X_test, r_mask_test, l_mask_test, margin)

    x_ = tf.placeholder(tf.float32, [None, read_size[0], read_size[1], 1])
    y_ = tf.placeholder(tf.int32, [None, ])
    model = BaseModel(x_, y_, nf)

    train_ce_var, train_rl_var, train_loss_var, train_acc_var, train_prob_var = model.build_train_model()
    test_ce_var, test_rl_var, test_loss_var, test_acc_var, test_prob_var = model.build_test_model()

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_loss_var)
    print('num_params total:', util.get_num_model_params())

    timer = util.Timer()
    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=3)
    model_dir = os.path.join(model_path, exp_name)
    print('model_dir:', model_dir)
    resume = load_exp_name != ''
    load_dir= os.path.join(model_path, load_exp_name)
    print('-'*10, exp_name, '-'*10)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with tf.Session() as sess:
        sess.run(init)

        if resume:
            checkpoint = tf.train.get_checkpoint_state(load_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                print('loading model from', load_dir)
                saver.restore(sess, checkpoint.model_checkpoint_path)

        for e in range(num_epochs):
        # for e in range(1):
            print('epoch: %d' %(e))
            num_patches = X_train.shape[0]
            num_batches = int(math.ceil(float(num_patches) / batch_size))

            train_loss_total = 0
            train_acc_total = 0
            train_prob_all = 0

            _idx = range(num_patches)
            random.shuffle(_idx)

            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, num_patches)
                # print('start-end', start, end)
                batch_idx = _idx[start:end]

                _, train_ce, train_rl, train_loss, train_acc, train_prob =  sess.run([optimizer, train_ce_var, train_rl_var, train_loss_var, train_acc_var, train_prob_var], 
                    feed_dict={x_: X_train[batch_idx], y_: y_train[batch_idx]})

                train_prob = np.squeeze(train_prob)

                if b == 0:
                    train_prob_all = train_prob
                else:
                    train_prob_all = np.concatenate((train_prob_all, train_prob))
                
                train_loss_total += train_loss
                train_acc_total += train_acc

            test_ce, test_rl, test_loss, test_acc, test_prob = sess.run(
                [test_ce_var, test_rl_var, test_loss_var, test_acc_var, 
                test_prob_var], feed_dict={x_: X_test, y_: y_test})

            train_preds = prob2pred(train_prob_all)
            train_acc = get_ACC(train_preds, y_train[_idx])
            preds = prob2pred(test_prob)
            AUC = get_AUC(y_test, test_prob)
            confusion = get_confusion(preds, y_test)
            prec = get_precision(confusion)
            recall = get_sensitivity(confusion)
            f1 = get_f1_score(prec, recall)

            print('train loss:', round(train_loss_total/num_batches, 6),
                'test loss:', round(test_loss, 6),
                '\ntrain_acc:', round(train_acc, 6),
                'test_acc:', round(test_acc/X_test.shape[0], 6))
            print('AUC', AUC)
            print('confusion matrix\n', confusion)
            print('precision', prec)
            print('recall', recall)
            print('f1', f1)


if __name__ == '__main__':
    main()
