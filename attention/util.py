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


def get_grid_param(side_num_grid, read_size):
	"""
	num: number of grid on each side
	"""
	assert (read_size[0]%side_num_grid) == 0
	grid_size = int(read_size[0] // side_num_grid)
	num_grid = side_num_grid**2
	return grid_size, num_grid


def get_area(patch):
	patch_cp = np.squeeze(patch)
	assert len(patch_cp.shape) == 2
	return patch_cp.shape[0] * patch_cp.shape[1]


def get_patch_overlap(gt_patch):
	overlap = np.sum(gt_patch)
	area = get_area(gt_patch)
	return overlap / area


def get_overlap(gts, grid_size, side_num_grid):
	num_overlaps = gts.shape[0]
	overlaps = np.zeros((num_overlaps, side_num_grid, side_num_grid))
	for i, gt in enumerate(gts):
		gt = np.squeeze(gt)
		for j in range(side_num_grid):
			for k in range(side_num_grid):
				gt_patch = gt[(j*grid_size):((j+1)*grid_size), (k*grid_size):((k+1)*grid_size)]
				overlaps[i,j,k] = get_patch_overlap(gt_patch)

	return overlaps


