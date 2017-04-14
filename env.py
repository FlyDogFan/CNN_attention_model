# -*- coding: utf-8 -*-
import os


data_path = '/storage/ndong/data/cxr'

mask_dir = os.path.join(data_path, 'shenzhen/mask_s01_wingspan100_dong')
left_mask_dir = os.path.join(mask_dir, 'left_lung')
right_mask_dir = os.path.join(mask_dir, 'right_lung')

png_dir = os.path.join(data_path, 'shenzhen/CXR_png_800x800')

proj_path = os.path.dirname(os.path.abspath(__file__))
proc_path = os.path.join(proj_path, 'processed')
model_path = os.path.join(proj_path, 'model')