# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import pandas as pd
from env import *

df = pd.read_csv('/home/ndong/tb/cxr/processed/shenzhen.dev.tsv', sep='\t')
df = df[['path', 'label','tb']]
df = df.sort_values(['path'])

df.to_csv('processed/shenzhen.dev.csv', index=False)
