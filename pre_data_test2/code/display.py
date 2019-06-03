import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt	
import os
from multiprocessing import Pool 
import timeit
import time
import numba as nb
import itertools as its

# 展示某个组合整个周期的PV值
def display_a_combination_period_value(combination):
  files_dir = "../Alltime_real_PV_table"
  files = os.listdir(files_dir)
  PV = []
  time = [i for i in range(len(files))]
  for file in files:
    abspath = os.path.join(os.path.abspath(files_dir), file)
    real_PV_array = np.load(file=abspath)
    PV.append(real_PV_array[combination[0],combination[1],combination[2],combination[3],combination[4]])
  plt.plot(time, PV)
  plt.xlabel(u'时间戳', fontproperties='SimHei')
  plt.ylabel(u'PV值', fontproperties='SimHei')
  plt.grid(True)
  plt.show()

if __name__ == "__main__":
  display_a_combination_period_value([1,1,1,12,3])  