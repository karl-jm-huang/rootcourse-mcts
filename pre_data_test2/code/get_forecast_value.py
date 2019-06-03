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

from get_real_value import *

# 用异常前十组数据平均值作为异常时刻的预测值
def generate_forecast_table_for_abnormal_time(abnormally_time):
  path = "../Alltime_real_PV_table"
  count = 10
  forecast_PV_array = np.zeros((150,15,10,37,6))
  while(count):
    before_time = abnormally_time - count * 300 * 1000
    abspath = os.path.join(path,str(before_time)+".npy")
    #print(abspath)
    forecast_PV_array += np.load(file=abspath)
    count -= 1
  forecast_PV_array /= 10
  #print(forecast_PV_array)
  # 存储
  # print("-----------------SAVING TABLE FILE---------------")
  np.save(file="../Abnormalytime_forecast_PV_table/"+str(abnormally_time)+".npy", arr=forecast_PV_array)
  # print("------------------------SAVED--------------------")

if __name__ == "__main__":
  # generate_forecast_table_for_abnormal_time(1538153400000)
  start0 = timeit.default_timer()
  abnormal_times = get_abnormal_times_list('../Anomalytime_test2.csv')
  pool = Pool(2)
  results0 = pool.map(generate_forecast_table_for_abnormal_time, abnormal_times)
  end0 = timeit.default_timer()

  print('multi:', end0 - start0)