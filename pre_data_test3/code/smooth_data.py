import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt	
import os
from multiprocessing import Pool # 引用进程池
import time
import timeit

from get_real_value import *

def get_smooth_value(y, first_box_pts, second_box_pts):
  first_box = np.ones(first_box_pts)/first_box_pts
  y_smooth_first = np.convolve(y, first_box, mode='same')
  second_box = np.ones(second_box_pts)/second_box_pts
  y_smooth_second = np.convolve(y_smooth_first, second_box, mode='same')
  return y_smooth_second[int(first_box_pts/2)]

# 用前后5个时刻来平滑异常时刻
def smooth_a_time(time, half_window):
  path = '../data_test3_modified'
  file_path = os.path.join(os.path.abspath(path), str(time)+'.csv')

  with open(file_path, 'rt') as csvfile:
    reader = csv.reader(csvfile)
    smooth_column = [float(row[5]) for row in reader]

  print(smooth_column, len(smooth_column))

  group_time = []

  # 前k个时刻list
  path = "../data_test3_modified"
  count = half_window
  forecast_PV_array = np.zeros((150,15,10,37,6))
  while(count):
    before_time = time - count * 300 * 1000
    abspath = os.path.join(path, str(before_time)+".csv")
    print('before:',abspath)
    with open(abspath, 'rt') as csvfile:
      reader = csv.reader(csvfile)
      column = [float(row[5]) for row in reader]
      group_time.append(column)
    count -= 1
  
  group_time.append(smooth_column)
  
  # 后k个时刻list
  count = 1
  while(count <= half_window):
    after_time = time + count * 300 * 1000
    abspath = os.path.join(path, str(after_time )+".csv")
    print('after:',abspath)
    with open(abspath, 'rt') as csvfile:
      reader = csv.reader(csvfile)
      column = [float(row[5]) for row in reader]
      group_time.append(column)
    count += 1

  # print(len(group_time), len(group_time[0]), group_time[5][16338])

  # group_time list: 11 * 30011
  # print([lt[0] for lt in group_time], get_smooth_value([lt[0] for lt in group_time],10,5))

  res = [get_smooth_value([lt[i] for lt in group_time], 10, 5) for i in range(len(group_time[0]))]

  print(res, len(res))

  # 把原来csv文件的前五列属性和现在平滑过的一列PV组成新的csv文件
  data = pd.read_csv(os.path.join(path, str(time)+".csv"), header=None, sep=',', names=['i','e','c','p','l','PV'])
  data = data.assign(PV=pd.Series(res))
  to_path = '../data_test3_smooth'
  to_abspath = os.path.join(to_path, str(time)+".csv")
  data.to_csv(to_abspath, index=False, header=False)


# 移动平均
def get_smooth_value_fore_real(y, box_pts):
  box = np.ones(box_pts)/box_pts
  y_smooth = np.convolve(y, box, mode='same')
  return y_smooth[int(box_pts/2)]

def smooth_a_time_for_real(time):
  half_window = 3

  path = '../data_test3_modified'
  file_path = os.path.join(os.path.abspath(path), str(time)+'.csv')

  with open(file_path, 'rt') as csvfile:
    reader = csv.reader(csvfile)
    smooth_column = [float(row[5]) for row in reader]

  print(smooth_column, len(smooth_column))

  group_time = []

  # 前k个时刻list
  path = "../data_test3_modified"
  count = half_window
  forecast_PV_array = np.zeros((150,15,10,37,6))
  while(count):
    before_time = time - count * 300 * 1000
    abspath = os.path.join(path, str(before_time)+".csv")
    print('before:',abspath)
    with open(abspath, 'rt') as csvfile:
      reader = csv.reader(csvfile)
      column = [float(row[5]) for row in reader]
      group_time.append(column)
    count -= 1
  
  # 当前时刻
  group_time.append(smooth_column)
  
  # 后k个时刻list
  count = 1
  while(count <= half_window):
    after_time = time + count * 300 * 1000
    abspath = os.path.join(path, str(after_time )+".csv")
    print('after:',abspath)
    with open(abspath, 'rt') as csvfile:
      reader = csv.reader(csvfile)
      column = [float(row[5]) for row in reader]
      group_time.append(column)
    count += 1

  # print(len(group_time), len(group_time[0]), group_time[5][16338])

  # group_time list: 11 * 30011
  # print([lt[0] for lt in group_time], get_smooth_value([lt[0] for lt in group_time],10,5))

  res = [get_smooth_value_fore_real([lt[i] for lt in group_time], half_window*2+1) for i in range(len(group_time[0]))]

  print(res, len(res))

  # 把原来csv文件的前五列属性和现在平滑过的一列PV组成新的csv文件
  data = pd.read_csv(os.path.join(path, str(time)+".csv"), header=None, sep=',', names=['i','e','c','p','l','PV'])
  data = data.assign(PV=pd.Series(res))
  to_path = '../data_test3_smooth_real'
  to_abspath = os.path.join(to_path, str(time)+".csv")
  data.to_csv(to_abspath, index=False, header=False)

# 左右取平均作为预测值
def smooth_a_time_for_forecast(time, half_window):
  path = '../data_test3_modified'
  file_path = os.path.join(os.path.abspath(path), str(time)+'.csv')

  with open(file_path, 'rt') as csvfile:
    reader = csv.reader(csvfile)
    smooth_column = [float(row[5]) for row in reader]

  print(smooth_column, len(smooth_column))

  group_time = []

  # 前k个时刻list
  path = "../data_test3_modified"
  count = half_window
  forecast_PV_array = np.zeros((150,15,10,37,6))
  while(count):
    before_time = time - count * 300 * 1000
    abspath = os.path.join(path, str(before_time)+".csv")
    print('before:',abspath)
    with open(abspath, 'rt') as csvfile:
      reader = csv.reader(csvfile)
      column = [float(row[5]) for row in reader]
      group_time.append(column)
    count -= 1
  
  # 当前时刻
  # group_time.append(smooth_column)
  
  # 后k个时刻list
  count = 1
  while(count <= half_window):
    after_time = time + count * 300 * 1000
    abspath = os.path.join(path, str(after_time )+".csv")
    print('after:',abspath)
    with open(abspath, 'rt') as csvfile:
      reader = csv.reader(csvfile)
      column = [float(row[5]) for row in reader]
      group_time.append(column)
    count += 1

  # print(len(group_time), len(group_time[0]), group_time[5][16338])

  # group_time list: 11 * 30011
  # print([lt[0] for lt in group_time], get_smooth_value([lt[0] for lt in group_time],10,5))

  res = [np.mean([lt[i] for lt in group_time]) for i in range(len(group_time[0]))]

  print(res, len(res))

  # 把原来csv文件的前五列属性和现在平滑过的一列PV组成新的csv文件
  data = pd.read_csv(os.path.join(path, str(time)+".csv"), header=None, sep=',', names=['i','e','c','p','l','PV'])
  data = data.assign(PV=pd.Series(res))
  to_path = '../data_test3_smooth_forecast'
  to_abspath = os.path.join(to_path, str(time)+".csv")
  data.to_csv(to_abspath, index=False, header=False)

# 前后各k个时刻平均
def my_smooth_v2(y, half_box_pts):
  y_smooth = y[0:half_box_pts]
  for i in range(half_box_pts, len(y)-half_box_pts):
    y_smooth.append((np.mean(y[(i-half_box_pts):i]) + np.mean(y[i+1:(i+half_box_pts+1)])) / 2)
  # print(len(y_smooth))
  y_smooth = y_smooth + y[(len(y)-half_box_pts):len(y)]
  return y_smooth 

# 获取目录下所有的文件名（去除后缀）,所有时刻是int类型
def get_filename_list_from_dir(files_dir):
  files = os.listdir(files_dir)
  time_list = [int(f[0:-4]) for f in files]
  # print(type(time_list), len(time_list))
  return time_list

if __name__ == "__main__":
  # smooth_a_time(1539778500000)
  # abnormal_times = get_abnormal_times_list('../Anomalytime_test3.csv')
  # start0 = timeit.default_timer()
  # pool = Pool(4)
  # result = pool.map(smooth_a_time, abnormal_times)
  # end0 = timeit.default_timer()

  # print('multi:', end0 - start0)

  all_time_list = get_filename_list_from_dir('../data_test3_modified')
  # 平滑后的真实值
  pool = Pool(11)
  # smooth_a_time_for_real(1540171800000, 3)
  result = pool.map(smooth_a_time_for_real, all_time_list)

  # 预测值
  # smooth_a_time_for_forecast(1540171800000, 3)