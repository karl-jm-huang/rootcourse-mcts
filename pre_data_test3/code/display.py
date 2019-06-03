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
import pywt

# 获取目录下所有的文件名（去除后缀）,所有时刻是int类型
def get_filename_list_from_dir(files_dir):
  files = os.listdir(files_dir)
  time_list = [int(f[0:-4]) for f in files]
  # print(type(time_list), len(time_list))
  return time_list

# 从文件获取异常时刻的列表，注意看第一行是列名还是数据，异常时刻是int类型
def get_abnormal_times_list(abnormal_time_file):
  times = pd.read_csv(abnormal_time_file, header=None, sep=',')
  times_list = list(map(int, list(times.iloc[0:,0].values)))
  # print(len(times_list), times_list)
  return times_list

# 展示某个组合整个周期的PV值 iecpl
def display_a_combination_period_original_value(combination):
  files_dir = "../Alltime_real_PV_table"
  files = os.listdir(files_dir)
  print(files)
  PV = []
  time = (np.array(get_filename_list_from_dir('../Alltime_real_PV_table')) - 1539360000000) / 300000#[i for i in range(len(files))]
  for file in files:
    abspath = os.path.join(os.path.abspath(files_dir), file)
    real_PV_array = np.load(file=abspath)
    # print(real_PV_array[combination[0],combination[1],combination[2],combination[3],combination[4]], file)
    PV.append(real_PV_array[combination[0],combination[1],combination[2],combination[3],combination[4]])
  
  PV_array = np.array(PV)
  np.save(file='../display_file/displayi6l2.npy', arr=PV_array)

  # PV2 = np.load('../display_file/displayi6l4.npy').tolist()
  # PV3 = np.load('../display_file/displayi6e10c5p19l3.npy').tolist()
  # PV4 = np.load('../display_file/displayi38e6c1p1l3.npy').tolist()

  # PV = np.load('../display_file/displayi6l4.npy').tolist()
  xt = [int((1539547500000 - 1539360000000) / 300000)]
  print(xt, PV[xt[0]])
  yt =  [PV[xt[0]]]
  plt.scatter(xt,yt,s=30,marker='o')

  # ab_time = ((np.array(get_abnormal_times_list('../Anomalytime_test3.csv')) - 1539360000000) / 300000).tolist()
  # x0 = list(map(int, ab_time))
  # # print(x0)
  # y0 = [PV[i] for i in x0]
  # plt.scatter(x0,y0,s=30,marker='o')
  # for x, y in zip([1395,1408,2706], [PV[1395],PV[1408],PV[2706]]):
  #   plt.text(x, y+0.3, '%.0f'%x, ha='center', va='bottom', fontsize=10.5)
  plt.plot(time, PV)
  # plt.plot(time, PV2)
  # plt.plot(time, PV3)
  # plt.plot(time, PV4)
  # plt.plot(time, wavelet_denoising(PV))
  # plt.plot(time, wavelet_denoising(PV))
  # plt.plot(time, smooth(PV, 6)) # 平滑后的真实值
  # plt.plot(time[1000:3000], smooth(PV, 6)[1000:3000]) # 平滑后的真实值
  # plt.plot(time[1000:3000], my_smooth_v2(smooth(PV, 6).tolist(), 6)[1000:3000])#做预测值
  # plt.plot(time[1000:3000], smooth(smooth(PV, 6).tolist(), 5)[1000:3000])
  # plt.plot(time, my_smooth(PV, 10))
  
  # plt.plot(time[600:1000], smooth_with_two_days(PV)[600:1000])
  # plt.plot(time, smooth_with_two_days(smooth(PV, 5).tolist()))
  # plt.plot(time[1000:3000], my_smooth(smooth(PV, 6).tolist(), 10)[1000:3000])
  # plt.plot(time[3000:4000], wavelet_denoising(smooth(smooth(PV, 10).tolist(), 5)[3000:4000]))
  plt.xlabel(u'时间戳', fontproperties='SimHei')
  plt.ylabel(u'PV值', fontproperties='SimHei')
  plt.grid(True)
  plt.show()

# 前十个时刻移动平均，效果不够好
def my_smooth(y, box_pts):
  y_smooth = y[0:box_pts]
  for i in range(box_pts, len(y)):
    y_smooth.append(np.mean(y[i-box_pts:i]))
  return y_smooth

# 前后各k个时刻平均
def my_smooth_v2(y, half_box_pts):
  y_smooth = y[0:half_box_pts]
  for i in range(half_box_pts, len(y)-half_box_pts):
    y_smooth.append((np.mean(y[(i-half_box_pts):i]) + np.mean(y[i+1:(i+half_box_pts+1)])) / 2)
  # print(len(y_smooth))
  y_smooth = y_smooth + y[(len(y)-half_box_pts):len(y)]
  return y_smooth  

# 移动平均..效果比小波好
def smooth(y, box_pts):
  box = np.ones(box_pts)/box_pts
  y_smooth = np.convolve(y, box, mode='same')
  return y_smooth

# 小波滤噪
def wavelet_denoising(data):
  # 小波函数取db4
  db4 = pywt.Wavelet('db4')
  # if type(data) is not types.NoneType:
  # 分解
  coeffs = pywt.wavedec(data, db4)
  # 高频系数置零
  coeffs[len(coeffs)-1] *= 0
  coeffs[len(coeffs)-2] *= 0
  # 重构
  meta = pywt.waverec(coeffs, db4)
  return meta

# 用前后一天的数据均值平滑
def smooth_with_two_days(y):
  print(type(y))
  res = []
  interval = 24 * 12
  for i in range(24*12, len(y)-24*12):
    mean = np.mean([y[i - interval], y[i + interval]])
    res.append(mean)
  res = y[0:interval] + res + y[len(y)-24*12: len(y)]#list(y[0:interval]).extend(res).extend(y[len(y)-24*12: len(y)])
  return res

# 用前后两天的左右5个时刻平滑后的均值
def smooth_with_two_days_five_times(y):
  pass


if __name__ == "__main__":
  display_a_combination_period_original_value([6,-1, -1, -1, 2])  #iecpl
  # print(wavelet_denoising([1,2,3,65,89,2,5]))
  # print(pywt.families()) 
  # print(get_filename_list_from_dir('../Alltime_real_PV_table'))
  # print((np.array(get_abnormal_times_list('../Anomalytime_test3.csv')) - 1539360000000) / 300000)
  # print(np.load(file='../Alltime_real_PV_table/1539888000000.npy')[6,10,5,19,3])