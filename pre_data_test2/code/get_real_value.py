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
# 从文件读取各个维度的属性
# 返回，一个二维list，每行代表一个维度的属性list
def read_from_dim_file():
  attributes_of_all_dims = []
  path = '../dims'
  dims = ['i', 'e', 'c', 'p', 'l']
  for dim in dims:
    abspath = os.path.join(os.path.abspath(path), dim+'.csv')
    try:
      attributes = pd.read_csv(abspath, header=None, sep=',')
      existed_attributes_list = list(attributes.iloc[0:,0].values)
    except:
      print("\nERROR: file doesn't have atts or open file failed\n")
      attributes = pd.DataFrame()
      existed_attributes_list = list()
    attributes_of_all_dims.append(existed_attributes_list)
  return attributes_of_all_dims

# 获得某个异常时刻、特定属性组合对应的的实际PV值
# 参数：异常时间戳、属性组合的列表['i', 'e', 'c', 'p', 'l']: [1,2,3,6,3] 或 ['*',2,'*',6,3]
# 返回组合对应的PV值，没记录到该组合则返回0
def get_real_PV_value(abnormal_time, combination):
  path = '../data_test2_modified'
  abspath = os.path.join(os.path.abspath(path), abnormal_time+'.csv')
  
  # test eg. abspath = abnormal_time
  df = pd.read_csv(abspath, header=None, sep=',', names=['i','e','c','p','l','PV'])

  #找出声明了具体属性的所有列的下标，并在这些列属性上对df进行筛选
  specific_col_list = [i for i in range(len(combination)) if type(combination[i]) == type(1)]
  for col in specific_col_list:
    df = df[df[list(df.columns)[col]] == combination[col]]
  
  if not df.empty:
    PV = df.apply(sum)['PV']
  else:
    PV = 0

  return PV

# 找出一个具体组合所对应的所有带*组合
# 输入不含*的组合 (6,2,3,7,9)
# 返回32个组合，其中31个为带*的组合
def get_all_combinations_of_a_concrete_combination(i, e, c, p ,l):
  # 输入 (6,2,3,7,9)
  origin = [i,e,c,p,l]

  # 其中*对应每个维度属性分别是149,14,9,36,5
  star = [149,14,9,36,5]
  
  # 返回的组合，先初始化带0个*和5个*的组合
  combinations = [[i, e, c, p ,l], [149,14,9,36,5]] # 自己和(*,*,*,*,*)

  # 找出每个组合中*的下标，再把原组合对应下标的属性替换为*
  star_indexs_combination = []
  for num_star in [1,2,3,4]:
    star_indexs_combination += its.combinations([0,1,2,3,4], num_star)
  #print(len(star_indexs_combination), star_indexs_combination)

  # 把组合中对应下标的属性替换为*
  for item in star_indexs_combination: # 如item 为（0,1），则(6,2,3,7,9)变为(*,*,3,7,9)，即(149,14,3,7,9)
    combinations.append([origin[i] if i not in item else star[i] for i in range(5)])
  
  #print(len(combinations), combinations)
  return combinations

# 生成5维的PV数组,把unknown属性用0代替,把*也考虑进去，另外数组下标相当于属性，因此数组中会多出几个不存在的属性，但不影响存取
# 参数：时间戳
# 返回一个五维数组 150*15*10*37*6
def generate_abnormal_time_table(abnormal_time):
  time = str(abnormal_time)
  # 读取df
  path = '../data_test2_modified'
  abspath = os.path.join(os.path.abspath(path), time+'.csv')
  df = pd.read_csv(abspath, header=None, sep=',', names=['i','e','c','p','l','PV'])
 
  # 450w元素的PV数组，下标即属性(因为属性缺失，一些下标是多出来的，其中最后一个下标代表属性*)，例如访问PV（2，3，*，1，*）即PV_table[2,3,9,1,5]
  PV_array = np.zeros((150,15,10,37,6))#np.array(list([0.0 for i in range(150) for e in range(15) for c in range(10) for p in range(37) for l in range(6)])).reshape((150,15,10,37,6))
  # print(PV_array.shape, type(PV_array))
  
  # 遍历3w行数据以对PV数组具体的属性组合项赋值
  for indexs in df.index:
    #print(df.loc[indexs].values[0:6])
    row_value = df.loc[indexs].values[0:6]
    # 组合转为PV数组下标，注意float属性转int
    i = int(row_value[0]) if row_value[0] != '*' else 149
    e = int(row_value[1]) if row_value[1] != '*' else 14
    c = int(row_value[2]) if row_value[2] != '*' else 9
    p = int(row_value[3]) if row_value[3] != '*' else 36
    l = int(row_value[4]) if row_value[4] != '*' else 5
    #print(i,e,c,p,l)

    # 把该组合对应的所有带*组合及其本身都赋值一遍
    combinations = get_all_combinations_of_a_concrete_combination(i, e, c, p ,l)
    for item in combinations:
      PV_array[item[0], item[1], item[2], item[3], item[4]] += row_value[-1]
  
  # print(PV_array[149,14,9,36,5])

  # 存储
  # print("-----------------SAVING TABLE FILE---------------")
  np.save(file="../Abnormalytime_real_PV_table/"+time+".npy", arr=PV_array)
  # print("------------------------SAVED--------------------")
  # return PV_array 多进程别返回数组，超过2.6G map返回奔溃

# 从文件获取异常时刻的列表，注意看第一行是列名还是数据，异常时刻是int类型
def get_abnormal_times_list(abnormal_time_file):
  times = pd.read_csv(abnormal_time_file, header=None, sep=',')
  times_list = list(times.iloc[0:,0].values)
  print(len(times_list), times_list)
  return times_list

# 属性转为数组下标
def attribute_to_index(attribute):
  if attribute == 'i':
    pass

# 数组下标转回属性
def index_to_attribute(attribute):
  pass

# 获取目录下所有的文件名（去除后缀）,所有时刻是string类型
def get_filename_list_from_dir(files_dir):
  files = os.listdir(files_dir)
  time_list = [f[0:-4] for f in files]
  # print(type(time_list), len(time_list))
  return time_list

def generate_all_time_table(one_of_alltime):
  try:
    time = str(one_of_alltime)
    # 读取df
    path = '../data_test2_modified'
    abspath = os.path.join(os.path.abspath(path), time+'.csv')
    df = pd.read_csv(abspath, header=None, sep=',', names=['i','e','c','p','l','PV'])
  
    # 450w元素的PV数组，下标即属性(因为属性缺失，一些下标是多出来的，其中最后一个下标代表属性*)，例如访问PV（2，3，*，1，*）即PV_table[2,3,9,1,5]
    PV_array = np.zeros((150,15,10,37,6))#np.array(list([0.0 for i in range(150) for e in range(15) for c in range(10) for p in range(37) for l in range(6)])).reshape((150,15,10,37,6))
    # print(PV_array.shape, type(PV_array))
    
    # 遍历3w行数据以对PV数组具体的属性组合项赋值
    for indexs in df.index:
      #print(df.loc[indexs].values[0:6])
      row_value = df.loc[indexs].values[0:6]
      # 组合转为PV数组下标，注意float属性转int
      i = int(row_value[0]) if row_value[0] != '*' else 149
      e = int(row_value[1]) if row_value[1] != '*' else 14
      c = int(row_value[2]) if row_value[2] != '*' else 9
      p = int(row_value[3]) if row_value[3] != '*' else 36
      l = int(row_value[4]) if row_value[4] != '*' else 5
      #print(i,e,c,p,l)

      # 把该组合对应的所有带*组合及其本身都赋值一遍
      combinations = get_all_combinations_of_a_concrete_combination(i, e, c, p ,l)
      for item in combinations:
        PV_array[item[0], item[1], item[2], item[3], item[4]] += row_value[-1]
    
    # 存储
    # print("-----------------SAVING TABLE FILE---------------")
    np.save(file="../Alltime_real_PV_table/"+time+".npy", arr=PV_array)
    # print("------------------------SAVED--------------------")
    # return PV_array
  except:
    print('ERRPR',time)

if __name__ == "__main__":

  # 多进程生成异常时刻PV值表
  # start0 = timeit.default_timer()
  # abnormal_times = get_abnormal_times_list('../Anomalytime_test2.csv') # [1538176200000,1538464200000,1538388000000,1538402700000] 
  # pool = Pool(12)
  # results0 = pool.map(generate_abnormal_time_table, abnormal_times)
  # end0 = timeit.default_timer()

  # 单进程生成一个异常时刻PV值表
  # start1 = timeit.default_timer()
  # generate_abnormal_time_table(1538150400000)
  # end1 = timeit.default_timer()
  

  # 多进程生成所有时刻PV值表
  start2 = timeit.default_timer()
  all_times = get_filename_list_from_dir('../data_test2_modified')# [1538176200000,1538464200000,1538388000000,1538402700000] 
  pool = Pool(12)
  results2 = pool.map(generate_all_time_table, all_times)
  end2 = timeit.default_timer()

  # 单进程生成一个时刻PV值表
  # generate_all_time_table(1539359700000)

  # print('multi ABNORMAL:',end0-start0)
  #print('original ABNORMAL:',end1-start1)
  print('multi ALL:',end2-start2)

  # get_filename_list_from_dir('../data_test2')
