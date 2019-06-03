import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt	
import os
from multiprocessing import Pool # 引用进程池
import time
import timeit

# -------小工具函数---------
# 检查缺失数据
def check_missing_data(df):
  # check for any missing data in the df (display in descending order)
  return df.isnull().sum().sort_values(ascending=False)

# 删除列中的字符串
def remove_col_str(df):
  # remove a portion of string in a dataframe column - col_1
  df['col_1'].replace('\n', '', regex=True, inplace=True)

  # remove all the characters after &# (including &#) for column - col_1
  df['col_1'].replace(' &#.*', '', regex=True, inplace=True)

# 删除列中的空格
def remove_col_white_space(df):
  # remove white space at the beginning of string 
  df[col] = df[col].str.lstrip()
# ---------------

# 删掉字母和unknown,保存到data_test3_modified
def deal_with_char_and_unknown(filepath):
  print(filepath)
  df=pd.read_csv(filepath, header=None, sep=',', names=['i','e','c','p','l','PV']) 
    
  df.replace('unknown', 0, inplace=True) #原地替换为0
  df.replace('i|e|c|p|l', '', regex=True, inplace=True)
  df = pd.DataFrame(df, dtype=np.float) #全部转为float类型
  save_path = os.path.join('../data_test3_modified',os.path.basename(filepath))
  df.to_csv(save_path, header=False, index=False)

# 处理某个时刻数据时，根据已存在的属性，对某个维度的属性去重
def deal_with_a_dim(df, dim):
  # 从对应维度的文件中读取已经存在的属性值
  path = '../dims'
  abspath = os.path.join(os.path.abspath(path), dim+'.csv')
  try:
    attributes = pd.read_csv(abspath, header=None, sep=',')
    existed_attributes_set = set(attributes.iloc[0:,0].values)
  except:
    attributes = pd.DataFrame()
    existed_attributes_set = set()
  
  # 当前时刻的文件中的属性值
  curr_attributes_set = set(df[dim].value_counts().index)

  # 两个集合合并
  combine_set = existed_attributes_set.union(curr_attributes_set)
  series = pd.Series(list(combine_set))

  # 把新的属性集合写回对应的维度属性文件
  series.to_csv(abspath, header=False, index=False)

# 统计各个维度属性个数
def count_attributes_of_dim(file):
  path = '../data_test3_modified'
  abspath = os.path.join(os.path.abspath(path), file)
  dims = ['i', 'e', 'c', 'p', 'l']
  df=pd.read_csv(abspath, header=None, sep=',', names=['i','e','c','p','l','PV'])
  df = pd.DataFrame(df, dtype=np.float) #全部转为float类型
  for dim in dims:
    # print(list(df[dim].value_counts().index))
    deal_with_a_dim(df, dim)

# 检测某个文件是否包含异常属性值
def detect_abnormal_attributes(file):
  path = '../data_test3_modified'
  abspath = os.path.join(os.path.abspath(path), file)
  dims = ['i', 'e', 'c', 'p', 'l']
  df=pd.read_csv(abspath, header=None, sep=',', names=['i','e','c','p','l','PV'])
  try:
    df = pd.DataFrame(df, dtype=np.float) #全部转为float类型
    dim_dict = {'i':147, 'e':13, 'c':9, 'p':35, 'l':5}
    for dim in dim_dict.keys():
      if dim_dict[dim] != len(list(df[dim].value_counts().index)):
        print(dim, len(list(df[dim].value_counts().index)), file)
  except:
    print(abspath, 'can not to float')

# 清洗data_test3目录下的数据，找出各维度的合理属性记录到文件(用0替换unknown)，并且把处理完的新数据放到data_test3_modified目录下
def clean_and_count_attributes():
  path = '../data_test3'
  files= os.listdir(path)
  for file in files:
    if not os.path.isdir(file):
      file_path = os.path.join(os.path.abspath(path), file)
      
      # replace char and 'unknown'
      print(file_path)
      df=pd.read_csv(file_path, header=None, sep=',', names=['i','e','c','p','l','PV']) 
      df.replace('unknown', 0, inplace=True) #原地替换unknown为0
      df.replace('i|e|c|p|l', '', regex=True, inplace=True)
      df = pd.DataFrame(df, dtype=np.float) #全部转为float类型

      # count attributes of each dim
      attributes_of_all_dims = []
      dims = ['i', 'e', 'c', 'p', 'l']
      for dim in dims:
        print(list(df[dim].value_counts().index))
        attributes_of_all_dims.append(list(df[dim].value_counts().index))
        deal_with_a_dim(df, dim)

      # save new data
      save_path = os.path.join('../data_test2_modified', os.path.basename(file_path))
      df.to_csv(save_path, header=False, index=False)
  return attributes_of_all_dims

# 用于多进程版本的目标函数，清洗data_test3目录下的数据，找出各维度的合理属性记录到文件(用0替换unknown)，并且把处理完的新数据放到data_test3_modified目录下
def multi_clean_and_count_attributes(file):
  path = '../data_test3'

  file_path = os.path.join(os.path.abspath(path), file)
  
  # 把维度字符和unknown替换掉
  print(file_path)
  df=pd.read_csv(file_path, header=None, sep=',', names=['i','e','c','p','l','PV']) 
  df.replace('unknown', 0, inplace=True) #原地替换unknown为0
  df.replace('i|e|c|p|l', '', regex=True, inplace=True)
  df = pd.DataFrame(df, dtype=np.float) #全部转为float类型

  # 计算每个维度的属性
  attributes_of_all_dims = []
  dims = ['i', 'e', 'c', 'p', 'l']
  for dim in dims:
    # print(list(df[dim].value_counts().index))
    attributes_of_all_dims.append(list(df[dim].value_counts().index))
    deal_with_a_dim(df, dim)

  # save new data
  save_path = os.path.join('../data_test3_modified', os.path.basename(file_path))
  df.to_csv(save_path, header=False, index=False)
  # return attributes_of_all_dims

# 从目录获取所有文件名(带后缀)，返回一个一维list
def get_file_list(path):
  file_list = os.listdir(path)
  # 分离文件名和后缀
  # for i in range(len(file_list)):
  #   file_list[i] = os.path.splitext(file_list[i])[0]
  # print(file_list)
  return file_list

if __name__=='__main__':

  # 检查每个处理后的文件的属性数据是否错误
  start = timeit.default_timer()
  path = '../data_test3_modified'
  files= os.listdir(path)
  pool = Pool(16)
  pool.map(detect_abnormal_attributes, files)
  # for file in files:
  #   if not os.path.isdir(file):
  #     detect_abnormal_attributes(file)
  end = timeit.default_timer()
  print(end-start)

  # 单进程统计属性
  # clean_and_count_attributes()

  # 多进程统计属性
 # start2 = timeit.default_timer()
  #files = get_file_list("../data_test3")
  #pool = Pool(16)
  #results = pool.map(multi_clean_and_count_attributes, files)
  #end2 = timeit.default_timer()
  #print(end2-start2)
  
  
  
