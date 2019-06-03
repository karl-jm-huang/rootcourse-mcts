import csv
import os
import json
from ast import literal_eval
import pandas as pd
import numpy as np


# 对比两个result.csv文件,高分的集合在一起
def merge_result_v1(ab_timelist, one_result_csv_file, another_result_csv_file, merge_result_csv_file):
  one_df = pd.read_csv(one_result_csv_file, header=0, sep=',', names=['timestamp', 'set','score'])
  another_df = pd.read_csv(another_result_csv_file, header=0, sep=',', names=['timestamp', 'set', 'score'])
  merge_df = pd.DataFrame(columns=('timestamp', 'set'))

  rows = len(ab_timelist)
  for i in range(rows):
    time = ab_timelist[i]
    root_cause_set = ''
    # print(time, one_df[one_df['timestamp'] == time].values[0,2], another_df.loc[another_df['timestamp'] == time].values[0,2])
    if one_df[one_df['timestamp'] == time].values[0,2] > another_df[another_df['timestamp'] == time].values[0,2] and abs(one_df[one_df['timestamp'] == time].values[0,2] - another_df[another_df['timestamp'] == time].values[0,2]) >= 0.1:
      print(time, one_df[one_df['timestamp'] == time].values[0,2], another_df[another_df['timestamp'] == time].values[0,2])
      root_cause_set = (one_df[one_df['timestamp'] == time].values)[0,1]
    else:
      root_cause_set = (another_df[another_df['timestamp'] == time].values)[0,1]
    merge_df.loc[i] = [time, root_cause_set]
    
  merge_df.to_csv(merge_result_csv_file, index=False)

# 对比两个result.csv文件，把差值多于0.05得分高的结果整合到新文件merge_result.csv，带时间和组合
def merge_result_v2(ab_timelist, one_result_csv_file, another_result_csv_file, merge_result_csv_file):
  one_df = pd.read_csv(one_result_csv_file, header=0, sep=',', names=['timestamp', 'set','score'])
  another_df = pd.read_csv(another_result_csv_file, header=0, sep=',', names=['timestamp', 'set', 'score'])
  merge_df = pd.DataFrame(columns=('timestamp', 'set'))

  rows = len(ab_timelist)
  for i in range(rows):
    time = ab_timelist[i]
    root_cause_set = ''
    # print(time, one_df[one_df['timestamp'] == time].values[0,2], another_df.loc[another_df['timestamp'] == time].values[0,2])
    if abs(one_df[one_df['timestamp'] == time].values[0,2] - another_df.loc[another_df['timestamp'] == time].values[0,2]) >= 0.05:
      if one_df[one_df['timestamp'] == time].values[0,2] > another_df[another_df['timestamp'] == time].values[0,2]:
        root_cause_set = (one_df[one_df['timestamp'] == time].values)[0,1]
      else:
        print(time, one_df[one_df['timestamp'] == time].values[0,2], len(literal_eval(one_df[one_df['timestamp'] == time].values[0,1])), another_df[another_df['timestamp'] == time].values[0,2], len(literal_eval(another_df[another_df['timestamp'] == time].values[0,1])))
        root_cause_set = (another_df[another_df['timestamp'] == time].values)[0,1]
      merge_df.loc[i] = [time, root_cause_set]
    else:
      # print(time, one_df[one_df['timestamp'] == time].values[0,2], another_df.loc[another_df['timestamp'] == time].values[0,2])
      if len(literal_eval(one_df[one_df['timestamp'] == time].values[0,1])) < len(literal_eval(another_df[another_df['timestamp'] == time].values[0,1])):
        root_cause_set = (one_df[one_df['timestamp'] == time].values)[0,1]
      else:
        root_cause_set = (another_df[another_df['timestamp'] == time].values)[0,1]
      merge_df.loc[i] = [time, root_cause_set]
  
  merge_df.to_csv(merge_result_csv_file, index=False)

# 对比两个result.csv文件，把得分高的结果整合到新文件,带时间、组合和分数
def merge_result_with_set_and_score_v1(ab_timelist, one_result_csv_file, another_result_csv_file, merge_result_csv_file):
  one_df = pd.read_csv(one_result_csv_file, header=0, sep=',', names=['timestamp', 'set','score'])
  another_df = pd.read_csv(another_result_csv_file, header=0, sep=',', names=['timestamp', 'set', 'score'])
  merge_df = pd.DataFrame(columns=('timestamp', 'set', 'score'))

  rows = len(ab_timelist)
  for i in range(rows):
    time = ab_timelist[i]
    root_cause_set = ''
    score = 0
    # print(time, one_df[one_df['timestamp'] == time].values[0,2], another_df.loc[another_df['timestamp'] == time].values[0,2])
    if one_df[one_df['timestamp'] == time].values[0,2] > another_df[another_df['timestamp'] == time].values[0,2] and abs(one_df[one_df['timestamp'] == time].values[0,2] - another_df[another_df['timestamp'] == time].values[0,2]) >= 0.15:
      root_cause_set = (one_df[one_df['timestamp'] == time].values)[0,1]
      score = one_df[one_df['timestamp'] == time].values[0,2]
    else:
      root_cause_set = (another_df[another_df['timestamp'] == time].values)[0,1]
      score = another_df[another_df['timestamp'] == time].values[0,2]
    merge_df.loc[i] = [time, root_cause_set, score]
  
  merge_df.to_csv(merge_result_csv_file, index=False)

# 对比两个result.csv文件，把差值多于0.05得分高的结果整合到新文件,带时间、组合和分数
def merge_result_with_set_and_score_v2(ab_timelist, one_result_csv_file, another_result_csv_file, merge_result_csv_file):
  one_df = pd.read_csv(one_result_csv_file, header=0, sep=',', names=['timestamp', 'set','score'])
  another_df = pd.read_csv(another_result_csv_file, header=0, sep=',', names=['timestamp', 'set', 'score'])
  merge_df = pd.DataFrame(columns=('timestamp', 'set', 'score'))

  rows = len(ab_timelist)
  for i in range(rows):
    time = ab_timelist[i]
    root_cause_set = ''
    score = 0
    # 如果分数差0.04则换高分的，否则选集合较小的
    if abs(one_df[one_df['timestamp'] == time].values[0,2] - another_df.loc[another_df['timestamp'] == time].values[0,2]) >= 0.05:
      # print(time, one_df[one_df['timestamp'] == time].values[0,2], another_df.loc[another_df['timestamp'] == time].values[0,2])
      if one_df[one_df['timestamp'] == time].values[0,2] > another_df[another_df['timestamp'] == time].values[0,2]:
        root_cause_set = (one_df[one_df['timestamp'] == time].values)[0,1]
        score = one_df[one_df['timestamp'] == time].values[0,2]
      else:
        root_cause_set = (another_df[another_df['timestamp'] == time].values)[0,1]
        score = another_df[another_df['timestamp'] == time].values[0,2]
      merge_df.loc[i] = [time, root_cause_set, score]
    else:
      print(time, one_df[one_df['timestamp'] == time].values[0,2], another_df.loc[another_df['timestamp'] == time].values[0,2])
      if len(literal_eval(one_df[one_df['timestamp'] == time].values[0,1])) < len(literal_eval(another_df[another_df['timestamp'] == time].values[0,1])):
        root_cause_set = (one_df[one_df['timestamp'] == time].values)[0,1]
        score = one_df[one_df['timestamp'] == time].values[0,2]
      else:
        root_cause_set = (another_df[another_df['timestamp'] == time].values)[0,1]
        score = another_df[another_df['timestamp'] == time].values[0,2]
      merge_df.loc[i] = [time, root_cause_set, score]
  
  merge_df.to_csv(merge_result_csv_file, index=False)

# 从文件获取异常时刻的列表，注意看第一行是列名还是数据，异常时刻是int类型
def get_abnormal_times_list(abnormal_time_file):
  times = pd.read_csv(abnormal_time_file, header=None, sep=',')
  times_list = list(times.iloc[0:,0].values)
  # print(len(times_list), times_list)
  return times_list

# 把结果文件中的下标转为具体属性值
def index_to_attr(from_csv, to_csv):
  data = pd.read_csv(from_csv, header=0, sep=',', names=['timestamp', 'set'])
  to_data = pd.DataFrame(columns=('timestamp', 'set'))
  count = 0
  for index, row in data.iterrows():
    time = row['timestamp']
    combination_set = literal_eval(row['set'])
    root_cause_set = ''
    for combination in combination_set:
      root_cause = ''
      # 把所有-1的维度排除掉，找出非*的维度,如[-1,-1,1,1,-1]找出维度在数组中对应下标为[2,3]
      root_index = [i for i in range(5) if combination[i] != -1]
      for dim in root_index:
        if dim == 0:
          root_cause = root_cause + 'i' + str(combination[dim]) + '&'
        elif dim == 1:
          root_cause = root_cause + 'e' + str(combination[dim]) + '&'
        elif dim == 2:
          root_cause = root_cause + 'c' + str(combination[dim]) + '&'
        elif dim == 3:
          root_cause = root_cause + 'p' + str(combination[dim]) + '&'
        else:
          root_cause = root_cause + 'l' + str(combination[dim]) + '&'
      if root_cause != '':
        root_cause_set = root_cause_set + ';' + root_cause[0:-1]
      
    
    # 删掉开头的;符号
    root_cause_set = root_cause_set[1:]

    to_data.loc[count] = [time, root_cause_set]
    count += 1
  
  to_data.to_csv(to_csv, index=False)
  


if __name__ == "__main__":
  timelist = get_abnormal_times_list('./Anomalytime_test3.csv')
  # merge_result_v1(timelist, './date18_rootset_and_score.csv', './date22_rootset_and_score.csv', './date22_merge_result.csv')
  # merge_result_with_set_and_score_v1(timelist, './date25_rootset_and_score_v2.csv', './date22_merge_rootset_and_score.csv', './date25_merge_rootset_and_score_v2_2.csv')


  # merge_result_v2(timelist, './date22_merge_rootset_and_score.csv', './date23_rootset_and_score.csv', './date23_merge_rootset_and_score.csv')
  # index_to_attr('./date23_merge_rootset_and_score.csv', './date23_final_result.csv')

  # merge_result_v1(timelist, './date23_rootset_and_score.csv', './date22_merge_rootset_and_score.csv', './date24_merge_result.csv')
  # index_to_attr('./date24_merge_result.csv', './date24_final_result.csv')

  # merge_result_v1(timelist, './date25_rootset_and_score_v2.csv', './date22_merge_rootset_and_score.csv', './date25_merge_result_v2_2.csv')
  # merge_result_v1(timelist, './date25_rootset_and_score_v1.csv', './date25_merge_rootset_and_score_v2_2.csv', './date25_merge_result_v1_v2_2.csv')

  index_to_attr('./date25_merge_result_v1_v2_2.csv', './date25_final_result.csv')