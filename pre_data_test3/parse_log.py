import csv
import os
import json
from ast import literal_eval
import pandas as pd
import numpy as np

def parse_record(from_file_path, to_file_path):
  keyword = "INFO"
  b = open(to_file_path, "w",encoding='UTF-8')
  a = open(from_file_path, "r",encoding='UTF-8') #注意此处的转义字符
  count=len(open(from_file_path,'r',encoding='UTF-8').readlines())#使用len+readlines读取行数

  i = 0
  need_write_next_line = False
  while i < count:
    line = a.readline()

    if need_write_next_line:
      b.write(line)
      need_write_next_line = False

    if "异常时刻" in line or "根因组合" in line:
      print(line)
      need_write_next_line = True

    i+=1

  a.close()
  b.close()

def parse_record_for_score(from_file_path, to_file_path):
  keyword = "INFO"
  b = open(to_file_path, "w",encoding='UTF-8')
  a = open(from_file_path, "r",encoding='UTF-8') #注意此处的转义字符
  count=len(open(from_file_path,'r',encoding='UTF-8').readlines())#使用len+readlines读取行数

  i = 0
  need_write_next_line = False
  while i < count:
    line = a.readline()

    if need_write_next_line:
      b.write(line)
      need_write_next_line = False

    if "异常时刻" in line or "组合得分" in line:
      print(line)
      need_write_next_line = True

    i+=1

  a.close()
  b.close()  

def parse_record_for_rootset_and_score(from_file_path, to_file_path):
  b = open(to_file_path, "w",encoding='UTF-8')
  a = open(from_file_path, "r",encoding='UTF-8') #注意此处的转义字符
  count=len(open(from_file_path,'r',encoding='UTF-8').readlines())#使用len+readlines读取行数

  i = 0
  need_write_next_line = False
  while i < count:
    line = a.readline()

    if need_write_next_line:
      b.write(line)
      need_write_next_line = False

    if "异常时刻" in line or "根因组合" in line or "组合得分" in line:
      print(line)
      need_write_next_line = True

    i+=1

  a.close()
  b.close()  

# 功能：转换为属性值并且按格式追加进结果文件 result.csv
# 输入 异常时刻，下标表示的根因组合
# 输出 把结果作为一行数据写进文件 result.csv
# 例如，输入135487820，[[-1,1,-1,1,-1], [-1,3,-1,7,-1]]，输出135487820,e1&p1;e3&p7
def transform_and_write_to_csv(time, combination_set):
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
  # print(root_cause_set)

  # 把该时刻的根因组合追加进结果文件
  path  = "./date23_result.csv"
  header = ['timestamp', 'set']
  need_write_header = (not os.path.exists('./date23_result.csv')) or (not os.path.getsize('./date23_result.csv'))
  with open(path, 'a+', newline='') as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=header)
    if need_write_header:
      csv_writer.writeheader()
    csv_writer.writerow({'timestamp': time, 'set': root_cause_set})

  return [time, root_cause_set]

# 输入 ./xxx/xaxa/1501475700.npy
# 输出 1501475700
def filepath_to_time(filepath):
  time = os.path.basename(filepath).split('.')[0]
  print(time, type(time))
  return time



def new_record_to_csv(new_record_file_path):
  record = open(new_record_file_path, "r",encoding='UTF-8') 
  count = len(open(new_record_file_path,'r',encoding='UTF-8').readlines())

  i = 0
  need_write_next_line = False
  time = '0'
  root_cause_set = ''
  while i < count:
    line = record.readline()

    if i % 2 == 0: time = line
    else: 
      root_cause_set = line
      # 注意传参的类型，时间str，根因组合list
      transform_and_write_to_csv(filepath_to_time(time), literal_eval(root_cause_set))

    i+=1

  record.close()

def new_score_record_to_csv(new_record_file_path, to_csv_file_path):
  record = open(new_record_file_path, "r",encoding='UTF-8') 
  count = len(open(new_record_file_path,'r',encoding='UTF-8').readlines())

  i = 0
  need_write_next_line = False
  time = '0'
  score = 0
  while i < count:
    line = record.readline()

    if i % 2 == 0: time = filepath_to_time(line)
    else: 
      score = float(line)

      # 把该时刻的根因分数追加进结果文件
      header = ['timestamp', 'score']
      need_write_header = (not os.path.exists(to_csv_file_path)) or (not os.path.getsize(to_csv_file_path))
      with open(to_csv_file_path, 'a+', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=header)
        if need_write_header:
          csv_writer.writeheader()
        csv_writer.writerow({'timestamp': time, 'score': score})
    i+=1

  record.close()  

def new_rootcause_score_record_to_csv(new_record_file_path, to_csv_file_path):
  record = open(new_record_file_path, "r",encoding='UTF-8') 
  count = len(open(new_record_file_path,'r',encoding='UTF-8').readlines())

  i = 1
  need_write_next_line = False
  time = 0
  root_cause_set = ''
  score = 0
  while i <= count:
    line = record.readline()

    if i % 3 == 1: time = filepath_to_time(line)
    elif i % 3 == 2: root_cause_set = literal_eval(line)
    else: 
      score = float(line)

      # 把该时刻的根因分数追加进结果文件
      header = ['timestamp', 'set', 'score']
      need_write_header = (not os.path.exists(to_csv_file_path)) or (not os.path.getsize(to_csv_file_path))
      with open(to_csv_file_path, 'a+', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=header)
        if need_write_header:
          csv_writer.writeheader()
        csv_writer.writerow({'timestamp': time, 'set': root_cause_set, 'score': score})
    i+=1

  record.close()  

# 从文件获取异常时刻的列表，注意看第一行是列名还是数据，异常时刻是int类型
def get_abnormal_times_list(abnormal_time_file):
  times = pd.read_csv(abnormal_time_file, header=None, sep=',')
  times_list = list(times.iloc[0:,0].values)
  print(len(times_list), times_list)
  return times_list

# 为缺少的时刻填充空白结果
def fill_result(record_csv_file, ab_timelist):
  count = len(ab_timelist)
  from_df = pd.read_csv(record_csv_file, header=0, sep=',', names=['timestamp','set'])
  to_df = pd.DataFrame(columns=('timestamp', 'set'))
  for i in range(count):
    time = ab_timelist[i]
    root_cause_set = ''
    # 若我们的结果缺少了该时刻，则填充为空
    if from_df.loc[from_df['timestamp'] == time].empty:
      root_cause_set = 'i1'
    else:
      print(np.array(from_df.loc[from_df['timestamp'] == time])[0,1])
      root_cause_set = np.array(from_df.loc[from_df['timestamp'] == time])[0,1]
    to_df.loc[i] = [time, root_cause_set]
  
  to_df.to_csv('./fullResult.csv',index=False)

if __name__ == "__main__":
  # 第一次测试

  # parse_record("./all_result.txt", "./new_record.txt")
  # new_record_to_csv("./new_record.txt")
  # fill_result("./result.csv", get_abnormal_times_list('./Anomalytime_test3.csv'))

  # parse_record_for_score("./all_result.txt", "./new_record_with_score.txt")
  # new_score_record_to_csv("./new_record_with_score.txt", "./new_record_with_score.csv")

  # parse_record_for_rootset_and_score("./date18_all_result.txt", "./date18_rootset_and_score.txt")
  # new_rootcause_score_record_to_csv("./date18_rootset_and_score.txt", "./date18_rootset_and_score.csv")

  # date22新版本结果
  # parse_record("./all_result_date22.txt", "./new_record.txt")
  # new_record_to_csv("./new_record.txt")

  # parse_record_for_rootset_and_score("./date22_all_result.txt", "./date22_rootset_and_score.txt")
  # new_rootcause_score_record_to_csv("./date22_rootset_and_score.txt", "./date22_rootset_and_score.csv")

  # new_record_to_csv("./new_record.txt")

  # date23
  # parse_record("./all_result_date23.txt", "./date23_new_record.txt")
  # new_record_to_csv("./date23_new_record.txt")
  
  # parse_record_for_rootset_and_score("./date23_all_result.txt", "./date23_rootset_and_score.txt")
  # new_rootcause_score_record_to_csv("./date23_rootset_and_score.txt", "./date23_rootset_and_score.csv")

  # date25
  # parse_record_for_rootset_and_score("./date25_all_result_v1.txt", "./date25_rootset_and_score_v1.txt")
  new_rootcause_score_record_to_csv("./date25_rootset_and_score_v1.txt", "./date25_rootset_and_score_v1.csv")

  # parse_record_for_rootset_and_score("./date25_all_result_v2.txt", "./date25_rootset_and_score_v2.txt")
  # new_rootcause_score_record_to_csv("./date25_rootset_and_score_v2.txt", "./date25_rootset_and_score_v2.csv")