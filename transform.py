import csv
import os

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
  path  = "./result.csv"
  header = ['timestamp', 'set']
  need_write_header = (not os.path.exists('./result.csv')) or (not os.path.getsize('./result.csv'))
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

# 文件格式
# timestamp,set
# 1501475700,a1&b2;a3&b4
# 1501475760,a1&b2&x3;a4&b5&x6
if __name__ == "__main__":
  # filepath_to_time('./sasa/asczxcxzc/1501475700.npy')
  path  = "./result.csv"
  if os.path.exists(path):
    os.remove(path)
  transform_and_write_to_csv(1501475700, [[-1,1,-1,1,-1], [-1,3,-1,7,-1]])
  transform_and_write_to_csv(1501475760, [[-1,-1,4,-1,-1], [-1,-1,9,-1,-1]])
  
  