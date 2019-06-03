import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt	

# df=pd.read_csv('./mytest.csv',header=None,sep=',',names=['i','e','c','p','l','PV'])
# print(len(list(df['i'].value_counts().index)), list(df['i'].value_counts().index))
# print(len(list(df['e'].value_counts().index)), list(df['e'].value_counts().index))
# print(len(list(df['c'].value_counts().index)), list(df['c'].value_counts().index))
# print(len(list(df['p'].value_counts().index)), list(df['p'].value_counts().index))
# print(len(list(df['l'].value_counts().index)), list(df['l'].value_counts().index))                                                  

# #把同一纬度的属性值合并存到一个series,写入专属文件并重新读取出成一维数组
# a=set({8.0, 5.0, 4.0})
# b=set(df['l'].value_counts().index)#不考虑NaN
# print(a.union(b))
# s = pd.Series(list(a.union(b)))
# print(s)
# s.to_csv('./l.csv', header=False, index=False)
# data = pd.read_csv('./l.csv', header=False, index=False)
# print(data.shape)

# print(type(data), set(data.iloc[:,0].values))

#根据某些列值筛选出df数据的分组
#print(df.loc[df['i'].isin(['01'])])

#根据数字下标访问某个元素
# print(type(df.loc[df['i'].isin(['6.0'])].iat[1895,1]))

#对PV这一列求和
# print(df.loc[df['i'].isin(['01'])].apply(sum)['PV'])



# 生成五维数组 ['i','e','c','p','l','PV']
table = []
for i in range(2):
  i_dim = []
  for e in range(3):
    e_dim = []
    for c in range(4):
      c_dim = []
      for p in range(5):
        p_dim = []
        for l in range(6):
          p_dim.append(i+e+c+p+l)
        c_dim.append(p_dim)
      e_dim.append(c_dim)
    i_dim.append(e_dim)
  table.append(i_dim)
print(table[0][0][0][0][0],table[0][0][0][0][3])
