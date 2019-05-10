# rootcause

- hotspot.py: 二维版本代码

- version1.py: 修复了一些bugs，但跑一次需要很久，怀疑因为每计算一个ps值需要扫描所有最细颗粒度的值，需进一步review

- real*.npy: 异常时刻的真实值数据，五维数组

- record_test1.md: 数据集描述

tips: version1.py中，元素值使用下标表示，若集合为(0, *, 5, *, *, *)， 则seq为(0, -1, 5, -1, -1,- 1)
