# -*- coding: utf-8 -*-
import json
import math
import numpy as np
import copy
import random
import operator


class Node(object):
    # node类初始化
    def __init__(self):
        self.parents = None
        self.children = []
        self.state = []
        self.seq = []  

        self.Q = 0
        self.N = 0


def selection(node, explored_leaf_node, maxLeafNode, seq):
    all_selected = False
    # 当叶子节点没有被搜索完时
    while len(explored_leaf_node) < maxLeafNode:
        # 当前节点不是叶子节点时
        while len(node.seq) < len(seq):
            # 第一次访问新节点，初始化它的孩子节点
            if len(node.children) == 0:
                init_children(node, seq)
            # 如果当前节点存在没有访问过的孩子节点，则依据概率选择深度优先还是广度优先
            Q_max = 0
            is_random = False
            for i in node.children:
                if i.Q > Q_max:
                    Q_max = i.Q
                if i.N == 0:
                    is_random = True

            if is_random:
                if random.random() > Q_max:
                    return node, all_selected

            # 否则依据UCB公式计算最优的孩子节点，重复这个过程
            node = best_child(node)

        # 当访问到新的叶子节点时，添加到叶子节点列表
        for n in explored_leaf_node:
            if operator.eq(node.seq, n) == False:
                explored_leaf_node.append(node.seq)
                break
        # 同时对到达叶子节点这条路径上的所有节点：N+1
        while True:
            if node.parents is not None:
                node.N += 1
                node = node.parents
            else:
                node.N += 1
                break



def init_children(node, seq):
    # 搜集不在当前节点中的元素，放入列表rest_e
    rest_e = []
    for i in seq:
        for j in node.seq:
            if operator.eq(i, j) == False:
                rest_e.append(i) 
                break
    # 取rest_e中的一个元素与当前节点状态组合，生成新的节点            
    for e in rest_e:
        child = Node()
        for parent_seq in node.seq:
            child.seq.append(parent_seq)
        child.seq.append(e)
        child.parents = node
        node.children.append(child)



def best_child(node):
    # 依据UCB公式计算最优孩子节点
    best_score = -1
    best = None

    for sub_node in node.children:

        # 在可选的节点里面选择最优
        if sub_node.Q > 0:
            C = math.sqrt(2.0)
            left = sub_node.Q
            right = math.log(node.N) / sub_node.N
            score = left + C * math.sqrt(right)

            if score > best_score:
                best = sub_node
                best_score = score

    return best


def expansion(selection_node, score_single_e, seq):
    # 得到所有孩子节点中的新元素
    e_field = []
    for i in selection_node.children:
        if i.N == 0:
            e_field.append(i.seq[-1])

    # 在新元素中选择Q值最大的一个
    max_e, max_seq = get_max_e(e_field, score_single_e, seq)
    return max_e, max_seq


def get_max_e(e_field, score_single_e, seq):
    max_e = -1
    max_score = -1
    max_seq = []
    for e in e_field:
        e_str = ''.join(str(j) for j in e)
        score = score_single_e[e_str]
        if score > max_score:
            max_score = score
            max_e = e
            max_seq = e
    return max_e, max_seq


def evalation(selection_node, max_seq, forecast, real, v, f):
    new_set = copy.deepcopy(selection_node.seq)
    new_set.append(max_seq)
    # 对新状态计算Q值大小
    new_q = get_scores(new_set, forecast, real, v, f)
    return new_q


def get_scores(set, forecast, real, v, f): #set-候选集合，一个元素为一个list， forecast-预测值， real-真实值， v-真实向量， f-预测向量
    # 复制预测值为cp(copy)
    cp = copy.deepcopy(forecast[:-1, :-1, :-1, :-1, :-1])
    # 在cp的基础上，根据状态中的所有元素，
    # 不为最细颗粒度时，将cp对应位置根据公式5改为计算值a，为最细颗粒度时，将cp对应位置改为真实值
    # flag = 0: 最细粒度，   flag = 1:不为最细粒度
    flag = 0
    for tmp in set[0]:
        if tmp == -1:
            flag = 1
            break
    # 不为最细颗粒度,改变为计算值
    if flag == 1:
        for i in set:
            dim = [] # dim: 第一列代表起始， 第二列代表终止。 若为-1。则为[0, shape-1]；若不为0，则为[n, n+1]
            for j in range(len(i)):
               if i[j] == -1: dim.append([0, forecast.shape[j]-1])
               else: dim.append([i[j], i[j]+1])

            for d1 in range(dim[0][0], dim[0][1]):
                for d2 in range(dim[1][0], dim[1][1]):
                    for d3 in range(dim[2][0], dim[2][1]):
                        for d4 in range(dim[3][0], dim[3][1]):
                            for d5 in range(dim[4][0], dim[4][1]):
                                cp[d1][d2][d3][d4][d5] = getValueA(forecast[d1][d2][d3][d4][d5], forecast[i[0]][i[1]][i[2]][i[3]][i[4]], real[i[0]][i[1]][i[2]][i[3]][i[4]])
    # 最细粒度,直接改变为真实值
    else:   
        for i in set:
            cp[i[0]][i[1]][i[2]][i[3]][i[4]] = real[i[0]][i[1]][i[2]][i[3]][i[4]]

    # 整理为一维
    a = cp.flatten()
    v = v.flatten()
    f = f.flatten()
    # 计算Q值的最终公式
    result = max(1 - getDistance(v, a) / getDistance(v, f), 0)
    return result


def getValueA(a, b, c):
    # 计算值公式
    return a - (b - c) * float(a) / b


def getDistance(u, w):
    # 计算两向量的距离
    sum = 0
    for i in range(len(u)):
        sum += (u[i] - w[i]) ** 2
    return math.sqrt(sum)


def backup(selection_node, max_e, new_q):
    index = -1
    # 获取计算节点在孩子中的序号
    for i in range(len(selection_node.children)):
        if operator.eq(selection_node.children[i].seq[-1], max_e):
            index = i

    # 从最下层节点开始，对整条路径上的节点：N+1，Q赋值为路径中最大Q值
    node = selection_node.children[index]
    while node is not None:
        node.N += 1
        if new_q > node.Q:
            node.Q = new_q
        node = node.parents


def get_best_node(node):
    # 获得最大Q值的所有节点中的最下层的节点
    best_score = node.Q
    while len(node.children) is not 0:
        for index in range(len(node.children)):
            if node.children[index].Q == best_score:
                node = node.children[index]
                break
    return node


def MCTS(forecast, real, seq, M, PT):
    # 累乘，计算叶子节点的最大数量。当搜索过所有叶子节点时，停止搜索
    maxLeafNode = 1
    for i in range(1, len(seq) + 1):
        maxLeafNode = maxLeafNode * i

    # 初始化探索过的叶子节点列表
    explored_leaf_node = []

    # 计算Q值公式中需要的真实向量v、预测向量f(最细粒度)，跟choise无关,去掉最后的累和
    v = copy.deepcopy(real[:-1, :-1, :-1, :-1, :-1])
    f = copy.deepcopy(forecast[:-1, :-1, :-1, :-1, :-1])
    # row_num = len(forecast) - 1
    # for i in range(row_num):
    #     v.extend(real[i][:-1][:-1][:-1][:-1])  #
    #     f.extend(forecast[i][:-1][:-1][:-1][:-1])
    v = np.array(v)
    f = np.array(f)

    # 计算单元素Q值
    score_single_e = {}
    for e in seq:
        e_str = ''.join(str(j) for j in e)
        score_single_e[e_str] = get_scores([e], forecast, real, v, f)

    # 初始化根节点,Q值记录，最优节点
    node = Node()
    max_q = 0
    best_node = None

    # 开始搜索，最大搜索次数可变
    for i in range(M):

        # 1、选择，如果所有节点搜索完毕，则跳出循环
        selection_node, all_selected = selection(node, explored_leaf_node, maxLeafNode, seq)
        if all_selected:
            break

        # 2、扩展，获得剩余元素中的最大元素值
        max_e, max_seq = expansion(selection_node, score_single_e, seq)

        # 3、评价，原状态与最大元素值组合成新状态，获得新状态的Q值
        new_q = evalation(selection_node, max_seq, forecast, real, v, f)

        # 4、更新，新状态节点至根节点路径中的每个节点：N+1，Q赋值为路径中最大Q值
        backup(selection_node, max_e, new_q)

        # 如果根节点Q值变大，则更新最优节点
        if node.Q > max_q:
            best_node = get_best_node(node)
            max_q = node.Q
        # 如果新节点的Q值超过预设阀值，则跳出循环
        if new_q >= PT:
            break
    return best_node


# def get_choise(number):
#     choise = []
#     for i in range(number - 1):
#         choise.append([i])
#     return choise

def get_seq(number, dimension):
    seq = []
    for i in range(number - 1):
        tmp = [-1, -1, -1, -1, -1] #-1代表*
        tmp[dimension] = i
        seq.append(tmp)
    return seq

def get_mix_seq(node1, node2, col1, col2): #两个维度的组合
    seq = []
    col1 = col1 - 1
    col2 = col2 - 1
    for s1 in node1.seq:
        for s2 in node2.seq:
            tmp = s1
            tmp[col2] = s2[col2]
            seq.append(tmp)
    return seq

def get_mix_seq3(node1, node2, node3, col1, col2, col3): # 三个维度的组合
    # mix_node12, mix_node23, mix_node13, 1, 2, 3
    seq = []
    col1 = col1 - 1 
    col2 = col2 - 1
    col3 = col3 - 1
    dim1 = []
    dim2 = []
    dim3 = []
    # 12
    set1 = set()
    set2 = set()
    for s1 in node1.seq:
        set1.add(s1[col1])
        set2.add(s1[col2])
    dim1.append(list(set1))
    dim2.append(list(set2))
    # 23
    set2 = set()
    set3 = set()
    for s2 in node2.seq:
        set2.add(s2[col2])
        set3.add(s2[col3])
    dim2.append(list(set2))
    dim3.append(list(set3))
    # 13
    set1 = set()
    set3 = set()
    for s3 in node3.seq:
        set1.add(s3[col1])
        set3.add(s3[col3])
    dim1.append(list(set1))
    dim3.append(list(set3))
    # 交集
    dim1 = set(dim1[0]).intersection(*dim1[1:])
    dim2 = set(dim2[0]).intersection(*dim2[1:])
    dim3 = set(dim3[0]).intersection(*dim3[1:])
    for d1 in dim1:
        for d2 in dim2:
            for d3 in dim3:
                tmp = [-1, -1, -1, -1, -1]
                tmp[col1] = d1
                tmp[col2] = d2
                tmp[col3] = d3
                seq.append(tmp)
    return seq

def get_mix_seq4(node1, node2, node3, node4, col1, col2, col3, col4): # 四个维度组合
    # mix_node123, mix_node124, mix_node134, mix_nod234, 1, 2, 3, 4
    seq = []
    col1 = col1 - 1 
    col2 = col2 - 1
    col3 = col3 - 1
    col4 = col4 - 1
    dim1 = []
    dim2 = []
    dim3 = []
    dim4 = []
    #123
    set1 = set()
    set2 = set()
    set3 = set()
    for s1 in node1.seq:
        set1.add(s1[col1])
        set2.add(s1[col2])
        set3.add(s1[col3])
    dim1.append(list(set1))
    dim2.append(list(set2))
    dim3.append(list(set3))
    #124
    set1 = set()
    set2 = set()
    set4 = set()
    for s2 in node2.seq:
        set1.add(s2[col1])
        set2.add(s2[col2])
        set4.add(s2[col4])
    dim1.append(list(set1))
    dim2.append(list(set2))
    dim4.append(list(set4))
    #134
    set1 = set()
    set3 = set()
    set4 = set()
    for s3 in node3.seq:
        set1.add(s3[col1])
        set3.add(s3[col3])
        set4.add(s3[col4])
    dim1.append(list(set1))
    dim3.append(list(set3))
    dim4.append(list(set4))
    #234
    set2 = set()
    set3 = set()
    set4 = set()
    for s4 in node4.seq:
        set2.add(s4[col2])
        set3.add(s4[col3])
        set4.add(s4[col4])
    dim2.append(list(set2))
    dim3.append(list(set3))
    dim4.append(list(set4))
    # 交集
    dim1 = set(dim1[0]).intersection(*dim1[1:])
    dim2 = set(dim2[0]).intersection(*dim2[1:])
    dim3 = set(dim3[0]).intersection(*dim3[1:])
    dim4 = set(dim4[0]).intersection(*dim4[1:])
    for d1 in dim1:
        for d2 in dim2:
            for d3 in dim3:
                for d4 in dim4:
                    tmp = [-1, -1, -1, -1, -1]
                    tmp[col1] = d1
                    tmp[col2] = d2
                    tmp[col3] = d3
                    tmp[col4] = d4
                    seq.append(tmp)

    return seq
            
def get_mix_seq5(node1, node2, node3, node4, node5): # 五个维度组合
    # mix_node1234, mix_node1235, mix_node1245, mix_node1345, mix_nod2345
    seq = []
    dim1 = []
    dim2 = []
    dim3 = []
    dim4 = []
    dim5 = []
    #1234
    set1 = set()
    set2 = set()
    set3 = set()
    set4 = set()
    for s1 in node1.seq:
        set1.add(s1[0])
        set2.add(s1[1])
        set3.add(s1[2])
        set4.add(s1[3])
    dim1.append(list(set1))
    dim2.append(list(set2))
    dim3.append(list(set3))
    dim4.append(list(set4))
    #1235
    set1 = set()
    set2 = set()
    set3 = set()
    set5 = set()
    for s2 in node2.seq:
        set1.add(s2[0])
        set2.add(s2[1])
        set3.add(s2[2])
        set5.add(s2[4])
    dim1.append(list(set1))
    dim2.append(list(set2))
    dim3.append(list(set3))
    dim5.append(list(set5))
    #1245
    set1 = set()
    set2 = set()
    set4 = set()
    set5 = set()
    for s3 in node3.seq:
        set1.add(s3[0])
        set2.add(s3[1])
        set4.add(s3[2])
        set5.add(s3[3])
    dim1.append(list(set1))
    dim2.append(list(set2))
    dim4.append(list(set4))
    dim5.append(list(set5))
    #1345
    set1 = set()
    set3 = set()
    set4 = set()
    set5 = set()
    for s4 in node4.seq:
        set1.add(s4[0])
        set3.add(s4[1])
        set4.add(s4[2])
        set5.add(s4[3])
    dim1.append(list(set1))
    dim3.append(list(set3))
    dim4.append(list(set4))
    dim5.append(list(set5))
    #2345
    set2 = set()
    set3 = set()
    set4 = set()
    set5 = set()
    for s5 in node5.seq:
        set2.add(s5[0])
        set3.add(s5[1])
        set4.add(s5[2])
        set5.add(s5[3])
    dim2.append(list(set2))
    dim3.append(list(set3))
    dim4.append(list(set4))
    dim5.append(list(set5))
    # 交集
    dim1 = set(dim1[0]).intersection(*dim1[1:])
    dim2 = set(dim2[0]).intersection(*dim2[1:])
    dim3 = set(dim3[0]).intersection(*dim3[1:])
    dim4 = set(dim4[0]).intersection(*dim4[1:])
    dim5 = set(dim5[0]).intersection(*dim5[1:])
    for d1 in dim1:
        for d2 in dim2:
            for d3 in dim3:
                for d4 in dim4:
                    for d5 in dim5:
                        tmp = [d1, d2, d3, d4, d5]
                        seq.append(tmp)

    return seq


def get_result(dim1_name, dim2_name, dim3_name, dim4_name, dim5_name, forecast, real, M, PT):
    forecast = np.array(forecast)
    real = np.array(real)
    #layer1 对每个维度进行搜索，找到BSet
    dim1_node = MCTS(forecast, real, get_seq(forecast.shape[0], 0), M, PT)
    dim2_node = MCTS(forecast, real, get_seq(forecast.shape[1], 1), M, PT)
    dim3_node = MCTS(forecast, real, get_seq(forecast.shape[2], 2), M, PT)
    dim4_node = MCTS(forecast, real, get_seq(forecast.shape[3], 3), M, PT)
    dim5_node = MCTS(forecast, real, get_seq(forecast.shape[4], 4), M, PT)   
    one_dim = [dim1_node, dim2_node, dim3_node, dim4_node, dim5_node]
    #去除父节点不在BSet中的element，即剪枝, 两两组合
    print(dim1_node.seq)
    
    #layer2 搜索
    mix_node12 = MCTS(forecast, real, get_mix_seq(dim1_node, dim2_node, 1, 2), M, PT)
    mix_node13 = MCTS(forecast, real, get_mix_seq(dim1_node, dim3_node, 1, 3), M, PT)
    mix_node14 = MCTS(forecast, real, get_mix_seq(dim1_node, dim4_node, 1, 4), M, PT)
    mix_node15 = MCTS(forecast, real, get_mix_seq(dim1_node, dim5_node, 1, 5), M, PT)
    mix_node23 = MCTS(forecast, real, get_mix_seq(dim2_node, dim3_node, 2, 3), M, PT)
    mix_node24 = MCTS(forecast, real, get_mix_seq(dim2_node, dim4_node, 2, 4), M, PT)
    mix_node25 = MCTS(forecast, real, get_mix_seq(dim2_node, dim5_node, 2, 5), M, PT)
    mix_node34 = MCTS(forecast, real, get_mix_seq(dim3_node, dim4_node, 3, 4), M, PT)
    mix_node35 = MCTS(forecast, real, get_mix_seq(dim3_node, dim5_node, 3, 5), M, PT)
    mix_node45 = MCTS(forecast, real, get_mix_seq(dim4_node, dim5_node, 4, 5), M, PT)
    two_dim = [mix_node12, mix_node13, mix_node14, mix_node15, mix_node23, mix_node24, mix_node25, mix_node34, mix_node35, mix_node45]

    #layer3 搜索
    mix_node123 = MCTS(forecast, real, get_mix_seq3(mix_node12, mix_node23, mix_node13, 1, 2, 3), M, PT)
    mix_node124 = MCTS(forecast, real, get_mix_seq3(mix_node12, mix_node24, mix_node14, 1, 2, 4), M, PT)
    mix_node125 = MCTS(forecast, real, get_mix_seq3(mix_node12, mix_node25, mix_node15, 1, 2, 5), M, PT)
    mix_node134 = MCTS(forecast, real, get_mix_seq3(mix_node13, mix_node34, mix_node14, 1, 3, 4), M, PT)
    mix_node135 = MCTS(forecast, real, get_mix_seq3(mix_node13, mix_node35, mix_node15, 1, 3, 5), M, PT)
    mix_node145 = MCTS(forecast, real, get_mix_seq3(mix_node14, mix_node45, mix_node15, 1, 4, 5), M, PT)
    mix_node234 = MCTS(forecast, real, get_mix_seq3(mix_node23, mix_node34, mix_node24, 2, 3, 4), M, PT)
    mix_node235 = MCTS(forecast, real, get_mix_seq3(mix_node23, mix_node35, mix_node25, 2, 3, 5), M, PT)
    mix_node245 = MCTS(forecast, real, get_mix_seq3(mix_node24, mix_node45, mix_node25, 2, 4, 5), M, PT)
    mix_node345 = MCTS(forecast, real, get_mix_seq3(mix_node34, mix_node45, mix_node35, 3, 4, 5), M, PT)
    three_dim = [mix_node123, mix_node124, mix_node125, mix_node134, mix_node135, mix_node145, mix_node234, mix_node235, mix_node245, mix_node345]

    #layer4 搜索
    mix_node1234 = MCTS(forecast, real, get_mix_seq4(mix_node123, mix_node124, mix_node134, mix_node234, 1, 2, 3, 4), M, PT)
    mix_node1235 = MCTS(forecast, real, get_mix_seq4(mix_node123, mix_node125, mix_node135, mix_node235, 1, 2, 3, 5), M, PT)
    mix_node1245 = MCTS(forecast, real, get_mix_seq4(mix_node124, mix_node125, mix_node145, mix_node245, 1, 2, 4, 5), M, PT)
    mix_node1345 = MCTS(forecast, real, get_mix_seq4(mix_node134, mix_node135, mix_node145, mix_node345, 1, 3, 4, 5), M, PT)
    mix_node2345 = MCTS(forecast, real, get_mix_seq4(mix_node234, mix_node235, mix_node245, mix_node345, 1, 2, 4, 5), M, PT)
    four_dim = [mix_node1234, mix_node1235, mix_node1245, mix_node1345, mix_node2345]

    mix_node12345 = MCTS(forecast, real, get_mix_seq5(mix_node1234, mix_node1235, mix_node1245, mix_node1345, mix_node2345), M, PT)



    result_seq = []
    result_Q = 0

    for node in one_dim:
        if node.Q > result_Q:
            result_Q = node.Q
            result_seq = node.seq
    for node in two_dim:
        if node.Q > result_Q:
            result_Q = node.Q
            result_seq = node.seq
    for node in three_dim:
        if node.Q > result_Q:
            result_Q = node.Q
            result_seq = node.seq
    for node in four_dim:
        if node.Q > result_Q:
            result_Q = node.Q
            result_seq = node.seq
    if mix_node12345.Q > result_Q:
        result_Q = mix_node12345.Q
        result_seq = mix_node12345.seq

    # # 返回综合结果
    # if row_node.Q >= column_node.Q and row_node.Q >= mix_node.Q:
    #     for i in row_node.state:
    #         result_name.append([row_name[i[0]]])
    #         result_Q = row_node.Q
    # elif column_node.Q >= row_node.Q and column_node.Q >= mix_node.Q:
    #     for i in column_node.state:
    #         result_name.append([column_name[i[0]]])
    #         result_Q = column_node.Q
    # elif mix_node.Q > row_node.Q and mix_node.Q > column_node.Q:
    #     for i in mix_node.state:
    #         result_name.append([row_name[i[0]], column_name[i[1]]])
    #         result_Q = mix_node.Q

    # 返回二维结果
    
    # for i in mix_node.state:
    #     result_name.append([dim1_name[i[0]], dim2_name[i[1]]])
    #     result_Q = mix_node.Q


    return result_seq, result_Q


if __name__ == '__main__':
    # M 是最大搜索次数
    M = 1000000
    # PT 是Q值的阀值
    PT = 0.75
    # 5维
    real = np.load(file="./real_table_1536954600000.npy")  #(148, 14, 10, 36, 6)
    forecast = np.load(file='./real_table_1536976500000.npy')
    
   
    # 测试数据1
    # dim1_name = ['Mobile', 'Unicom']
    # dim2_name = ['Beijing', 'Shanghai', 'Guangzhou']
    # forecast = [[20, 15, 10, 45],
    #             [10, 25, 20, 55],
    #             [30, 40, 30, 100]]
    # real = [[14, 9, 10, 33],
    #         [7, 15, 20, 42],
    #         [21, 24, 30, 75]]
    #dimension, element
    dim1_name = ['Mobile', 'Unicom']
    dim2_name = ['Beijing', 'Shanghai', 'Guangzhou']
    dim3_name = ['a', 'b']
    dim4_name = ['1', '2']
    dim5_name = ['!', '@']
    


    name, Q = get_result(dim1_name, dim2_name, dim3_name, dim4_name, dim5_name, forecast, real, M, PT)

    print ("根因组合: ")
    print (json.dumps(name, ensure_ascii=False))#.encode("utf8")
    print ("组合得分: ")
    print (Q)
