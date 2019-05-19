# -*- coding: utf-8 -*-
import json
import math
import numpy as np
import copy
import random
import operator
import datetime
import sys
import multiprocessing


class Node(object):
    # node类初始化
    def __init__(self):
        self.parents = None
        self.children = []
        self.seq = [] 

        self.Q = 0
        self.N = 0


def selection(node, explored_leaf_node, maxLeafNode, seq):
    all_selected = False
    # 节点未全部搜完时
    while len(explored_leaf_node) < maxLeafNode:
        # 当前节点不是所有属性值组合时
        while len(node.seq) < len(seq):
            # 第一次访问新节点，初始化它的孩子节点
            if len(node.children) == 0:
                init_children(node, seq)
            # 如果当前节点存在没有访问过的孩子节点，则依据概率选择是否访问该孩子节点
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

        # 当前节点为所有属性值组合，即最底层节点时，回溯有子节点未被访问过的节点
        have_unvisited = False
        while have_unvisited == False and node.parents is not None:
            node = node.parents
            for i in node.children:
                if i.N == 0:
                    have_unvisited = True
                    break
    # 全部搜完时
    all_selected = True
    return node, all_selected

def add_explore_node(explored_leaf_node, node):
    # 叶子节点列表为空
    if len(explored_leaf_node) == 0:
        explored_leaf_node.append(node.seq)
    # 叶子节点列表不为空
    else:
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
        # 当前节点元素为空，即根节点
        if len(node.seq) == 0:
            rest_e.append(i)
        # 当前节点元素非空
        else:
            is_exit = False
            for j in node.seq:
                if operator.eq(i, j) == True:
                    is_exit = True
                    break
            if is_exit == False: rest_e.append(i)
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
        # if sub_node.Q > 0:
        if sub_node.N > 0:
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
    max_seq = get_max_e(e_field, score_single_e, seq)
    return max_seq


def get_max_e(e_field, score_single_e, seq):
    max_score = -1
    max_seq = []
    for e in e_field:
        e_str = ''.join(str(j) for j in e)
        score = score_single_e[e_str]
        if score > max_score:
            max_score = score
            max_seq = e
    return max_seq


def evalation(selection_node, max_seq, forecast, real, v, f, explored_leaf_node):
    new_set = copy.deepcopy(selection_node.seq)
    new_set.append(max_seq)
    # 将新节点加入已探索节点列表
    explored_leaf_node.append(new_set)
    # 对新状态计算Q值大小
    new_q = get_scores(new_set, forecast, real, v, f)
    return new_q


def get_scores(set, forecast, real, v, f): #set-当前集合，一个元素为一个list， forecast-预测值， real-真实值， v-真实向量， f-预测向量
    # 复制预测值为cp(copy)，去除最后的累和
    cp = copy.deepcopy(forecast[:-1, :-1, :-1, :-1, :-1])
    # 在cp的基础上，看集合中的元素，
    # 元素不为最细颗粒度时，将cp对应位置根据公式5改为计算值a，为最细颗粒度时，将cp对应位置改为真实值
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
                                if forecast[i[0]][i[1]][i[2]][i[3]][i[4]] == 0: cp[d1][d2][d3][d4][d5] = 0  # 预测值为0，暂时将计算值置零
                                else: cp[d1][d2][d3][d4][d5] = getValueA(forecast[d1][d2][d3][d4][d5], forecast[i[0]][i[1]][i[2]][i[3]][i[4]], real[i[0]][i[1]][i[2]][i[3]][i[4]])
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


def backup(selection_node, max_seq, new_q):
    index = -1
    # 获取计算节点在孩子中的序号
    for i in range(len(selection_node.children)):
        if operator.eq(selection_node.children[i].seq[-1], max_seq):
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
    v = np.array(v)
    f = np.array(f)

    # 计算单元素Q值
    # score_single_e为字典，key为字符串化的元素seq，value为ps值
    score_single_e = {}
    for e in seq:
        e_str = ''.join(str(j) for j in e)
        score_single_e[e_str] = get_scores([e], forecast, real, v, f)

    # 初始化根节点,Q值记录，最优节点
    node = Node()
    max_q = -1
    best_node = None
    

    # 开始搜索，最大搜索次数可变
    for i in range(M):

        # node = Node()

        # 1、选择，如果所有节点搜索完毕，则跳出循环
        selection_node, all_selected = selection(node, explored_leaf_node, maxLeafNode, seq)
        if all_selected:
            break

        # 2、扩展，获得剩余元素中的最大元素值
        max_seq = expansion(selection_node, score_single_e, seq)

        # 3、评价，原状态与最大元素值组合成新状态，获得新状态的Q值
        new_q = evalation(selection_node, max_seq, forecast, real, v, f, explored_leaf_node)

        # 4、更新，新状态节点至根节点路径中的每个节点：N+1，Q赋值为路径中最大Q值
        backup(selection_node, max_seq, new_q)

        # node在selection操作中被改变了，需要将node重新指向根节点
        node = selection_node
        while node.parents is not None:
            node = node.parents

        # 如果根节点Q值变大，则更新最优节点
        if node.Q > max_q:
            best_node = get_best_node(node)
            max_q = node.Q
        

        # node在get_best_node操作中被改变了，需要将node重新指向根节点
        node = selection_node
        while node.parents is not None:
            node = node.parents

        # 如果新节点的Q值超过预设阀值，则跳出循环
        if new_q >= PT:
            break

    return best_node


def get_seq(number, dimension):
    seq = []
    for i in range(number - 1):
        tmp = [-1, -1, -1, -1, -1] #-1代表*
        tmp[dimension] = i
        seq.append(tmp)
    return seq

def get_mix_seq(node1, node2, col1, col2): #两个维度的组合
    if node1 is None:
        node1 = Node()
        node1.seq = [[-1, -1, -1, -1, -1]]
    if node2 is None:
        node2 = Node()
        node2.seq = [[-1, -1, -1, -1, -1]]
    seq = []
    col1 = col1 - 1
    col2 = col2 - 1
    dim1 = []
    dim2 = []
    #1
    set1 = set()
    for s1 in node1.seq:
        set1.add(s1[col1])
    dim1 = list(set1)
    #2
    set2 = set()
    for s2 in node2.seq:
        set2.add(s2[col2])
    dim2 = list(set2)
    # 交集
    for d1 in dim1:
        for d2 in dim2:
            tmp = [-1, -1, -1, -1, -1]
            tmp[col1] = d1
            tmp[col2] = d2
            seq.append(tmp)

    return seq

def get_mix_seq3(node1, node2, node3, col1, col2, col3): # 三个维度的组合
    if node1 is None:
        node1 = Node()
        node1.seq = [[-1, -1, -1, -1, -1]]
    if node2 is None:
        node2 = Node()
        node2.seq = [[-1, -1, -1, -1, -1]]
    if node3 is None:
        node3 = Node()
        node3.seq = [[-1, -1, -1, -1, -1]]
        
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
    if node1 is None:
        node1 = Node()
        node1.seq = [[-1, -1, -1, -1, -1]]
    if node2 is None:
        node2 = Node()
        node2.seq = [[-1, -1, -1, -1, -1]]
    if node3 is None:
        node3 = Node()
        node3.seq = [[-1, -1, -1, -1, -1]]
    if node4 is None:
        node4 = Node()
        node4.seq = [[-1, -1, -1, -1, -1]]
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
    if node1 is None:
        node1 = Node()
        node1.seq = [[-1, -1, -1, -1, -1]]
    if node2 is None:
        node2 = Node()
        node2.seq = [[-1, -1, -1, -1, -1]]
    if node3 is None:
        node3 = Node()
        node3.seq = [[-1, -1, -1, -1, -1]]
    if node4 is None:
        node4 = Node()
        node4.seq = [[-1, -1, -1, -1, -1]]
    if node5 is None:
        node5 = Node()
        node5.seq = [[-1, -1, -1, -1, -1]]
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
        set4.add(s3[3])
        set5.add(s3[4])
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
        set3.add(s4[2])
        set4.add(s4[3])
        set5.add(s4[4])
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
        set2.add(s5[1])
        set3.add(s5[2])
        set4.add(s5[3])
        set5.add(s5[4])
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
    
    #layer1 对每个维度进行搜索，找到BSet
    dim1_node = MCTS(forecast, real, get_seq(forecast.shape[0], 0), M, PT)
    dim2_node = MCTS(forecast, real, get_seq(forecast.shape[1], 1), M, PT)
    dim3_node = MCTS(forecast, real, get_seq(forecast.shape[2], 2), M, PT)
    dim4_node = MCTS(forecast, real, get_seq(forecast.shape[3], 3), M, PT)
    dim5_node = MCTS(forecast, real, get_seq(forecast.shape[4], 4), M, PT)   

    one_dim = [dim1_node, dim2_node, dim3_node, dim4_node, dim5_node]
    print('one dimension result')
    for d in one_dim:
        if d is None: print('None')
        else: print(d.seq)
    print()
    
    # get_mix_seq函数用于去除父节点不在BSet中的element，即剪枝, 两两组合

    #layer2 搜索
    if dim1_node is None or dim2_node is None: mix_node12 = None
    else: mix_node12 = MCTS(forecast, real, get_mix_seq(dim1_node, dim2_node, 1, 2), M, PT)
    if dim1_node is None or dim3_node is None: mix_node13 = None
    else: mix_node13 = MCTS(forecast, real, get_mix_seq(dim1_node, dim3_node, 1, 3), M, PT)
    if dim1_node is None or dim4_node is None: mix_node14 = None
    else: mix_node14 = MCTS(forecast, real, get_mix_seq(dim1_node, dim4_node, 1, 4), M, PT)
    if dim1_node is None or dim5_node is None: mix_node15 = None
    else: mix_node15 = MCTS(forecast, real, get_mix_seq(dim1_node, dim5_node, 1, 5), M, PT)
    if dim2_node is None or dim3_node is None: mix_node23 = None
    else: mix_node23 = MCTS(forecast, real, get_mix_seq(dim2_node, dim3_node, 2, 3), M, PT)
    if dim2_node is None or dim4_node is None: mix_node24 = None
    else: mix_node24 = MCTS(forecast, real, get_mix_seq(dim2_node, dim4_node, 2, 4), M, PT)
    if dim2_node is None or dim5_node is None: mix_node25 = None
    else: mix_node25 = MCTS(forecast, real, get_mix_seq(dim2_node, dim5_node, 2, 5), M, PT)
    if dim3_node is None or dim4_node is None: mix_node34 = None
    else: mix_node34 = MCTS(forecast, real, get_mix_seq(dim3_node, dim4_node, 3, 4), M, PT)
    if dim3_node is None or dim5_node is None: mix_node35 = None
    else: mix_node35 = MCTS(forecast, real, get_mix_seq(dim3_node, dim5_node, 3, 5), M, PT)
    if dim4_node is None or dim5_node is None: mix_node45 = None
    else: mix_node45 = MCTS(forecast, real, get_mix_seq(dim4_node, dim5_node, 4, 5), M, PT)

    two_dim = [mix_node12, mix_node13, mix_node14, mix_node15, mix_node23, mix_node24, mix_node25, mix_node34, mix_node35, mix_node45]
    print('two dimension result')
    for d in two_dim:
        if d is None: print('None')
        else: print(d.seq)
    print()

    #layer3 搜索
    if mix_node12 is None or mix_node23 is None or mix_node13 is None: mix_node123 = None
    else: mix_node123 = MCTS(forecast, real, get_mix_seq3(mix_node12, mix_node23, mix_node13, 1, 2, 3), M, PT)
    if mix_node12 is None or mix_node24 is None or mix_node14 is None: mix_node124 = None
    else: mix_node124 = MCTS(forecast, real, get_mix_seq3(mix_node12, mix_node24, mix_node14, 1, 2, 4), M, PT)
    if mix_node12 is None or mix_node25 is None or mix_node15 is None: mix_node125 = None
    else: mix_node125 = MCTS(forecast, real, get_mix_seq3(mix_node12, mix_node25, mix_node15, 1, 2, 5), M, PT)
    if mix_node13 is None or mix_node34 is None or mix_node14 is None: mix_node134 = None
    else: mix_node134 = MCTS(forecast, real, get_mix_seq3(mix_node13, mix_node34, mix_node14, 1, 3, 4), M, PT)
    if mix_node13 is None or mix_node35 is None or mix_node15 is None: mix_node135 = None
    else: mix_node135 = MCTS(forecast, real, get_mix_seq3(mix_node13, mix_node35, mix_node15, 1, 3, 5), M, PT)
    if mix_node14 is None or mix_node45 is None or mix_node15 is None: mix_node145 = None
    else: mix_node145 = MCTS(forecast, real, get_mix_seq3(mix_node14, mix_node45, mix_node15, 1, 4, 5), M, PT)
    if mix_node23 is None or mix_node34 is None or mix_node24 is None: mix_node234 = None
    else: mix_node234 = MCTS(forecast, real, get_mix_seq3(mix_node23, mix_node34, mix_node24, 2, 3, 4), M, PT)
    if mix_node23 is None or mix_node35 is None or mix_node25 is None: mix_node235 = None
    else: mix_node235 = MCTS(forecast, real, get_mix_seq3(mix_node23, mix_node35, mix_node25, 2, 3, 5), M, PT)
    if mix_node24 is None or mix_node45 is None or mix_node25 is None: mix_node245 = None
    else: mix_node245 = MCTS(forecast, real, get_mix_seq3(mix_node24, mix_node45, mix_node25, 2, 4, 5), M, PT)
    if mix_node34 is None or mix_node45 is None or mix_node35 is None: mix_node345 = None
    else: mix_node345 = MCTS(forecast, real, get_mix_seq3(mix_node34, mix_node45, mix_node35, 3, 4, 5), M, PT)

    three_dim = [mix_node123, mix_node124, mix_node125, mix_node134, mix_node135, mix_node145, mix_node234, mix_node235, mix_node245, mix_node345]
    print('three dimension result')
    for d in three_dim:
        if d is None: print('None')
        else: print(d.seq)
    print()

    #layer4 搜索
    if mix_node123 is None or mix_node124 is None or mix_node134 is None or mix_node234 is None: mix_node1234 = None
    else: mix_node1234 = MCTS(forecast, real, get_mix_seq4(mix_node123, mix_node124, mix_node134, mix_node234, 1, 2, 3, 4), M, PT)
    if mix_node123 is None or mix_node125 is None or mix_node135 is None or mix_node235 is None: mix_node1235 = None
    else: mix_node1235 = MCTS(forecast, real, get_mix_seq4(mix_node123, mix_node125, mix_node135, mix_node235, 1, 2, 3, 5), M, PT)
    if mix_node124 is None or mix_node125 is None or mix_node145 is None or mix_node245 is None: mix_node1245 = None
    else: mix_node1245 = MCTS(forecast, real, get_mix_seq4(mix_node124, mix_node125, mix_node145, mix_node245, 1, 2, 4, 5), M, PT)
    if mix_node134 is None or mix_node135 is None or mix_node145 is None or mix_node345 is None: mix_node1345 = None
    else: mix_node1345 = MCTS(forecast, real, get_mix_seq4(mix_node134, mix_node135, mix_node145, mix_node345, 1, 3, 4, 5), M, PT)
    if mix_node234 is None or mix_node235 is None or mix_node245 is None or mix_node345 is None: mix_node2345 = None
    else: mix_node2345 = MCTS(forecast, real, get_mix_seq4(mix_node234, mix_node235, mix_node245, mix_node345, 2, 3, 4, 5), M, PT)

    four_dim = [mix_node1234, mix_node1235, mix_node1245, mix_node1345, mix_node2345]
    print('four dimension result')
    for d in four_dim:
        if d is None: print('None')
        else: print(d.seq)
    print()

    #layer5 搜索
    if mix_node1234 is None or mix_node1235 is None or mix_node1245 is None or mix_node1345 is None or mix_node2345 is None: mix_node12345 = None
    else: mix_node12345 = MCTS(forecast, real, get_mix_seq5(mix_node1234, mix_node1235, mix_node1245, mix_node1345, mix_node2345), M, PT)
    print('five dimension result')
    if mix_node12345 is None: print('None')
    else: print(mix_node12345.seq)
    print()
    

    # 从Bset中选取拥有最大PS值的Rset
    result_seq = []
    result_Q = 0
    for node in one_dim:
        if node is not None and node.Q > result_Q:
            result_Q = node.Q
            result_seq = node.seq
    for node in two_dim:
        if node is not None and node.Q > result_Q:
            result_Q = node.Q
            result_seq = node.seq
    for node in three_dim:
        if node is not None and node.Q > result_Q:
            result_Q = node.Q
            result_seq = node.seq
    for node in four_dim:
        if node is not None and node.Q > result_Q:
            result_Q = node.Q
            result_seq = node.seq
    if mix_node12345 is not None and mix_node12345.Q > result_Q:
        result_Q = mix_node12345.Q
        result_seq = mix_node12345.seq


    return result_seq, result_Q


if __name__ == '__main__':
    # M 是最大搜索次数
    M = 5
    # M = int(sys.argv[1])
    # PT 是Q值的阀值
    PT = 0.75
    # PT = float(sys.argv[2])
    # 5维
    forecast = np.load(file='./test_data/Abnormalytime_forecast_PV_table/1539894000000.npy')  #(150, 15, 10, 37, 6)
    real = np.load(file='./test_data/Abnormalytime_real_PV_table/1539894000000.npy')
    # forecast = np.load(file=sys.argv[3])  #(150, 15, 10, 37, 6)
    # real = np.load(file=sys.argv[4])
    forecast = np.array(forecast)
    real = np.array(real)


   
    # 测试数据
    # dimension, element, 随便写的
    dim1_name = ['Mobile', 'Unicom']
    dim2_name = ['Beijing', 'Shanghai', 'Guangzhou']
    dim3_name = ['a', 'b']
    dim4_name = ['1', '2']
    dim5_name = ['!', '@']
    
    start_time = datetime.datetime.now()
    name, Q = get_result(dim1_name, dim2_name, dim3_name, dim4_name, dim5_name, forecast, real, M, PT)
    end_time = datetime.datetime.now()


    print('---------------------------------------------------------------')
    # print("异常时刻：")
    # print(sys.argv[3])
    print("运行时间：")
    print(end_time-start_time)
    print ("根因组合: ")
    print (json.dumps(name, ensure_ascii=False))#.encode("utf8")
    print ("组合得分: ")
    print (Q)
    print('---------------------------------------------------------------')
    print()
