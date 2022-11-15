import pandas as pd
import numpy as np
from pandas import Series, DataFrame

'''
# Series数据结构
a = np.random.randn(5)
print("a is an array:")
print(a)
s = Series(a)
print("s is a Series:")
print(s)
# 指定index
s = Series(a, index=['a', 'b', 'c', 'd', 'e'])
print(s.index)
# 指定name
s = Series(a, index=['a', 'b', 'c', 'd', 'e'], name='my_series')
print(s.name)

# Series数据访问
s = Series(np.random.randn(10),index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
# 下标访问
print(s[0])
print(s[:2])
print(s[[2, 0, 4]])
# 索引访问
print(s[['e', 'i']])
# 过滤条件
print(s[s > 0.5])
print('e' in s)
'''
'''
# DataFrame数据结构
# DataFrame是将数个Series按列合并而成的二维数据结构，每一列单独取出来是一个Series
# 创建Series
# 从字典创建
d = {'one': Series([1., 2., 3.], index=['a', 'b', 'c']),
     'two': Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = DataFrame(d)
print(df)
# 可以指定所需的行和列，若字典中不含有对应的元素，则置为NaN：
df = DataFrame(d, index=['r', 'd', 'd'], columns=['two', 'three'])
print(df)
# 可以使用dataframe.index和dataframe.columns来查看DataFrame的行和列，
# dataframe.values则以数组的形式返回DataFrame的元素：
print("DataFrame index:")
print(df.index)
print("DataFrame columns:")
print(df.columns)
print("DataFrame values:")
print(df.values)
# DataFrame也可以从值是数组的字典创建，但是各个数组的长度需要相同：
d = {"one": [1., 2., 3., 4.], "two": [4., 3., 2., 1.]}
df = DataFrame(d, index=['a', 'b', 'c', 'd'])
print(df)
# 值非数组时，没有这一限制，并且缺失值补成NaN：
d = [{'a': 1.6, 'b': 2}, {'a': 3, 'b': 6, 'c': 9}]
df = DataFrame(d)
print(df)

# 使用concat函数基于Series或者DataFrame创建一个DataFrame
a = Series(range(5))
b = Series(np.linspace(4, 20, 5))
df = pd.concat([a, b], axis=1)
print(df)
#其中的axis=1表示按列进行合并，axis=0表示按行合并，并且，Series都处理成一列，
# 所以这里如果选axis=0的话，将得到一个10×1的DataFrame。
# 下面这个例子展示了如何按行合并DataFrame成一个大的DataFrame：
'''
df = DataFrame()
index = ['alpha', 'beta', 'gamma', 'delta', 'eta']
for i in range(5):
    a = DataFrame([np.linspace(i, 5 * i, 5)], index=[index[i]]) # 0 0 0 0 0
    df = pd.concat([df, a], axis=0)
print(df)

# DataFrame数据的访问
# DataFrame是以列作为操作的基础的，全部操作都想象成先从DataFrame里取一列，再从这个Series取元素即可。
# 可以用datafrae.column_name选取列，也可以使用dataframe[]操作选取列，我们可以马上发现前一种方法只能选取一列，
# 而后一种方法可以选择多列。若DataFrame没有列名，[]可以使用非负整数，也就是“下标”选取列；
# 若有列名，则必须使用列名选取，另外datafrae.column_name在没有列名的时候是无效的
print(df[1])
df.columns = ['a', 'b', 'c', 'd', 'e']
print(df['b'])
print(df.b)
print(df[['a', 'd']])
# 访问特定的元素可以如Series一样使用下标或者是索引:
print(df['b'][2])
print(df['b']['gamma'])
# 若需要选取行，可以使用dataframe.iloc按下标选取，或者使用dataframe.loc按索引选取：
print(df.iloc[1])
print(df.loc['beta'])
# 选取行还可以使用切片的方式或者是布尔类型的向量：
print("Selecting by slices:")
print(df[1: 3])
bool_vec = [True, False, True, True, False]
print("Selecting by boolen vector:")
print(df[bool_vec])

# 行列组合起来选择数据
print(df[['b', 'd']].iloc[[1, 3]])
print(df.iloc[[1, 3]][['b', 'd']])
print(df[['b', 'd']].loc[['beta', 'delta']])
print(df.loc[['beta', 'delta']][['b', 'd']])
# 如果不是需要访问特定行列，而只是某个特殊位置的元素的话，dataframe.at和dataframe.iat是最快的方式，
# 它们分别用于使用索引和下标进行访问：
print(df.iat[2, 3]) # 行前列后
print(df.at['gamma', 'd'])