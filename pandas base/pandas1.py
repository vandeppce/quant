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

# 数据访问
# 使用:来获取部行或者全部列
print(df.iloc[1:4][:])

# 扩展上篇介绍的使用布尔类型的向量获取数据的方法，可以很方便地过滤数据，例如，我们要选出收盘价在均值以上的数据
print(df[df.closePrice > df.closePrice.mean()].head())

# isin()函数可以方便地过滤DataFrame中的数据
print(df[df['secID'].isin(['601628.XSHG', '000001.XSHE', '600030.XSHG'])].head())

# 处理缺失数据
# 在访问数据的基础上，我们可以更改数据，例如，修改某些元素为缺失值：
df['openPrice'][df['secID'] == '000001.XSHE'] = np.nan
df['highestPrice'][df['secID'] == '601111.XSHG'] = np.nan
df['lowestPrice'][df['secID'] == '601111.XSHG'] = np.nan
df['closePrice'][df['secID'] == '000002.XSHE'] = np.nan
df['turnoverVol'][df['secID'] == '601111.XSHG'] = np.nan
print(df.head(10))

# 处理缺失数据有多种方式，通常使用dataframe.dropna()，dataframe.dropna()可以按行丢弃带有nan的数据
# 若指定how='all'（默认是'any'），则只在整行全部是nan时丢弃数据；
# 若指定thresh，则表示当某行数据非缺失列数超过指定数值时才保留
# 要指定根据某列丢弃可以通过subset完成。
print("Data size before filtering:")
print(df.shape)
print("Drop all rows that have any NaN values:")
print("Data size after filtering:")
print(df.dropna().shape)
# print(df.dropna().head(10))

print("Drop only if all columns are NaN:")
print("Data size after filtering:")
print(df.dropna(how='all').shape)
# print(df.dropna(how='all').head(10))

print("Drop rows who do not have at least six values that are not NaN")
print("Data size after filtering:")
print(df.dropna(thresh=6).shape)
# print(df.dropna(thresh=6).head(10))

print("Drop only if NaN in specific column:")
print("Data size after filtering:")
print(df.dropna(subset=['closePrice']).shape)
# print(df.dropna(subset=['closePrice']).head(10))

# 有数据缺失时也未必是全部丢弃，dataframe.fillna(value=value)可以指定填补缺失值的数值
print(df.fillna(value=20150101).head())

# 数据操作
# Series和DataFrame的类函数提供了一些函数，如mean()、sum()等，指定0按列进行，指定1按行进行：
df = raw_data[['secID', 'tradeDate', 'secShortName', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice', 'turnoverVol']]
print(df.mean(0))

# value_counts函数可以方便地统计频数：
print(df['closePrice'].value_counts().head())

# 在panda中，Series可以调用map函数来对每个元素应用一个函数，
# DataFrame可以调用apply函数对每一列（行）应用一个函数，
# applymap对每个元素应用一个函数。这里面的函数可以是用户自定义的一个lambda函数，
# 也可以是已有的其他函数。下例展示了将收盘价调整到[0, 1]区间：
print(df[['closePrice']].apply(lambda x: (x - x.min()) / (x.max() - x.min())).head())

# 使用append可以在Series后添加元素，以及在DataFrame尾部添加一行：
dat1 = df[['secID', 'tradeDate', 'closePrice']].head()
dat2 = df[['secID', 'tradeDate', 'closePrice']].iloc[2]
print("Before appending:")
print(dat1)
dat = dat1.append(dat2, ignore_index=True)
print("After appending:")
print(dat)

# DataFrame可以像在SQL中一样进行合并，
# 在上篇中，我们介绍了使用concat函数创建DataFrame，这就是一种合并的方式。
# 另外一种方式使用merge函数，需要指定依照哪些列进行合并，
# 下例展示了如何根据security ID和交易日合并数据
dat1 = df[['secID', 'tradeDate', 'closePrice']]
dat2 = df[['secID', 'tradeDate', 'turnoverVol']]
dat = dat1.merge(dat2, on=['secID'])
print("The first DataFrame:")
print(dat1.head())
print("The second DataFrame:")
print(dat2.head())
print("Merged DataFrame:")
print(dat.head())
dat = dat1.merge(dat2, on=['secID', 'tradeDate'])
print("Merged DataFrame:")
print(dat.head())

# DataFrame另一个强大的函数是groupby，可以十分方便地对数据分组处理，
# 我们对2015年一月内十支股票的开盘价，最高价，最低价，收盘价和成交量求平均值：
df_grp = df.groupby('secID') # 用secID分组
grp_mean = df_grp.mean()
print(grp_mean)

# 获取每支股票的最新数据。用drop_duplicates实现这个功能，首先对数据按日期排序，再按security ID去重
df2 = df.sort(columns=['secID', 'tradeDate'], ascending=[True, False])
print(df2.drop_duplicates(subset='secID'))

# 若想要保留最老的数据，可以在降序排列后取最后一个记录，
# 通过指定take_last=True（默认值为False，取第一条记录）可以实现
print(df2.drop_duplicates(subset='secID', take_last=True))

# 数据可视化
# set_index('tradeDate')['closePrice']表示将DataFrame的'tradeDate'这一列作为索引，
# 将'closePrice'这一列作为Series的值，返回一个Series对象，随后调用plot函数绘图
dat = df[df['secID'] == '600028.XSHG'].set_index('tradeDate')['closePrice']
dat.plot(title="Close Price of SINOPEC (600028) during Jan, 2015")