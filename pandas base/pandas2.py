import numpy as np
import pandas as pd
from pandas import Series, DataFrame

pd.set_option('display.width', 200)

# 数据创建的其他方式
# 创建一个以日期为元素的Series
dates = pd.date_range('20150101', periods=5)
print(dates)
# 将这个日期Series作为索引赋给一个DataFrame：
df = pd.DataFrame(np.random.randn(5, 4), index=dates, columns=list('ABCD'))
print(df)
# 只要是能转换成Series的对象，都可以用于创建DataFrame：
df2 = pd.DataFrame({'A': 1, 'B': pd.Timestamp('20150214'), 'C': pd.Series(1.6, index=(range(4))),
                    'D': np.array([4] * 4), 'E': 'hello pandas!'})
print(df2)

# 数据的查看
# 使用DataAPI接口获取股票数据
stock_list = ['000001.XSHE', '000002.XSHE', '000568.XSHE', '000625.XSHE', '000768.XSHE', '600028.XSHG', '600030.XSHG', '601111.XSHG', '601390.XSHG', '601998.XSHG']
raw_data = DataAPI.MktEqudGet(secID=stock_list, beginDate='20210101', endDate='20210131', pandas='1')
df = raw_data[['secID', 'tradeDate', 'secShortName', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice', 'turnoverVol']]

# 预览数据，预览头尾几行
print("Head of this DataFrame:")
print(df.head())
print("Tail of this DataFrame:")
print(df.tail(3))

# dataframe.describe()提供了DataFrame中纯数值数据的统计信息：
print(df.describe())

# 对数据的排序将便利我们观察数据，DataFrame提供了两种形式的排序。
# 一种是按行列排序，即按照索引（行名）或者列名进行排序，可调用dataframe.sort_index，
# 指定axis=0表示按索引（行名）排序，axis=1表示按列名排序，并可指定升序或者降序：
print("Order by column names, descending:")
print(df.sort_index(axis=1, ascending=False).head())

# 第二种排序是按值排序，可指定列名和排序方式，默认的是升序排序：
print("Order by column value, ascending:")
print(df.sort(columns='tradeDate').head())
print("Order by multiple columns value:")
df = df.sort(columns=['tradeDate', 'secID'], ascending=[False, True])
print(df.head())

