from scipy import interpolate

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import date
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 使用scipy做函数插值
# print(dir(interpolate)[:5])
'''
# 三角函数插值
x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interpolate.interp1d(x, y)
f2 = interpolate.interp1d(x, y, kind='cubic')
xnew = np.linspace(0, 10, num=41, endpoint=True)
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()
'''
# 期权波动率曲面构造
# 市场上期权价格一般以隐含波动率的形式报出，一般来讲在市场交易时间，
# 交易员可以看到类似的波动率矩阵

pd.options.display.float_format = '{:,>.2f}'.format
dates = [date(2015, 3, 25), date(2015, 4, 25), date(2015, 6, 25), date(2015, 9, 25)]
strikes = [2.2, 2.3, 2.4, 2.5, 2.6]
blackVolMatrix = np.array([[ 0.32562851,  0.29746885,  0.29260648,  0.27679993],
                  [ 0.28841840,  0.29196629,  0.27385023,  0.26511898],
                  [ 0.27659511,  0.27350773,  0.25887604,  0.25283775],
                  [ 0.26969754,  0.25565971,  0.25803327,  0.25407669],
                  [ 0.27773032,  0.24823248,  0.27340796,  0.24814975]])
table = pd.DataFrame(blackVolMatrix * 100, index=strikes, columns=dates)
table.index.name = '行权价'
table.columns.name = '到期时间'
print('2015年3月3日10时波动率矩阵')
print(table)

# 交易员可以看到市场上离散值的信息，但是如果可以获得一些隐含的信息更好：
# 例如，在2015年6月25日以及2015年9月25日之间，波动率的形状会是怎么样的？
# 方差曲面插值
# 我们并不是直接在波动率上进行插值，而是在方差矩阵上面进行插值。
# 获取方差矩阵
evaluationDate = date(2015, 3, 3)
ttm = np.array([(d - evaluationDate).days / 365.0 for d in dates])
varianceMatrix = (blackVolMatrix ** 2) * ttm
print(varianceMatrix)

# 下面我们将在行权价方向以及时间方向同时进行线性插值
# 这个过程在scipy中可以直接通过interpolate模块下interp2d来实现
interp = interpolate.interp2d(ttm, strikes, varianceMatrix, kind='linear')
# 返回的interp对象可以用于获取任意点上插值获取的方差值：
print(interp(ttm[0], strikes[0]))

# 最后我们获取整个平面上所有点的方差值，再转换为波动率曲面。
sMeshes = np.linspace(strikes[0], strikes[-1], 400)
tMeshes = np.linspace(ttm[0], ttm[-1], 200)
interpolatedVarianceSurface = np.zeros((len(sMeshes), len(tMeshes)))
for i, s in enumerate(sMeshes):
    for j, t in enumerate(tMeshes):
        interpolatedVarianceSurface[i][j] = interp(t, s)
interpolatedVolatilitySurface = np.sqrt((interpolatedVarianceSurface / tMeshes))
print('行权价方向网格数：', np.size(interpolatedVolatilitySurface, 0))
print('到期时间方向网格数：', np.size(interpolatedVolatilitySurface, 1))
# 选取某一个到期时间上的波动率点，看一下插值的效果。这里我们选择到期时间最近的点：2015年3月25日：
plt.plot(strikes, blackVolMatrix[:, 0], 'o', sMeshes, interpolatedVolatilitySurface[:, 0], '--')
plt.show()

# 把整个曲面的图像画出来看看

maturityMesher, strikeMesher = np.meshgrid(tMeshes, sMeshes)
plt.figure(figsize=(16, 9))
ax = plt.gca(projection = '3d')
surface = ax.plot_surface(strikeMesher, maturityMesher, interpolatedVolatilitySurface * 100, cmap = cm.jet)
plt.colorbar(surface, shrink=0.75)
plt.title("2015.3.3")
plt.xlabel("strike")
plt.ylabel("maturity")
ax.set_zlabel(r"volatility(%)")
plt.show()