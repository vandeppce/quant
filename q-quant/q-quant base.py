import math
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import norm

# Black - Scholes公式做欧式期权定价
def call_option_pricer(spot, strike, maturity, r, vol):
    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * maturity) / vol / math.sqrt(maturity)
    d2 = d1 - vol * math.sqrt(maturity)

    price = spot * norm.cdf(d1) - strike * math.exp(-r * maturity) * norm.cdf(d2)
    return price

spot = 2.45
strike = 2.50
maturity = 0.25
r = 0.05
vol = 0.25
print("期权价格：%.4f" % call_option_pricer(spot, strike, maturity, r, vol))

# 使用循环计算一组期权
portfolioSize = range(1, 10000, 500)
'''
timeSpent = []
for size in portfolioSize:
    now = time.time()
    strikes = np.linspace(2.0, 3.0, size)
    for i in range(size):
        res = call_option_pricer(spot, strikes[i], maturity, r, vol)
    timeSpent.append(time.time() - now)

sns.set(style="ticks")
plt.figure(figsize = (12,8))
plt.bar(portfolioSize, timeSpent, color = 'r', width =300)
plt.grid(True)
plt.title('期权计算时间耗时（单位：秒）', fontsize = 18)
plt.ylabel('时间（s)', fontsize = 15)
plt.xlabel('组合数量', fontsize = 15)
plt.show()
'''
'''
# 使用numpy计算
# 使用numpy的向量函数重写Black - Scholes公式
def call_option_pricer_numpy(spot, strike, maturity, r, vol):
    d1 = (np.log(spot / strike) + (r + 0.5 * vol * vol) * maturity) / vol / np.sqrt(maturity)
    d2 = d1 - vol * np.sqrt(maturity)

    price = spot * norm.cdf(d1) - strike * np.exp(-r * maturity) * norm.cdf(d2)
    return price

timeSpentNumpy = []
for size in portfolioSize:
    now = time.time()
    strikes = np.linspace(2.0,3.0, size)
    res = call_option_pricer_numpy(spot, strikes, maturity, r, vol)
    timeSpentNumpy.append(time.time() - now)

sns.set(style="ticks")
plt.figure(figsize = (12,8))
plt.bar(portfolioSize, timeSpentNumpy, color = 'r', width =300)
plt.grid(True)
plt.title('期权计算时间耗时（单位：秒）', fontsize = 18)
plt.ylabel('时间（s)', fontsize = 15)
plt.xlabel('组合数量', fontsize = 15)
plt.imshow()
'''

# 使用scipy做仿真计算
# 期权价格的计算方法中有一类称为 蒙特卡洛 方法。这是利用随机抽样的方法，
# 模拟标的股票价格随机游走，计算期权价格（未来的期望）
# 标准正态分布的随机数获取，可以方便的求助于 scipy 库

'''
# 期权计算的蒙特卡洛方法
def call_option_pricer_monte_carlo(spot, strike, maturity, r, vol, numOfPath = 5000):
    randomSeries = scipy.random.randn(numOfPath)
    s_t = spot * np.exp((r - 0.5 * vol * vol) * maturity + randomSeries * vol * np.sqrt(maturity))
    sumValue = np.maximum(s_t - strike, 0.0).sum()
    price = np.exp(-r * maturity) * sumValue / numOfPath
    return price

print('期权价格（蒙特卡洛） : %.4f' % call_option_pricer_monte_carlo(spot, strike, maturity, r, vol))

# 实验从1000次模拟到50000次模拟的结果，每次同样次数的模拟运行100遍
pathScenario = range(1000, 50000, 1000)
numberOfTrials = 100

confidenceIntervalUpper = []
confidenceIntervalLower = []
means = []
for scenario in pathScenario:
    res = np.zeros(numberOfTrials)
    for i in range(numberOfTrials):
        res[i] = call_option_pricer_monte_carlo(spot, strike, maturity, r, vol, numOfPath = scenario)
    means.append(res.mean())
    confidenceIntervalUpper.append(res.mean() + 1.96*res.std())
    confidenceIntervalLower.append(res.mean() - 1.96*res.std())

# 蒙特卡洛方法会有收敛速度的考量。这里我们可以看到随着模拟次数的上升，仿真结果的置信区间也在逐渐收敛。
plt.figure(figsize = (12,8))
tabel = np.array([means,confidenceIntervalUpper,confidenceIntervalLower]).T
plt.plot(pathScenario, tabel)
plt.title('期权计算蒙特卡洛模拟')
plt.legend(['均值', '95%置信区间上界', '95%置信区间下界'])
plt.ylabel('价格')
plt.xlabel('模拟次数')
plt.grid(True)
plt.imshow()
'''

# 计算隐含波动率
# 作为BSM期权定价最重要的参数，波动率σ是标的资产本身的波动率。
# 我们更关心的是当时的报价所反映的市场对波动率的估计，
# 这个估计的波动率称为隐含波动率（Implied Volatility）。
# 这里的过程实际上是在BSM公式中，假设另外4个参数确定，期权价格已知，反解σ
# 由于对于欧式看涨期权而言，其价格为对应波动率的单调递增函数，所以这个求解过程是稳定可行的。
# 一般来说我们可以类似于试错法来实现。在scipy中已经有很多高效的算法可以为我们所用，例如Brent算法:

# 目标函数，目标价格由target确定
class cost_function:
    def __init__(self, target):
        self.targetValue = target

    def __call__(self, x):
        return call_option_pricer(spot, strike, maturity, r, x) - self.targetValue

# 假设我们使用vol初值作为目标
target = call_option_pricer(spot, strike, maturity, r, vol)
cost_sample = cost_function(target)

# 使用Brent算法求解
impliedVol = scipy.optimize.brentq(cost_sample, 0.01, 0.5)
print("真实波动率：%.2f" % (vol * 100, ) + '%')
print("隐含波动率：%.2f" % (impliedVol * 100, ) + '%')