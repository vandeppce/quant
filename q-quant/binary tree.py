import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

# 使用Black-Scholes模型构建二叉树
# 设置基本参数
ttm = 3.0        # 到期时间，单位年
tSteps = 25      # 时间方向步数
r = 0.03         # 无风险利率
d = 0.02         # 标的股息率
sigma = 0.2      # 波动率
strike = 100.0   # 期权行权价
spot = 100.0     # 标的现价

# 用作例子的树结构被称为 Jarrow - Rudd 树
dt = ttm / tSteps
up = math.exp((r - d - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt))
down = math.exp((r - d - 0.5 * sigma * sigma) * dt - sigma * math.sqrt(dt))
discount = math.exp(-r * dt)

lattice = np.zeros((tSteps + 1, tSteps + 1))
lattice[0][0] = spot
for i in range(tSteps):
    for j in range(i + 1):
        lattice[i + 1][j + 1] = up * lattice[i][j]
    lattice[i + 1][0] = down * lattice[i][0]

plt.plot(lattice[tSteps])
plt.show()

# 在节点上计算payoff
def call_payoff(spot):
    global strike
    return max(spot - strike, 0.0)

res = []
for i in range(tSteps + 1):
    res.append(call_payoff(lattice[tSteps][i]))
plt.plot(res)
plt.show()