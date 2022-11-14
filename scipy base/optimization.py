import numpy as np
import scipy.optimize as opt

# 无约束优化问题
'''
# 以Rosenbrock函数为例，进行优化
def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

# Nelder-Mead单纯形法，不使用函数梯度
# 在minimize中使用method='powell'来指定使用Powell's method。
# 这两种简单的方法并不使用函数的梯度，在略微复杂的情形下收敛速度比较慢
x_0 = np.array([0.5, 1.6, 1.1, 0.8, 1.2])
res = opt.minimize(rosen, x_0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print("Result of minimizing Rosenbrock function via Nelder-Mead Simplex algorithm:")
print(res)

# Broyden-Fletcher-Goldfarb-Shanno法(BFGS)，用到了梯度信息
# 定义梯度向量的计算函数
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der

# 梯度信息的引入在minimize函数中通过参数jac指定：
res = opt.minimize(rosen, x_0, method="BFGS", jac=rosen_der, options={'disp': True})
print("Result of minimizing Rosenbrock function via Broyden-Fletcher-Goldfarb-Shanno algorithm:")
print(res)

# 牛顿共轭梯度法（Newton-Conjugate-Gradient algorithm）
# 牛顿法是收敛速度最快的方法，其缺点在于要求Hessian矩阵（二阶导数矩阵）。
# 牛顿法大致的思路是采用泰勒展开的二阶近似，若Hessian矩阵是正定的，函数的局部最小值可以通过使上面的二次型的一阶导数等于0来获取
# 可使用共轭梯度近似Hessian矩阵的逆矩阵
def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H

res = opt.minimize(rosen, x_0, method="Newton-CG", jac=rosen_der, hess=rosen_hess, options={'xtol': 1e-8, 'disp': True})
print("Result of minimizing Rosenbrock function via Newton-Conjugate-Gradient algorithm (Hessian):")
print(res)
# 对于一些大型的优化问题，Hessian矩阵将异常大，牛顿共轭梯度法用到的仅是Hessian矩阵和一个任意向量的乘积，
# 为此，用户可以提供两个向量，一个是Hessian矩阵和一个任意向量p的乘积，另一个是向量p，这就减少了存储的开销
def rosen_hess_p(x, p):
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
    Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
               -400*x[1:-1]*p[2:]
    Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
    return Hp

res = opt.minimize(rosen, x_0, method='Newton-CG', jac=rosen_der, hessp=rosen_hess_p, options={'xtol': 1e-8, 'disp': True})
print("Result of minimizing Rosenbrock function via Newton-Conjugate-Gradient algorithm (Hessian times arbitrary vector):")
print(res)
'''

# 约束优化问题

# 目标函数，f(x,y)=2xy+2x-x^2-2y^2
# 约束条件，x^3-y=0, y-1>=0

# 定义目标函数及其导数为：
def func(x, sign=1.0):
    """ Objective function """
    return sign*(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)

def func_deriv(x, sign=1.0):
    """ Derivative of objective function """
    dfdx0 = sign*(-2*x[0] + 2*x[1] + 2)
    dfdx1 = sign*(2*x[0] - 4*x[1])
    return np.array([ dfdx0, dfdx1 ])
# 其中sign表示求解最小或者最大值，我们进一步定义约束条件：
cons = ({'type': 'eq',  'fun': lambda x: np.array([x[0]**3 - x[1]]), 'jac': lambda x: np.array([3.0*(x[0]**2.0), -1.0])},
        {'type': 'ineq', 'fun': lambda x: np.array([x[1] - 1]), 'jac': lambda x: np.array([0.0, 1.0])})
# 最后我们使用SLSQP（Sequential Least SQuares Programming optimization algorithm）方法
# 进行约束问题的求解（作为比较，同时列出了无约束优化的求解）：
res = opt.minimize(func, [-1.0, 1.0], args=(-1.0,), jac=func_deriv, method='SLSQP', options={'disp': True})
print("Result of unconstrained optimization:")
print(res)
res = opt.minimize(func, [-1.0, 1.0], args=(-1.0,), jac=func_deriv, constraints=cons, method='SLSQP', options={'disp': True})
print("Result of constrained optimization:")
print(res)