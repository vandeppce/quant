import numpy as np
import scipy.stats as stats

# 生成随机数
'''
# 均匀分布
rv_unif = stats.uniform.rvs(size=10)
print(rv_unif)
# beta分布
rv_beta = stats.beta.rvs(size=10, a=4, b=2)
print(rv_beta)
'''

# 假设检验
'''
norm_dist = stats.norm(loc=0.5, scale=2)
n = 200
dat = norm_dist.rvs(size=n)
print("mean of data is: " + str(np.mean(dat)))
print("median of data is: " + str(np.median(dat)))
print("standard deviation of data is: " + str(np.std(dat)))

# 使用K-S检验查看数据是否服从假设分布,如果p-value很大，那么可以假设原数据通过了正态检验
mu = np.mean(dat)
sigma = np.std(dat)
stat_val, p_val = stats.kstest(dat, 'norm', (mu, sigma))
print("KS-statistic D = %6.3f p-value = %6.4f" % (stat_val, p_val))
# 在正态性的前提下，使用t检验可进一步检验这组数据的均值是不是0
# 单样本检验
stat_val, p_val = stats.ttest_1samp(dat, 0)
print("One-sample t-statistic D = %6.3f, p-value = %6.4f" % (stat_val, p_val))
# 我们看到p-value<0.05，即给定显著性水平0.05的前提下，我们应拒绝原假设：数据的均值为0
# 双样本检验
norm_dist2 = stats.norm(loc=-0.2, scale=1.2)
dat2 = norm_dist2.rvs(size=n//2)
stat_val, p_val = stats.ttest_ind(dat, dat2, equal_var=False)
print("Two-sample t-statistic D = %6.3f, p-value = %6.4f" % (stat_val, p_val))
# 在运用t检验时需要使用Welch's t-test，即指定ttest_ind中的equal_var=False。
# 我们同样得到了比较小的p-value$，在显著性水平0.05的前提下拒绝原假设，即认为两组数据均值不等。

# stats还提供其他大量的假设检验函数，如bartlett和levene用于检验方差是否相等stats还提供其他大量的假设检验函数，
# 如bartlett和levene用于检验方差是否相等；anderson_ksamp用于进行Anderson-Darling的K-样本检验等。
'''

# 其他函数

# 有时需要知道某数值在一个分布中的分位，或者给定了一个分布，求某分位上的数值。这可以通过cdf和ppf函数完成：
g_dist = stats.gamma(a=2)
print("quantiles of 2, 4, and 5:")
print(g_dist.cdf([2, 4, 5]))
print("Values of 25%, 50%, and 90%:")
print(g_dist.pdf([0.25, 0.5, 0.95]))

# 对于一个给定的分布，可以用moment很方便的查看分布的矩信息，例如我们查看N(0,1)的六阶原点矩：
n_moment = stats.norm.moment(6, loc=0, scale=1)
print(n_moment)

# describe函数提供对数据集的统计描述分析，包括数据样本大小，极值，均值，方差，偏度和峰度：
norm_dist = stats.norm(loc=0, scale=1.8)
dat = norm_dist.rvs(size=100)
info = stats.describe(dat)
print("Data size is: " + str(info[0]))
print("Minimum value is: " + str(info[1][0]))
print("Maximum value is: " + str(info[1][1]))
print("Arithmetic mean is: " + str(info[2]))
print("Unbiased variance is: " + str(info[3]))
print("Biased skewness is: " + str(info[4]))
print("Biased kurtosis is: " + str(info[5]))

# 当我们知道一组数据服从某些分布的时候，
# 可以调用fit函数来得到对应分布参数的极大似然估计（MLE, maximum-likelihood estimation）
norm_dist = stats.norm(loc=0, scale=1.8)
dat = norm_dist.rvs(size=100)
mu, sigma = stats.norm.fit(dat)
print("MLE of data mean:" + str(mu))
print("MLE of data standard deviation:" + str(sigma))

# pearsonr和spearmanr可以计算Pearson和Spearman相关系数，这两个相关系数度量了两组数据的相互线性关联程度：
norm_dist = stats.norm()
dat1 = norm_dist.rvs(size=100)
exp_dist = stats.expon()
dat2 = exp_dist.rvs(size=100)
cor, pval = stats.pearsonr(dat1, dat2)
print("Pearson correlation coefficient: " + str(cor))
cor = stats.spearmanr(dat1, dat2)
print("Spearman's rank correlation coefficient: " + str(cor))

# 线性回归
x = stats.chi2.rvs(3, size=50)
y = 2.5 + 1.2 * x + stats.norm.rvs(size=50, loc=0, scale=1.5)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print("Slope of fitted model is:" , slope)
print("Intercept of fitted model is:", intercept)
print("R-squared:", r_value**2)