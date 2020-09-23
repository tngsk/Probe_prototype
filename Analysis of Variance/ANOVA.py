#!/usr/bin/env python
# coding: utf-8

# # Analysis of variance
# 
# 1因子の場合

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#  データの作成
data = np.array([
    [33,31,33],
    [30,29,31],
    [33,28,32],
    [29,29,32],
    [32,27,36]])
df = pd.DataFrame(data,columns=['Mimizu','Batta','Mix'],index=[1,2,3,4,5])
df


# In[3]:


# 全データの平均
all_mean = df.stack().mean()
all_mean


# In[4]:


# 列の効果
df_effect = df.mean(axis=0) - all_mean
df_effect


# In[5]:


# 誤差 - データから列の平均を引く
df_error = df - df.mean()
df_error


# In[6]:


# 誤差の合計は0になる
df_error.sum()


# 不偏分散の式
# $$
# V = \frac{\sum(x_i - \bar{x})^2}{n-1}
# $$

# In[7]:


# 列の効果の不偏分散 V1

dfn = df.columns.size - df_effect.mean().size #分子の自由度
V_1 = np.sum(df.index.size * (np.square(df_effect - df_effect.mean() ))) / dfn
V_1


# In[8]:


# 誤差の不偏分散 V2

err = df_error.stack()
dfd = err.size - df_error.mean().size #分母の自由度
V_2 = np.square(err - err.mean()).sum() / dfd
V_2


# In[9]:


# F値 V1/V2
F = V_1 / V_2
F


# In[10]:


from scipy.stats import f

five = f.ppf(0.95, dfn,dfd)
one = f.ppf(0.99, dfn, dfd)
print('上側確率 5%',five)
print('上側確率 1%',one)


# # Analysis of Variance
# 2因子の場合

# In[47]:


import numpy as np
import pandas as pd
from scipy.stats import f


# In[2]:


data = np.array([
    [29,35,29,29],
    [26,33,30,28],
    [32,34,33,34]
    ])
df = pd.DataFrame(data, columns=['火','砂','腐','粘'], index=['ミ','バ','混'])
df


# In[5]:


# 行の効果（餌の効果）
row_effect = df.mean(axis=1) - df.stack().mean()
row_effect


# In[6]:


# 列の効果（土の効果）
col_effect = df.mean(axis=0) - df.stack().mean()
col_effect


# In[29]:


# 誤差
df_error = df - df.stack().mean()
df_error


# In[30]:


# 誤差を分離する
df_error = df_error.subtract(row_effect, axis=0)
df_error = df_error.subtract(col_effect, axis=1)
df_error


# In[40]:


# 誤差の不偏分散
phi_2 = (df.columns.size - 1) * (df.index.size -1)
V_2 = np.sum(np.square(df_error.stack())) / phi_2
V_2


# In[45]:


# 行の不偏分散
phi_11 = row_effect.size - df.stack().mean().size
V_11 = np.sum(np.square(row_effect) * df.columns.size) / phi_11
V_11


# In[48]:


# F1 行の効果の有意差
F_1 = V_11 / V_2

five = f.ppf(0.95, phi_11,phi_2)
one = f.ppf(0.99, phi_11,phi_2)
print('上側確率 5%',five)
print('上側確率 1%',one)
print('F1 = ',F_1)


# In[49]:


# 列の不偏分散
phi_12 = col_effect.size - df.stack().mean().size
V_12 = np.sum(np.square(col_effect) * df.index.size) / phi_12
V_12


# In[52]:


# F2 列の効果の有意差

F_2 = V_12 / V_2

five = f.ppf(0.95, phi_12,phi_2)
one = f.ppf(0.99, phi_12,phi_2)
print('上側確率 5%',five)
print('上側確率 1%',one)
print('F2 = ',F_2)




# Analysis of Variance
# 3因子の場合 ラテン方格

import numpy as np

data = np.array([
    [31,33,31,33],
    [34,24,29,29],
    [35,32,30,39],
    [30,25,34,27]])

# 割り付け
def factor_matrix(factor) -> np.ndarray:
    num = len(factor)
    mat =[]
    for i in range(0,num):
        for j in range(0,num):
            index = (i + j) % num
            mat.append(factor[index])
    return np.array(mat).reshape(num,num)

factor_mat = factor_matrix(['A','B','C','D'])
'''
[['A' 'B' 'C' 'D']
 ['B' 'C' 'D' 'A']
 ['C' 'D' 'A' 'B']
 ['D' 'A' 'B' 'C']]
'''


# 全体平均
all_mean = data.mean()
print(all_mean)

# 列の効果
col_effect = data.mean(axis=0) - all_mean
col_effect = np.vstack([col_effect] * 4)
'''
[[ 1.5 -2.5  0.   1. ]
 [ 1.5 -2.5  0.   1. ]
 [ 1.5 -2.5  0.   1. ]
 [ 1.5 -2.5  0.   1. ]]
'''

# 行の効果
row_effect = data.mean(axis=1) - all_mean
row_effect = np.hstack([row_effect[:,np.newaxis]] * 4)
'''
[[ 1.  1.  1.  1.]
 [-2. -2. -2. -2.]
 [ 3.  3.  3.  3.]
 [-2. -2. -2. -2.]]
'''

# A, B, C, D それぞれの効果

a_data = (factor_mat == 'A') * data
a_effect = np.mean(a_data[a_data > 0]) - all_mean
a_effect *= (factor_mat == 'A')
'''
[[-2.25 -0.   -0.   -0.  ]
 [-0.   -0.   -0.   -2.25]
 [-0.   -0.   -2.25 -0.  ]
 [-0.   -2.25 -0.   -0.  ]]
'''

b_data = (factor_mat == 'B') * data
b_effect = np.mean(b_data[b_data > 0]) - all_mean
b_effect *= (factor_mat == 'B')
'''
[[0. 4. 0. 0.]
 [4. 0. 0. 0.]
 [0. 0. 0. 4.]
 [0. 0. 4. 0.]]
'''

c_data = (factor_mat == 'C') * data
c_effect = np.mean(c_data[c_data > 0]) - all_mean
c_effect *= (factor_mat == 'C')
'''
[[-0.   -0.   -1.75 -0.  ]
 [-0.   -1.75 -0.   -0.  ]
 [-1.75 -0.   -0.   -0.  ]
 [-0.   -0.   -0.   -1.75]]
'''

d_data = (factor_mat == 'D') * data
d_effect = np.mean(d_data[d_data > 0]) - all_mean
d_effect *= (factor_mat == 'D')
'''
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
'''

# 誤差の計算
error = data - all_mean - col_effect - row_effect - a_effect - b_effect - c_effect
'''
[[-0.25 -0.5   0.75  0.  ]
 [-0.5  -0.75  0.    1.25]
 [ 1.25  0.5  -1.75  0.  ]
 [-0.5   0.75  1.   -1.25]]
'''

# 誤差の不偏分散
phi_2 = 6.0
V_2 = np.sum(np.square(error)) / phi_2

# row
phi_11 = 3.0
V_11 = np.sum(np.square(row_effect)) / phi_11
F_1 = V_11 / V_2

# col
phi_12 = 3.0
V_12 = np.sum(np.square(col_effect)) / phi_12
F_2 = V_12 / V_2

# factor 3
phi_13 = 3.0
V_13 = (np.sum(np.square(a_effect)) + np.sum(np.square(b_effect)) + np.sum(np.square(c_effect)) + np.sum(np.square(d_effect))) / phi_13
F_3 = V_13 / V_2

# 検定
from scipy.stats import f

five = f.ppf(0.95, phi_11,phi_2)
one = f.ppf(0.99, phi_11,phi_2)
print('上側確率 5%',five)
print('上側確率 1%',one)
print(f'F_1 = {F_1}')
print(f'F_2 = {F_2}')
print(f'F_3 = {F_3}')

'''
上側確率 5% 4.757062663089414
上側確率 1% 9.779538240923273
F_1 = 12.521739130434781
F_2 = 6.608695652173912
F_3 = 16.782608695652172
'''



