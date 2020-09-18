#!/usr/bin/env python
# coding: utf-8

# Analysis of Variance
# 2因子の場合

import numpy as np
import pandas as pd
from scipy.stats import f

# ----- データの準備

data = np.array([
    [29,35,29,29],
    [26,33,30,28],
    [32,34,33,34]
    ])
df = pd.DataFrame(data, columns=['火','砂','腐','粘'], index=['ミ','バ','混'])

# ----- 効果と誤差の計算

# 行の効果（餌の効果）
row_effect = df.mean(axis=1) - df.stack().mean()

# 列の効果（土の効果）
col_effect = df.mean(axis=0) - df.stack().mean()

# 誤差
df_error = df - df.stack().mean()

# ----- 誤差を分離する

# 誤差を分離する
df_error = df_error.subtract(row_effect, axis=0)
df_error = df_error.subtract(col_effect, axis=1)

# ----- 不偏分散の計算

# 誤差の不偏分散
phi_2 = (df.columns.size - 1) * (df.index.size -1)
V_2 = np.sum(np.square(df_error.stack())) / phi_2

# 行の不偏分散
phi_11 = row_effect.size - df.stack().mean().size
V_11 = np.sum(np.square(row_effect) * df.columns.size) / phi_11

# 列の不偏分散
phi_12 = col_effect.size - df.stack().mean().size
V_12 = np.sum(np.square(col_effect) * df.index.size) / phi_12

# ----- 有意差の確認

# F1 行の効果の有意差

F_1 = V_11 / V_2
five = f.ppf(0.95, phi_11,phi_2)
one = f.ppf(0.99, phi_11,phi_2)
print('上側確率 5%',five)
print('上側確率 1%',one)
print('F1 = ',F_1)

"""
上側確率 5% 5.143252849784718
上側確率 1% 10.92476650083833
F1 =  6.347368421052631
"""


# F2 列の効果の有意差

F_2 = V_12 / V_2
five = f.ppf(0.95, phi_12,phi_2)
one = f.ppf(0.99, phi_12,phi_2)
print('上側確率 5%',five)
print('上側確率 1%',one)
print('F2 = ',F_2)

"""
上側確率 5% 4.757062663089414
上側確率 1% 9.779538240923273
F2 =  5.136842105263158
"""

