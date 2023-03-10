import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'

import pandas as pd
import numpy as np
print("**************")

GloUse = {}
GloUse['train_file'] = ['data/' + 'train_FD00' + str(i) for i in range(1, 5)]
GloUse['test_file'] = ['data/' + 'test_FD00' + str(i) for i in range(1, 5)]
GloUse['RUL_file'] = ['data/' + 'RUL_FD00' + str(i) for i in range(1, 5)]
GloUse['sensor'] = [[2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
                    [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
                    [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
                    [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]] #挑选14个传感器数据
GloUse['SL'] = [14, 14, 14, 14]  # 传感器数
GloUse['train_units'] = [100, 260, 100, 249]
GloUse['test_units'] = [100, 259, 100, 248]
n = 1
# 读取数据
raw1 = np.loadtxt(GloUse['train_file'][n - 1] + ".txt")
raw2 = np.loadtxt(GloUse['test_file'][n - 1] + ".txt")
raw3 = np.loadtxt(GloUse['RUL_file'][n - 1] + ".txt")
# 创建DataFrame
df1 = pd.DataFrame(raw1, columns=['unit', 'cycles', 'operational setting 1', 'operational setting 2',
                                  'operational setting 3'] + ['sensor measurement' + str(i) for i in range(1, 22)])
df2 = pd.DataFrame(raw2, columns=['unit', 'cycles', 'operational setting 1', 'operational setting 2',
                                  'operational setting 3'] + ['sensor measurement' + str(i) for i in range(1, 22)])
# 设定保留的传感器识数
indices = ['sensor measurement' + str(i) for i in GloUse['sensor'][n - 1]]
# 剔除无意义的传感器识数
df1 = df1.loc[:, ['unit', 'cycles'] + indices] #train
df2 = df2.loc[:, ['unit', 'cycles'] + indices] #test

# 计算出训练集数据的最大值、最小值
max = [df1[i].max() for i in indices]
min = [df1[i].min() for i in indices]

label1 = []
label2 = []
# min-max标准化数据 & 提取出剩余寿命（RUL & label）,RUL最大值设置为125.
for i in range(df1.shape[0]):
    k = df1.loc[i, 'unit']
    m = df1.cycles[df1['unit'] == k].max()
    label1.append(m - df1.loc[i, 'cycles'] if (m - df1.loc[i, 'cycles']) < 125.0 else 125.0)
    for j in range(GloUse['SL'][n - 1]):
        df1.iloc[i, j + 2] = (df1.iloc[i, j + 2] - min[j]) / (max[j] - min[j])
for i in range(df2.shape[0]):
    k = df2.loc[i, 'unit']
    m = df2.cycles[df2.unit == k].max()
    label2.append(
        m - df2.loc[i, 'cycles'] + raw3[int(k - 1)] if (m - df2.loc[i, 'cycles'] + raw3[int(k - 1)]) < 125 else 125)
    for j in range(GloUse['SL'][n - 1]):
        df2.iloc[i, j + 2] = (df2.iloc[i, j + 2] - min[j]) / (max[j] - min[j])

df1['label'] = label1
df2['label'] = label2

# 提取HFF特征
HFF1 = pd.DataFrame(columns=['unit', 'cycles'] + [str(i) for i in range(1, 29)] + ['label'])
HFF2 = pd.DataFrame(columns=['unit', 'cycles'] + [str(i) for i in range(1, 29)] + ['label'])

for i in range(int(df1['unit'].max())):
    k = i + 1
    tmp1 = np.array(df1.loc[df1['unit'] == k])
    tmp_HFF1 = np.zeros((tmp1.shape[0] - 1, (2 * GloUse['SL'][n - 1] + 3)))
    tmp_HFF1[:, 0] = tmp1[1:len(tmp1), 0]#unit
    tmp_HFF1[:, 1] = tmp1[1:len(tmp1), 1]#cycle
    tmp_HFF1[:, 30] = tmp1[1:len(tmp1), 16]#label
    tmp_11 =[]
    for t in range((tmp1.shape[0]) - 1):
        tmp_1 = []
        for j in range(GloUse['SL'][n - 1]):
            tmp_1.append(np.mean(tmp1[0:t + 2, j + 2]))
            # x = np.arange(1, len(tmp1[0: t+2, j+2]))
            # y = tmp1[0:t+2, j+2]
            tmp_1.append(np.polyfit(np.arange(1, len(tmp1[0: t+2, j+2]) + 1), tmp1[0:t+2, j+2], 1)[0])
        tmp_11.append(tmp_1)
    tmp_11 = np.array(tmp_11)
    tmp_HFF1[:, 2:30] = tmp_11
    tmp_HFF1 = pd.DataFrame(tmp_HFF1, columns=['unit', 'cycles'] + [str(i) for i in range(1, 29)] + ['label'])
    HFF1 = pd.concat([HFF1, tmp_HFF1])

for i in range(int(df2['unit'].max())):
    k = i + 1
    tmp2 = df2.loc[df2['unit'] == k]
    tmp2 = np.array(tmp2)
    tmp_HFF2 = np.zeros((tmp2.shape[0] - 1, (2 * GloUse['SL'][n - 1] + 3)))
    tmp_HFF2[:, 0] = tmp2[1:len(tmp2), 0]
    tmp_HFF2[:, 1] = tmp2[1:len(tmp2), 1]
    tmp_HFF2[:, 30] = tmp2[1:len(tmp2), 16]
    tmp_22 = []
    for t in range(tmp2.shape[0] - 1):
        tmp_2 = []
        for j in range(GloUse['SL'][n - 1]):
            tmp_2.append(np.mean(tmp2[0:t + 2, j + 2]))
            tmp_2.append(np.polyfit(np.arange(1, len(tmp2[0: t + 2, j + 2]) + 1), tmp2[0:t + 2, j + 2], 1)[0])
        tmp_22.append(tmp_2)
    tmp_22 = np.array(tmp_22)
    tmp_HFF2[:, 2:30] = tmp_22
    tmp_HFF2 = pd.DataFrame(tmp_HFF2, columns=['unit', 'cycles'] + [str(i) for i in range(1, 29)] + ['label'])
    HFF2 = pd.concat([HFF2, tmp_HFF2])

HFF1.to_pickle('train_HFF' + str(n) + ".pickle")
HFF2.to_pickle('test_HFF' + str(n) + ".pickle")
df1 = HFF1
df2 = HFF2


slabel1 = []
slabel2 = []
unit1 = []
unit2 = []
value1 = []
value2 = []
# 构建时间序列数据，窗口大小为30
for i in range(df1.shape[0] - 29):
    HFF_1 = np.array(df1)
    value_1 = []
    if HFF_1[i, 0] == HFF_1[i + 29, 0]:  # unit 'i' = unit 'i+29'
        slabel1.append(HFF_1[i + 29, 30])
        unit1.append(HFF_1[i + 29, 0])
        for j in range(HFF_1.shape[1] - 3):
            if HFF_1[i, 0] == HFF_1[i + 29, 0]:
                value_1.append(HFF_1[i:i + 30, j + 2])
            else:
                break
        value1.append(np.array(value_1).reshape(-1))

for i in range(df2.shape[0] - 29):
    HFF_2 = np.array(df2)
    value_2 = []
    if HFF_2[i, 0] == HFF_2[i + 29, 0]:
        slabel2.append(HFF_2[i + 29, 30])
        unit2.append(HFF_2[i + 29, 0])
        for j in range(HFF_2.shape[1] - 3):
            if HFF_2[i, 0] == HFF_2[i + 29, 0]:
                value_2.append(HFF_2[i:i + 30, j + 2])
            else:
                break
        value2.append(np.array(value_2).reshape(-1))

# 时间序列数据转化为DataFrame格式
value1 = np.array(value1)
value2 = np.array(value2)
slabel1 = np.array(slabel1).reshape(-1,1)
slabel2 = np.array(slabel2).reshape(-1,1)
unit1 = np.array(unit1).reshape(-1,1)
unit2 = np.array(unit2).reshape(-1,1)
df1 = np.concatenate([value1, unit1, slabel1], axis = 1)
df2 = np.concatenate([value2, unit2, slabel2], axis = 1)

df1 = pd.DataFrame(df1, columns=[str(i) for i in range(value1.shape[1])] + ['unit', 'label'])
df2 = pd.DataFrame(df2, columns=[str(i) for i in range(value2.shape[1])] + ['unit', 'label'])



# 保存处理后的数据

df1.to_pickle('HFF_' + GloUse['train_file'][n - 1] + ".pickle")
df2.to_pickle('HFF_' + GloUse['test_file'][n - 1]  + ".pickle")