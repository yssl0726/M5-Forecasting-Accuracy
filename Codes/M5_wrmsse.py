# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:29:28 2020

@author: Lebesgue
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import gc
import os 
os.chdir('G:/kaggle/M5/juyterD')

# 1914 1886 
# 1913 1885

# 转换数据类型，减少内存占用空间
def reduce_mem_usage(df, verbose=True):
# =============================================================================
#     df = calendar
# =============================================================================
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# 加载数据
data_pass = './m5-forecasting-accuracy/'

# sale数据
sales = pd.read_csv(data_pass+'sales_train_validation.csv') 

# 日期数据
calendar = pd.read_csv(data_pass+'calendar.csv')
calendar = reduce_mem_usage(calendar)

# 价格数据
sell_prices = pd.read_csv(data_pass+'sell_prices.csv')
sell_prices = reduce_mem_usage(sell_prices)

# 计算价格
# 按照定义，只需要计算最近的 28 天售卖量（售卖数*价格），通过这个可以得到 weight
# 可以不是 1886
cols = ["d_{}".format(i) for i in range(1886-28, 1886)]
data = sales[["id", 'store_id', 'item_id'] + cols]

# 从横表改为纵表
data = data.melt(id_vars=["id", 'store_id', 'item_id'], 
                 var_name="d", value_name="sale")

# 和日期数据做关联
data = pd.merge(data, calendar, how = 'left', 
                left_on = ['d'], right_on = ['d'])

data = data[["id", 'store_id', 'item_id', "sale", "d", "wm_yr_wk"]]

# 和价格数据关联
data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
data.drop(columns = ['wm_yr_wk'], inplace=True)

# 计算售卖量总价（售卖数量*商品单价）
data['sale_usd'] = data['sale'] * data['sell_price']

# 得到聚合矩阵
# 30490 -> 42840
# 需要聚合的维度明细计算出来 
# 根据图片【聚合】对应下面就是
# 2 3 4 5 6 7 8 9 10 11 12  
dummies_list = [sales.state_id, sales.store_id, 
                sales.cat_id, sales.dept_id, 
                sales.state_id + sales.cat_id, sales.state_id + sales.dept_id,
                sales.store_id + sales.cat_id, sales.store_id + sales.dept_id, 
                sales.item_id, sales.state_id + sales.item_id, sales.id]

# =============================================================================
# dummies_list = [sales.state_id + sales.item_id]
# =============================================================================


# 全部聚合为一个， 最高 level 
# 即 1 
dummies_df_list =[pd.DataFrame(np.ones(sales.shape[0]).astype(np.int8), 
                               index=sales.index, columns=['all']).T]

# 挨个计算其他 level 等级聚合
for i, cats in enumerate(dummies_list):
    print(i)
    dummies_df_list +=[pd.get_dummies(cats, drop_first=False, dtype=np.int8).T]
    
# 得到聚合矩阵
roll_mat_df = pd.concat(dummies_df_list, keys=list(range(12)), 
                        names=['level','id'])#.astype(np.int8, copy=False)

# =============================================================================
#                                        0      1      2      ...  30487  30488  30489
# level id                                                    ...                     
# 0     all                                  1      1      1  ...      1      1      1
# 1     CA                                   1      1      1  ...      0      0      0
#       TX                                   0      0      0  ...      0      0      0
#       WI                                   0      0      0  ...      1      1      1
# 2     CA_1                                 1      1      1  ...      0      0      0
#                                      ...    ...    ...  ...    ...    ...    ...
# 11    HOUSEHOLD_2_516_TX_2_validation      0      0      0  ...      0      0      0
#       HOUSEHOLD_2_516_TX_3_validation      0      0      0  ...      0      0      0
#       HOUSEHOLD_2_516_WI_1_validation      0      0      0  ...      0      0      0
#       HOUSEHOLD_2_516_WI_2_validation      0      0      0  ...      0      0      0
#       HOUSEHOLD_2_516_WI_3_validation      0      0      0  ...      0      0      0
# =============================================================================
      
# 保存聚合矩阵 
# 将矩阵的索引保存  
roll_index = roll_mat_df.index 

# =============================================================================
# MultiIndex([( 0,                             'all'),
#             ( 1,                              'CA'),
#             ( 1,                              'TX'),
#             ( 1,                              'WI'),
#             ( 2,                            'CA_1'),
#             ( 2,                            'CA_2'),
#             ( 2,                            'CA_3'),
#             ( 2,                            'CA_4'),
#             ( 2,                            'TX_1'),
#             ( 2,                            'TX_2'),
#             ...
#             (11, 'HOUSEHOLD_2_516_CA_1_validation'),
#             (11, 'HOUSEHOLD_2_516_CA_2_validation'),
#             (11, 'HOUSEHOLD_2_516_CA_3_validation'),
#             (11, 'HOUSEHOLD_2_516_CA_4_validation'),
#             (11, 'HOUSEHOLD_2_516_TX_1_validation'),
#             (11, 'HOUSEHOLD_2_516_TX_2_validation'),
#             (11, 'HOUSEHOLD_2_516_TX_3_validation'),
#             (11, 'HOUSEHOLD_2_516_WI_1_validation'),
#             (11, 'HOUSEHOLD_2_516_WI_2_validation'),
#             (11, 'HOUSEHOLD_2_516_WI_3_validation')],
#            names=['level', 'id'], length=42840)
# =============================================================================

# 将矩阵的值也保存 
roll_mat_csr = csr_matrix(roll_mat_df.values)
roll_mat_df.to_pickle('roll_mat_df.pkl')

# 销毁对象 
# 即销毁连接前的各个聚合组合的数列 与 连接后的矩阵 
del dummies_df_list, roll_mat_df

# 释放内存
gc.collect()


# 按照定义，计算每条时间序列 RMSSE 的权重:
def get_s(drop_days=0):
    
    """
    drop_days: int, equals 0 by default, so S is calculated on all data.
               If equals 28, last 28 days won't be used in calculating S.
    """
    
    # 要计算的时间序列长度 
    # 如果drop_days为0 则有d_1 ~ d_1885 
    d_name = ['d_' + str(i+1) for i in range(1885-drop_days)]
    # 得到聚合结果 
    # 矩阵乘法 (42840, 30490) * (30490, 1885) = (42840, 1885) 
    # roll_mat_csr : 行为每个聚合层次 列为30490条销售索引 值代表取某列时属于某行的就标记1 否则为0 
    # 列即为唯一商品id的索引 所以可以这么理解 如果aij=1 则意味着索引为j所对应的商品在i聚合层次里面有销售记录 
    # sales[d_name] :  行为30489条销售索引 列为d_name那么多天的天字段 值为销售量 
    # 结果即为每个聚合层次（行） d_name那么多天（行）的每天 的销售量 如果i为行 j为列 那么aij就是
    # 第i个聚合层次下 j这天的销售量  
    # 也可以这么写 roll_mat_csr @ sales[d_name].values
    sales_train_val = roll_mat_csr * sales[d_name].values

    # 按照定义，前面连续为 0 的不参与计算 
    # 即找出每个聚类层次 前面连续取值为0（销售量为0的天）不参与计算 返回的是第一个不为0 的列索引j（第j-1天） 
    start_no = np.argmax(sales_train_val>0, axis=1)
    
    # 这些连续为 0 的设置为 nan 
    # flag 的形状与 sales_train_val 的形状一致 
    # np.diag(1/(start_no+1)) 对角矩阵 (42840, 42840) 加1的原因 一是为了分母非0 二是为了等下矩阵相乘 
    # 能起到用是否小于1来判断该聚类层次的第几天开始是第一个非0的 
    # 例如 第一行的聚类层次 在列索引j等于3的时候（即第4天）才开始不为0 则对角矩阵的第一个元素是 1/4 
    # 乘过去后面一部分的第一行时 因其值为1 2 3 4 5...1885 那么除以4的话有 1/4 2/4 3/4 4/4 5/4 
    # 可见只要判断是否小于1 就能把第一个非0值对应的天找出来 这里就是第4天了 因为4/4不小于1 
    # 后面一部分的为 (42840, 1885) 生成的过程 里面是A:(1885, ) B:(42840, 1) A按照列方向复制了42840行 
# =============================================================================
#     flag = np.dot(np.diag(1/(start_no+1)), np.tile(np.arange(1,1886-drop_days),(roll_mat_csr.shape[0],1)))<1 
# =============================================================================
    # 上述对角阵内存开销较大 可以利用数组的广播功能 将其写成如下等价的式子 
# =============================================================================
#     flag = (1/(start_no+1)).reshape(len(start_no), -1) * np.tile(np.arange(1,1886-drop_days),(roll_mat_csr.shape[0],1)) < 1 
# =============================================================================
    # 事实上还可以把右边的部分同样利用数组的广播功能 免去tile 即等价于下面的式子 
    flag = (1/(start_no+1)).reshape(len(start_no), -1) * np.arange(1,1886-drop_days) < 1
    # 上式也可以写成规整一点的形式 
    # (1/(start_no+1)).reshape(len(start_no), -1) * np.arange(1,1886-drop_days).reshape(-1, len(np.arange(1,1886-drop_days)))
    # 返回的flag是如果上述矩阵乘积的值小于1（即连续为0的那些天） 则返回True 否则为False  
    # np.where 作用是根据flag的布尔值 如果为True 则设置为nan 否则取sales_train_val原值 
    sales_train_val = np.where(flag, np.nan, sales_train_val)

    # 根据公式计算每条时间序列 rmsse的权重
    # nansum忽略nan 对非nan进行相加 
    # 就是公式分母那一部分 但不包含h 
    weight1 = np.nansum(np.diff(sales_train_val,axis=1)**2,axis=1)/(1885-start_no-1)
    
    return weight1

# 每个聚合层次的权重 
S = get_s(drop_days=0)

# 根据定义计算 WRMSSE 的权重，这里指 w 
def get_w(sale_usd):
# =============================================================================
#     sale_usd = data[['id','sale_usd']]
# =============================================================================
    """
    """
    # 得到最细维度的每条时间序列的权重 
    # 30490个唯一商品id 形状为(30490,) 聚合求出对应的销售量 
    total_sales_usd = sale_usd.groupby(
        ['id'], sort=False)['sale_usd'].apply(np.sum).values
    
    # 通过聚合矩阵得到不同聚合下的权重
    # (42840,) = (42840, 30490) * (30490,) 
    weight2 = roll_mat_csr * total_sales_usd
    # 因为聚合层次一共有12 所以需要乘以12 
    return 12*(weight2/np.sum(weight2))


W = get_w(data[['id','sale_usd']])

SW = W/np.sqrt(S)

sw_df = pd.DataFrame(np.stack((S, W, SW), axis=-1),index = roll_index,columns=['s','w','sw'])
sw_df.to_pickle('sw_df.pkl')


# 评分函数
# 得到聚合的结果
def rollup(v):
    '''
    '''
    return (v.T*roll_mat_csr.T).T


# 计算 WRMSSE 评估指标
def wrmsse(preds, y_true, score_only=False,s = S, w = W, sw=SW):
    '''
    preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)
    y_true - True values: pd.DataFrame of size (30490 rows, N day columns)
    sequence_length - np.array of size (42840,)
    sales_weight - sales weights based on last 28 days: np.array (42840,)
    '''
    
    if score_only:
        return np.sum(
                np.sqrt(
                    np.mean(
                        np.square(rollup(preds.values-y_true.values))
                            ,axis=1)) * sw *(1/12))
    else: 
        score_matrix = (np.square(rollup(preds.values-y_true.values)) * np.square(w)[:, None]) / s[:, None]
        score = np.sum(np.sqrt(np.mean(score_matrix,axis=1)))*(1/12)
        return score, score_matrix


# 加载前面预先计算好的各个权重
file_pass = './'
sw_df = pd.read_pickle(file_pass+'sw_df.pkl')
S = sw_df.s.values
W = sw_df.w.values
SW = sw_df.sw.values

roll_mat_df = pd.read_pickle(file_pass+'roll_mat_df.pkl')
roll_index = roll_mat_df.index
roll_mat_csr = csr_matrix(roll_mat_df.values)

# =============================================================================
# print(sw_df.loc[(11,slice(None))].sw)
# 
# print(1)
# =============================================================================
