import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
from math import ceil
from multiprocessing import Pool       
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
from utils import *

data_dir='../dataset/'
feature_dir='../features/'
############## base_fe #################
class Make_fe(object):
    def __init__(self, base_m5, base_pkl, **kwargs):       
        self.base_m5 = base_m5
        self.base_pkl = base_pkl
        self.TARGET = 'sales'
        self.END_TRAIN = 1941
        self.MAIN_INDEX = ['id','d']
        self.index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']
        self.train_df = pd.read_csv(self.base_m5+'sales_train_evaluation.csv')
        self.prices_df = pd.read_csv(self.base_m5+'sell_prices.csv')
        self.calendar_df = pd.read_csv(self.base_m5+'calendar.csv')
        self.grid_df = pd.DataFrame()
        self.SHIFT_DAY = 28
        
    def base_fe(self):
        self.grid_df = pd.melt(self.train_df, 
                  id_vars = self.index_columns, 
                  var_name = 'd', 
                  value_name = self.TARGET)

        print('Train rows:', len(self.train_df), len(self.grid_df))
        add_grid = pd.DataFrame()
        for i in range(1,29):
            temp_df = self.train_df[self.index_columns]
            temp_df = temp_df.drop_duplicates()
            temp_df['d'] = 'd_'+ str(self.END_TRAIN+i)
            temp_df[self.TARGET] = np.nan
            add_grid = pd.concat([add_grid,temp_df])
    
        self.grid_df = pd.concat([self.grid_df,add_grid])
        self.grid_df = self.grid_df.reset_index(drop=True)
    
        del temp_df, add_grid
        
        print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(self.grid_df.memory_usage(index=True).sum())))
        
        for col in self.index_columns:
            self.grid_df[col] = self.grid_df[col].astype('category')
    
        print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(self.grid_df.memory_usage(index=True).sum())))
        print('Release week')
    
        release_df = self.prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
    
        release_df.columns = ['store_id', 'item_id', 'release']
        
        self.grid_df = merge_by_concat(self.grid_df, release_df, ['store_id','item_id'])
        del release_df
    
        self.grid_df = merge_by_concat(self.grid_df, self.calendar_df[['wm_yr_wk','d']], ['d'])
    
        self.grid_df = self.grid_df[self.grid_df['wm_yr_wk']>=self.grid_df['release']]
        self.grid_df = self.grid_df.reset_index(drop=True)

        print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(self.grid_df.memory_usage(index=True).sum())))

        self.grid_df['release'] = self.grid_df['release'] - self.grid_df['release'].min()
        self.grid_df['release'] = self.grid_df['release'].astype(np.int16)

        print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(self.grid_df.memory_usage(index=True).sum())))

        print('Save Part 1')

        self.grid_df.to_pickle(self.base_pkl+'grid_part_1.pkl')

        print('Size:', self.grid_df.shape)
        
    def prices_fe(self):
        print('Prices')
        self.grid_df = pd.read_pickle(self.base_pkl+'grid_part_1.pkl')
        self.prices_df['price_max'] = self.prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
        self.prices_df['price_min'] = self.prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')
        self.prices_df['price_std'] = self.prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')
        self.prices_df['price_mean'] = self.prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')
        self.prices_df['price_norm'] = self.prices_df['sell_price']/self.prices_df['price_max']
        self.prices_df['price_nunique'] = self.prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
        self.prices_df['item_nunique'] = self.prices_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')
    
        calendar_prices = self.calendar_df[['wm_yr_wk','month','year']]
        calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
        self.prices_df = self.prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')
        del calendar_prices

        self.prices_df['price_momentum'] = self.prices_df['sell_price']/self.prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
        self.prices_df['price_momentum_m'] = self.prices_df['sell_price']/self.prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
        self.prices_df['price_momentum_y'] = self.prices_df['sell_price']/self.prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')

        del self.prices_df['month'], self.prices_df['year']

        print('Merge prices and save part 2')
        print(self.grid_df.shape)
        original_columns = list(self.grid_df)
        self.grid_df = self.grid_df.merge(self.prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
        keep_columns = [col for col in list(self.grid_df) if col not in original_columns]
        self.grid_df = self.grid_df[self.MAIN_INDEX+keep_columns]
        self.grid_df = reduce_mem_usage(self.grid_df)

        self.grid_df.to_pickle(self.base_pkl+'grid_part_2.pkl')
        print('Size:', self.grid_df.shape)

        self.grid_df = pd.read_pickle(self.base_pkl+'grid_part_1.pkl')


        self.grid_df = self.grid_df[self.MAIN_INDEX]
        icols = ['date',
                 'd',
                 'event_name_1',
                 'event_type_1',
                 'event_name_2',
                 'event_type_2',
                 'snap_CA',
                 'snap_TX',
                 'snap_WI']

        self.grid_df = self.grid_df.merge(self.calendar_df[icols], on=['d'], how='left')

        # Minify data
        # 'snap_' columns we can convert to bool or int8
        icols = ['event_name_1',
                 'event_type_1',
                 'event_name_2',
                 'event_type_2',
                 'snap_CA',
                 'snap_TX',
                 'snap_WI',
                 ]
        for col in icols:
            self.grid_df[col] = self.grid_df[col].astype('category')


        # Convert to DateTime
        self.grid_df['date'] = pd.to_datetime(self.grid_df['date'])

        # Make some features from date
        self.grid_df['tm_d'] = self.grid_df['date'].dt.day.astype(np.int8)
        self.grid_df['tm_w'] = self.grid_df['date'].dt.week.astype(np.int8)
        self.grid_df['tm_m'] = self.grid_df['date'].dt.month.astype(np.int8)
        self.grid_df['tm_y'] = self.grid_df['date'].dt.year
        self.grid_df['tm_y'] = (self.grid_df['tm_y'] - self.grid_df['tm_y'].min()).astype(np.int8)
        self.grid_df['tm_wm'] = self.grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8)

        self.grid_df['tm_dw'] = self.grid_df['date'].dt.dayofweek.astype(np.int8)
        self.grid_df['tm_w_end'] = (self.grid_df['tm_dw']>=5).astype(np.int8)


        del self.grid_df['date']
        print('Save part 3')
        self.grid_df.to_pickle(self.base_pkl+'grid_part_3_holidays.pkl')
        print('Size:', self.grid_df.shape)

        self.grid_df = pd.read_pickle(self.base_pkl+'grid_part_1.pkl')
        self.grid_df['d'] = self.grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)
        del self.grid_df['wm_yr_wk']
        self.grid_df.to_pickle(self.base_pkl+'grid_part_1.pkl')
        
    def lag_fe(self):
    
        self.grid_df = pd.read_pickle(self.base_pkl+'grid_part_1.pkl')

        self.grid_df = self.grid_df[['id','d','sales']]
        
        start_time = time.time()
        print('Create lags')

        LAG_DAYS = [col for col in range(self.SHIFT_DAY, self.SHIFT_DAY+15)]
        self.grid_df = self.grid_df.assign(**{
                '{}_lag_{}'.format(col, l): self.grid_df.groupby(['id'])[col].transform(lambda x: x.shift(l))
                for l in LAG_DAYS
                for col in [self.TARGET]
            })

        for col in list(self.grid_df):
            if 'lag' in col:
                self.grid_df[col] = self.grid_df[col].astype(np.float16)

        print('Create rolling aggs')
        for i in [7,14,30,60,180]:
            print('Rolling period:', i)
            self.grid_df['rolling_mean_'+str(i)] = self.grid_df.groupby(['id'])[self.TARGET].transform(lambda x: x.shift(self.SHIFT_DAY).rolling(i).mean()).astype(np.float16)
            self.grid_df['rolling_std_'+str(i)]  = self.grid_df.groupby(['id'])[self.TARGET].transform(lambda x: x.shift(self.SHIFT_DAY).rolling(i).std()).astype(np.float16)

        for d_shift in [1,7,14]: 
            print('Shifting period:', d_shift)
            for d_window in [7,14,30,60]:
                col_name = 'rolling_mean_tmp_'+str(d_shift)+'_'+str(d_window)
                self.grid_df[col_name] = self.grid_df.groupby(['id'])[self.TARGET].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)

        print('%0.2f min: Lags' % ((time.time() - start_time) / 60))
        print('Save lags and rollings')
        self.grid_df.to_pickle(self.base_pkl+'lags_df_'+str(self.SHIFT_DAY)+'_std'+'.pkl')
        
    def encoding_fe(self):
        self.grid_df = pd.read_pickle(self.base_pkl+'grid_part_1.pkl')
        self.grid_df[self.TARGET][self.grid_df['d']>(1941)] = np.nan

        base_cols = list(self.grid_df)

        icols =  [
                    ['state_id'],
                    ['store_id'],
                    ['cat_id'],
                    ['dept_id'],
                    ['state_id', 'cat_id'],
                    ['state_id', 'dept_id'],
                    ['store_id', 'cat_id'],
                    ['store_id', 'dept_id'],
                    ['item_id'],
                    ['item_id', 'state_id'],
                    ['item_id', 'store_id']
                    ]

        for col in icols:
            print('Encoding', col)
            col_name = '_'+'_'.join(col)+'_'
            self.grid_df['enc'+col_name+'mean'] = self.grid_df.groupby(col)[self.TARGET].transform('mean').astype(np.float16)
            self.grid_df['enc'+col_name+'std'] = self.grid_df.groupby(col)[self.TARGET].transform('std').astype(np.float16)

        keep_cols = [col for col in list(self.grid_df) if col not in base_cols]
        self.grid_df = self.grid_df[['id','d']+keep_cols]

        print('Save Mean/Std encoding')
        self.grid_df.to_pickle(self.base_pkl+'mean_encoding_df_28.pkl')
        
    def holidays_CA1_TX2_TX3_fe(self):
    
        calendar_win = self.calendar_df[['event_name_2','event_name_1','d']]
        calendar_win.rename(columns={'event_name_1':'name1','event_name_2':'name2'},inplace=True)
        calendar_win['event_name_1']=np.nan

        ### SuperBowl
        SuperBowl_d = [9, 373, 737, 1101, 1465, 1836]
        for i in SuperBowl_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'SuperBowl1'
            d_2['event_name_1'] = 'SuperBowl1'
            d_3['event_name_1'] = 'SuperBowl2'

            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_2
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_3


        ## ValentinesDay
        ValentinesDay_d = [17, 382, 748, 1113, 1478, 1843]
        for i in ValentinesDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'ValentinesDay'
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1


        ### LentStart
        LentStart_d = [40, 390, 747, 1132, 1482, 1839]
        for i in LentStart_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']


            d_1['event_name_1'] = 'LentStart1'
            d_2['event_name_1'] = 'LentStart2'


            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## LentWeek2
        LentWeek_d = [47, 397, 754, 1139, 1489, 1846]
        for i in LentWeek_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'LentWeek1'
            d_2['event_name_1'] = 'LentWeek2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## Easter
        Ester_d = [86, 436, 793, 1178, 1528, 1885]
        for i in Ester_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'Ester1'
            d_2['event_name_1'] = 'Ester1'

            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_2

        #Mother's day(Cultural) 
        ### 
        Mother_d = [100, 471, 835, 1199, 1563, 1927]
        for i in Mother_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'Mother_day1'
            d_2['event_name_1'] = 'Mother_day1'

            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_2


        ## MemorialDay(National)
        MemorialDay_d = [122, 486, 850, 1214, 1578, 1949]
        for i in MemorialDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']


            d_1['event_name_1'] = 'MemorialDay1'
            d_2['event_name_1'] = 'MemorialDay2'


            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2


        ## Father's day
        Father_d = [142, 506, 870, 1234, 1605, 1969]
        for i in Father_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'Father1'
            d_2['event_name_1'] = 'Father1'
            d_3['event_name_1'] = 'Father2'

            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_2
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_3

        ## IndependenceDay
        IndependenceDay_d = [157, 523, 888, 1253, 1618]
        for i in IndependenceDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'IndependenceDay1'
            d_2['event_name_1'] = 'IndependenceDay2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## LaborDay
        LaborDay_d = [220, 584, 948, 1312, 1683]
        for i in LaborDay_d:
            #d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_1 = calendar_win[calendar_win['d']==f'd_{i+1}']

            #d_2['event_name_1'] = 'LaborDay1'
            d_1['event_name_1'] = 'LaborDay1'

            #calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_1


        ## ColumbusDay(National)
        ColumbusDay_d = [255, 619, 990, 1354, 1718]
        for i in ColumbusDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'ColumbusDay1'
            d_2['event_name_1'] = 'ColumbusDay2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## Halloween
        Halloween_d = [276, 642, 1007, 1372, 1737]
        for i in Halloween_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'Halloween1'
            d_2['event_name_1'] = 'Halloween2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2


        # EidAlAdha 
        EidAlAdha_d = [283, 637, 991, 1345, 1700]
        for i in EidAlAdha_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'EidAlAdha1'
            d_2['event_name_1'] = 'EidAlAdha2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## VeteransDay 
        VeteransDay_d = [287, 653, 1018, 1383, 1748]
        for i in VeteransDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'VeteransDay1'
            d_2['event_name_1'] = 'VeteransDay1'
            d_3['event_name_1'] = 'VeteransDay2'

            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_2
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_3


        # Thanksgiving 
        Thanksgiving_d = [300, 664, 1035, 1399, 1763]
        for i in Thanksgiving_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-3}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_4 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'Thanksgiving1'
            d_2['event_name_1'] = 'Thanksgiving1'
            d_3['event_name_1'] = 'Thanksgiving2'
            d_4['event_name_1'] = 'Thanksgiving3'

            calendar_win[calendar_win['d']==f'd_{i-3}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_2
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_3
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_4


        # Christmas
        Christmas_d = [331, 697, 1062, 1427, 1792]
        for i in Christmas_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-6}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-5}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i-4}']
            d_4 = calendar_win[calendar_win['d']==f'd_{i-3}']
            d_5 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_6 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_7 = calendar_win[calendar_win['d']==f'd_{i+1}']
            d_8 = calendar_win[calendar_win['d']==f'd_{i+2}']
            d_9 = calendar_win[calendar_win['d']==f'd_{i+3}']
            d_10 = calendar_win[calendar_win['d']==f'd_{i+4}']
            d_11 = calendar_win[calendar_win['d']==f'd_{i+5}']
            d_12 = calendar_win[calendar_win['d']==f'd_{i+6}']

            d_1['event_name_1'] = 'Christmas1'
            d_2['event_name_1'] = 'Christmas1'
            d_3['event_name_1'] = 'Christmas1'
            d_4['event_name_1'] = 'Christmas1'
            d_5['event_name_1'] = 'Christmas1'
            d_6['event_name_1'] = 'Christmas1'

            d_7['event_name_1'] = 'Christmas2'
            d_8['event_name_1'] = 'Christmas2'
            d_9['event_name_1'] = 'Christmas2'
            d_10['event_name_1'] = 'Christmas2'
            d_11['event_name_1'] = 'Christmas2'
            d_12['event_name_1'] = 'Christmas2'

            calendar_win[calendar_win['d']==f'd_{i-6}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-5}'] = d_2
            calendar_win[calendar_win['d']==f'd_{i-4}'] = d_3
            calendar_win[calendar_win['d']==f'd_{i-3}'] = d_4

            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_5
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_6
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_7
            calendar_win[calendar_win['d']==f'd_{i+2}'] = d_8

            calendar_win[calendar_win['d']==f'd_{i+3}'] = d_9
            calendar_win[calendar_win['d']==f'd_{i+4}'] = d_10
            calendar_win[calendar_win['d']==f'd_{i+5}'] = d_11
            calendar_win[calendar_win['d']==f'd_{i+6}'] = d_12

        ## NewYear
        NewYear_d = [338, 704, 1069, 1434, 1799]
        for i in NewYear_d:
            d_4 = calendar_win[calendar_win['d']==f'd_{i+1}']
            d_5 = calendar_win[calendar_win['d']==f'd_{i+2}']
            d_6 = calendar_win[calendar_win['d']==f'd_{i+3}']
            d_7 = calendar_win[calendar_win['d']==f'd_{i+4}']
            d_8 = calendar_win[calendar_win['d']==f'd_{i+5}']
            d_9 = calendar_win[calendar_win['d']==f'd_{i+6}']
            d_4['event_name_1'] = 'NewYear2'
            d_5['event_name_1'] = 'NewYear2'
            d_6['event_name_1'] = 'NewYear2'
            d_7['event_name_1'] = 'NewYear2'
            d_8['event_name_1'] = 'NewYear2'
            d_9['event_name_1'] = 'NewYear2'
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_4
            calendar_win[calendar_win['d']==f'd_{i+2}'] = d_5
            calendar_win[calendar_win['d']==f'd_{i+3}'] = d_6
            calendar_win[calendar_win['d']==f'd_{i+4}'] = d_7
            calendar_win[calendar_win['d']==f'd_{i+5}'] = d_8
            calendar_win[calendar_win['d']==f'd_{i+6}'] = d_9

        calendar_win.rename(columns={'event_name_1':'event_name_1_win','event_name_2':'event_name_2_win'},inplace=True)
        calendar_win.drop(columns=['name2','name1'],inplace=True)

        ## save CA1_TX2_TX3_holidays to pkl
        calendar_win.to_pickle(self.base_pkl+'CA1_TX2_TX3_holidays.pkl')
        
        
    def holidays_CA2_fe(self):
        calendar_win = self.calendar_df[['event_name_2','event_name_1','d']]
        calendar_win.rename(columns={'event_name_1':'name1','event_name_2':'name2'},inplace=True)
        calendar_win['event_name_1']=np.nan
        
        ### SuperBowl
        SuperBowl_d = [9, 373, 737, 1101, 1465, 1836]

        for i in SuperBowl_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_1['event_name_1'] = 'SuperBowl1'
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1

        ## ValentinesDay
        ValentinesDay_d = [17, 382, 748, 1113, 1478, 1843]
        for i in ValentinesDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']
            d_1['event_name_1'] = 'ValentinesDay1'
            d_2['event_name_1'] = 'ValentinesDay2'
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## LentWeek
        LentWeek_d = [47, 397, 754, 1139, 1489, 1846]
        for i in LentWeek_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_1['event_name_1'] = 'LentWeek1'
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
         
        Mother_d = [100, 471, 835, 1199, 1563, 1927]
        for i in Mother_d:

            d_1 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_1['event_name_1'] = 'Mother_day1'
            d_2['event_name_1'] = 'Mother_day2'
            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_2

        ## MemorialDay(National)
        MemorialDay_d = [122, 486, 850, 1214, 1578, 1949]
        for i in MemorialDay_d:

            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']
            d_1['event_name_1'] = 'MemorialDay1'
            d_2['event_name_1'] = 'MemorialDay2'
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## Father
        Father_d = [142, 506, 870, 1234, 1605, 1969]
        for i in Father_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i+1}']
            d_1['event_name_1'] = 'Father1'
            d_2['event_name_1'] = 'Father1'
            d_3['event_name_1'] = 'Father2'
            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_2
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_3

        ## IndependenceDay
        IndependenceDay_d = [157, 523, 888, 1253, 1618]
        for i in IndependenceDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']
            d_1['event_name_1'] = 'IndependenceDay1'
            d_2['event_name_1'] = 'IndependenceDay2'
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## LaborDay
        LaborDay_d = [220, 584, 948, 1312, 1683]
        for i in LaborDay_d:

            d_1 = calendar_win[calendar_win['d']==f'd_{i+1}']
            d_1['event_name_1'] = 'LaborDay2'
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_1

        ## ColumbusDay(National)
        ColumbusDay_d = [255, 619, 990, 1354, 1718]
        for i in ColumbusDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'ColumbusDay1'
            d_2['event_name_1'] = 'ColumbusDay2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## Halloween
        Halloween_d = [276, 642, 1007, 1372, 1737]
        for i in Halloween_d:

            d_1 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'Halloween1'

            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_1

        ## VeteransDay 
        VeteransDay_d = [287, 653, 1018, 1383, 1748]
        for i in VeteransDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'VeteransDay1'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1

        # Thanksgiving 
        Thanksgiving_d = [300, 664, 1035, 1399, 1763]
        for i in Thanksgiving_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'Thanksgiving1'
            d_2['event_name_1'] = 'Thanksgiving2'

            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_2

        # Christmas
        Christmas_d = [331, 697, 1062, 1427, 1792]
        for i in Christmas_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-3}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_4 = calendar_win[calendar_win['d']==f'd_{i+1}']
            d_5 = calendar_win[calendar_win['d']==f'd_{i+2}']
            d_6 = calendar_win[calendar_win['d']==f'd_{i+3}']



            d_1['event_name_1'] = 'Christmas1'
            d_2['event_name_1'] = 'Christmas1'
            d_3['event_name_1'] = 'Christmas2'
            d_4['event_name_1'] = 'Christmas3'
            d_5['event_name_1'] = 'Christmas4'
            d_6['event_name_1'] = 'Christmas4'
            calendar_win[calendar_win['d']==f'd_{i-3}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_2
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_3
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_4
            calendar_win[calendar_win['d']==f'd_{i+2}'] = d_5
            calendar_win[calendar_win['d']==f'd_{i+3}'] = d_6

        ## NewYear
        NewYear_d = [338, 704, 1069, 1434, 1799]
        for i in NewYear_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-3}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_4 = calendar_win[calendar_win['d']==f'd_{i+1}']
            d_5 = calendar_win[calendar_win['d']==f'd_{i+2}']
            d_6 = calendar_win[calendar_win['d']==f'd_{i+3}']
            d_1['event_name_1'] = 'NewYear1'
            d_2['event_name_1'] = 'NewYear1'
            d_3['event_name_1'] = 'NewYear2'
            d_4['event_name_1'] = 'NewYear3'
            d_5['event_name_1'] = 'NewYear4'
            d_6['event_name_1'] = 'NewYear4'
            calendar_win[calendar_win['d']==f'd_{i-3}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_2
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_3
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_4
            calendar_win[calendar_win['d']==f'd_{i+2}'] = d_5
            calendar_win[calendar_win['d']==f'd_{i+3}'] = d_6
          
        ## Easter
        Ester_d = [86, 436, 793, 1178, 1528, 1885]
        for i in Ester_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'Ester1'
            d_2['event_name_1'] = 'Ester2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        # MartinLutherKingDay:
        MartinLutherKingDay_d = [353, 724, 1088, 1452, 1816]
        for i in MartinLutherKingDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'MartinLutherKingDay'

            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_1

        PresidentsDay_d = [24, 388, 752, 1116, 1480, 1844]
        for i in PresidentsDay_d:
           # d_1 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-1}']
           # d_0 = calendar_win[calendar_win['d'] == f'd_{i}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i+1}']
            #d_1['event_name_1'] = 'PresidentsDay_1_1'
            d_2['event_name_1'] = 'PresidentsDay_1_2'
            #d_0['event_name_1'] = 'PresidentsDay'
            d_3['event_name_1'] = 'PresidentsDay_2'
            #calendar_win[calendar_win['d']==f'd_{i-2}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_2
           # calendar_win[calendar_win['d'] == f'd_{i}'] = d_0
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_3

        calendar_win.rename(columns={'event_name_1':'event_name_1_win','event_name_2':'event_name_2_win'},inplace=True)
        calendar_win.drop(columns=['name2','name1'],inplace=True)
        calendar_win.to_pickle(self.base_pkl+'CA2_holidays.pkl')
        
    def holidays_WI3_fe(self):
        calendar_win = self.calendar_df[['event_name_2','event_name_1','d']]
        calendar_win.rename(columns={'event_name_1':'name1','event_name_2':'name2'},inplace=True)
        calendar_win['event_name_1']=np.nan

        ### SuperBowl
        SuperBowl_d = [9, 373, 737, 1101, 1465, 1836]
        for i in SuperBowl_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'SuperBowl1'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1

        ## ValentinesDay
        ValentinesDay_d = [17, 382, 748, 1113, 1478, 1843]
        for i in ValentinesDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'ValentinesDay1'
            d_2['event_name_1'] = 'ValentinesDay2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## LentWeek
        LentWeek_d = [47, 397, 754, 1139, 1489, 1846]
        for i in LentWeek_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'LentWeek1'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
         
        Mother_d = [100, 471, 835, 1199, 1563, 1927]
        for i in Mother_d:

            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'Mother_day1'
            d_2['event_name_1'] = 'Mother_day2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## MemorialDay(National)
        MemorialDay_d = [122, 486, 850, 1214, 1578, 1949]
        for i in MemorialDay_d:

            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']


            d_1['event_name_1'] = 'MemorialDay1'
            d_2['event_name_1'] = 'MemorialDay2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## Father
        Father_d = [142, 506, 870, 1234, 1605, 1969]
        for i in Father_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'Father1'
            d_2['event_name_1'] = 'Father1'
            d_3['event_name_1'] = 'Father2'

            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_2
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_3

        ## IndependenceDay
        IndependenceDay_d = [157, 523, 888, 1253, 1618]
        for i in IndependenceDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'IndependenceDay1'
            d_2['event_name_1'] = 'IndependenceDay2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## LaborDay
        LaborDay_d = [220, 584, 948, 1312, 1683]
        for i in LaborDay_d:

            d_1 = calendar_win[calendar_win['d']==f'd_{i+1}']


            d_1['event_name_1'] = 'LaborDay2'

            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_1


        ColumbusDay_d = [255, 619, 990, 1354, 1718]
        for i in ColumbusDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'ColumbusDay1'
            d_2['event_name_1'] = 'ColumbusDay2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        ## Halloween，
        Halloween_d = [276, 642, 1007, 1372, 1737]
        for i in Halloween_d:

            d_1 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'Halloween1'

            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_1

        ## VeteransDay 
        VeteransDay_d = [287, 653, 1018, 1383, 1748]
        for i in VeteransDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'VeteransDay1'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1

        # Thanksgiving 
        Thanksgiving_d = [300, 664, 1035, 1399, 1763]
        for i in Thanksgiving_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'Thanksgiving1'
            d_2['event_name_1'] = 'Thanksgiving2'

            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_2

        # Christmas
        Christmas_d = [331, 697, 1062, 1427, 1792]
        for i in Christmas_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-3}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_4 = calendar_win[calendar_win['d']==f'd_{i+1}']
            d_5 = calendar_win[calendar_win['d']==f'd_{i+2}']
            d_6 = calendar_win[calendar_win['d']==f'd_{i+3}']



            d_1['event_name_1'] = 'Christmas1'
            d_2['event_name_1'] = 'Christmas1'
            d_3['event_name_1'] = 'Christmas2'
            d_4['event_name_1'] = 'Christmas3'
            d_5['event_name_1'] = 'Christmas4'
            d_6['event_name_1'] = 'Christmas4'


            calendar_win[calendar_win['d']==f'd_{i-3}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_2
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_3
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_4

            calendar_win[calendar_win['d']==f'd_{i+2}'] = d_5
            calendar_win[calendar_win['d']==f'd_{i+3}'] = d_6

        ## NewYear
        NewYear_d = [338, 704, 1069, 1434, 1799]
        for i in NewYear_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-3}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i-2}']
            d_3 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_4 = calendar_win[calendar_win['d']==f'd_{i+1}']
            d_5 = calendar_win[calendar_win['d']==f'd_{i+2}']
            d_6 = calendar_win[calendar_win['d']==f'd_{i+3}']



            d_1['event_name_1'] = 'NewYear1'
            d_2['event_name_1'] = 'NewYear1'
            d_3['event_name_1'] = 'NewYear2'
            d_4['event_name_1'] = 'NewYear3'
            d_5['event_name_1'] = 'NewYear4'
            d_6['event_name_1'] = 'NewYear4'

            calendar_win[calendar_win['d']==f'd_{i-3}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i-2}'] = d_2
            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_3
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_4

            calendar_win[calendar_win['d']==f'd_{i+2}'] = d_5
            calendar_win[calendar_win['d']==f'd_{i+3}'] = d_6
          
        ## Easter
        Ester_d = [86, 436, 793, 1178, 1528, 1885]
        for i in Ester_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']
            d_2 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'Ester1'
            d_2['event_name_1'] = 'Ester2'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_2

        # MartinLutherKingDay:
        MartinLutherKingDay_d = [353, 724, 1088, 1452, 1816]
        for i in MartinLutherKingDay_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'MartinLutherKingDay'

            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_1


        # Cinco De Mayo:
        Mayo_d = [97, 463, 1193, 1558, 1924]
        for i in Mayo_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'Mayo'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1

        # OrthodoxChristmas:
        OrthodoxChristmas_d = [344, 710, 1075, 1440, 1805]
        for i in OrthodoxChristmas_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i+1}']

            d_1['event_name_1'] = 'OrthodoxChristmas'

            calendar_win[calendar_win['d']==f'd_{i+1}'] = d_1

        # OrthodoxEaster
        OrthodoxEaster_d = [443, 828, 1535, 1920]
        for i in OrthodoxEaster_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'OrthodoxEaster'

            calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1

        # Purim
        Purim_d = [51, 405, 758, 1143, 1497, 1882]
        for i in Purim_d:
            d_1 = calendar_win[calendar_win['d']==f'd_{i-1}']

            d_1['event_name_1'] = 'Purim'

        calendar_win[calendar_win['d']==f'd_{i-1}'] = d_1
        calendar_win.rename(columns={'event_name_1':'event_name_1_win','event_name_2':'event_name_2_win'},inplace=True)
        calendar_win.drop(columns=['name2','name1'],inplace=True)
        calendar_win.to_pickle(self.base_pkl+'WI_3_holidays.pkl')
        
        
        
making_features = Make_fe(base_m5=data_dir, base_pkl=feature_dir) 
making_features.base_fe()
making_features.prices_fe()
making_features.lag_fe()
making_features.encoding_fe()

## We found that different states have different holiday effects,
## so we have made holiday analysis on stores in different states, 
## and the effects before and after holidays of each store are different
making_features.holidays_CA1_TX2_TX3_fe()
making_features.holidays_CA2_fe()
making_features.holidays_WI3_fe()
    
    