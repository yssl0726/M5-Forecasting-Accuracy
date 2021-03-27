import numpy as np
import pandas as pd
from utils import *
import os, sys, gc, time, warnings, pickle, psutil, random
import lightgbm as lgb
from multiprocessing import Pool 
base_m5='../dataset/'
USE_AUX=True
AUX_MODELS='../models/'
sub_dir = '../sub/'
base_pkl = '../features/'
END_TRAIN = 1941
lgb_CA1_params =  {
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'metric': 'rmse',
    'feature_fraction': 0.9329928507911868,   
    'lambda_l2': 0.021982796763644744,
    'learning_rate': 0.04723157294394453,
    'max_bin': 67,
    'min_data_in_leaf': 11801,
    'n_estimators': 1600,
    'num_leaves': 25631,            
    'subsample': 0.6184421797679618,
    'subsample_freq': 2,
    'tweedie_variance_power': 1.1833186379351004,
    'seed': 42,
    'boost_from_average': False,
    }

lgb_CA3_params =  {
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'metric': 'rmse',
    'feature_fraction': 0.8050328925461323,
    'lambda_l2': 0.05756643694569541,
    'learning_rate': 0.017804685422408588,
    'max_bin': 100,
    'min_data_in_leaf': 6227,
    'n_estimators': 1600,
    'num_leaves': 3479,
    'subsample': 0.5687395598191918,
    'subsample_freq': 2,
    'tweedie_variance_power': 1.3810263355485752,
    'seed': 42,
    'boost_from_average': False,
    }

lgb_others_params = {
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'subsample': 0.5,
    'subsample_freq': 1,
    'learning_rate': 0.03,
    'num_leaves': 2**11-1,
    'min_data_in_leaf': 2**12-1,
    'feature_fraction': 0.5,
    'max_bin': 100,
    'n_estimators': 1,
    'boost_from_average': False,
    'verbose': -1,
    'seed': 42,
            }

VER = 1                          # Our model version
SEED = 42                        # We want all things
seed_everything(SEED)            # to be as deterministic 
# lgb_params['seed'] = SEED        # as possible

#LIMITS and const
START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
P_HORIZON   = 28                 # Prediction horizon
USE_AUX     = True               # Use or not pretrained models

remove_features = ['id','state_id', 'store_id', 'date','wm_yr_wk','release','Holiday','d', 'sales']          
mean_features   = ['enc_cat_id_mean', 'enc_cat_id_std', 'enc_dept_id_mean','enc_dept_id_std', 'enc_item_id_mean', 'enc_item_id_std'] 

BASE     = base_pkl+'grid_part_1.pkl'
PRICE    = base_pkl+'grid_part_2.pkl'
CALENDAR = base_pkl+'grid_part_3_holidays.pkl'
LAGS     = base_pkl+'lags_df_28_std.pkl'
MEAN_ENC = base_pkl+'mean_encoding_df_28.pkl'

#SPLITS for lags creation
SHIFT_DAY  = 28
N_LAGS     = 15
LAGS_SPLIT = [col for col in range(SHIFT_DAY,SHIFT_DAY+N_LAGS)]
ROLS_SPLIT = []
for i in [1,7,14]:
    for j in [7,14,30,60]:
        ROLS_SPLIT.append([i,j])

STORES_IDS = ['CA_1']
END_TRAIN = 1941

save_dir = './models/'
########################### Train Models #################################################################################
for store_id in STORES_IDS:
    print('Train', store_id)
    
    # Get grid for current store
    grid_df, features_columns = get_data_by_store(store_id,LAGS,BASE,PRICE,CALENDAR,MEAN_ENC,mean_features,remove_features)

    ########################## Add holiday features #############################
    if (store_id=='CA_1') or (store_id=='TX_2') or (store_id=='TX_3'):
        calendar_win = pd.read_pickle(base_pkl+'CA1_TX2_TX3_holidays.pkl')
        calendar_win['d'] = calendar_win['d'].apply(lambda x: x.split('_',2)[1])
        calendar_win['d'] = calendar_win['d'].astype(np.int16)

        calendar_win['event_name_1_win'] = calendar_win['event_name_1_win'].astype('category')
        grid_df = grid_df.merge(calendar_win, on='d', how='left')

        features_columns = features_columns+['event_name_1_win']
        if store_id=='CA1':
            lgb_params = lgb_CA1_params
        else:
            lgb_params = lgb_others_params
    elif store_id=='CA_2':
        calendar_win = pd.read_pickle(base_pkl+'CA2_holidays.pkl')
        calendar_win['d'] = calendar_win['d'].apply(lambda x: x.split('_',2)[1])
        calendar_win['d'] = calendar_win['d'].astype(np.int16)

        calendar_win['event_name_1_win'] = calendar_win['event_name_1_win'].astype('category')
        grid_df = grid_df.merge(calendar_win, on='d', how='left')


        features_columns = features_columns+['event_name_1_win']
        lgb_params = lgb_others_params

    elif store_id=='TX_1':
        calendar_df = pd.read_csv(base_m5+'calendar.csv')
        calendar_win = calendar_df[['event_name_2','event_name_1','d']]
        calendar_win['d'] = calendar_win['d'].apply(lambda x: x.split('_',2)[1])

 
        even = calendar_df[['event_name_2','event_name_1']]
        even = even.fillna('')

        even1 = even.shift(-1)
        even1 = even1.fillna('')
        even_after = pd.DataFrame()
        even_after['event_name_2'] = even['event_name_2'].apply(lambda x: x+'_after' if x!='' else x)
        even_after['event_name_1'] = even['event_name_1'].apply(lambda x: x+'_after' if x!='' else x)

        even3 = even_after.shift(1)
        even3 = even3.fillna('')

        even_sum = even1+even3
        even_sum['event_name_2'] = even_sum['event_name_2'].apply(lambda x: np.nan if x=='' else x)
        even_sum['event_name_1'] = even_sum['event_name_1'].apply(lambda x: np.nan if x=='' else x)

        calendar_win['event_name_1_win'] = even_sum['event_name_1']
        calendar_win['event_name_2_win'] = even_sum['event_name_2']
        calendar_win.drop(columns=['event_name_1','event_name_2'],inplace=True)

        calendar_win['event_name_1_win'] = calendar_win['event_name_1_win'].astype('category')
        calendar_win['event_name_2_win'] = calendar_win['event_name_2_win'].astype('category')
        calendar_win['d'] = calendar_win['d'].astype(np.int16)

        grid_df = grid_df.merge(calendar_win, on='d', how='left')

        features_columns=features_columns+['event_name_1_win','event_name_2_win']

        calendar_win = pd.read_pickle(base_pkl+'CA1_TX2_TX3_holidays.pkl')
        calendar_win.rename(columns={'event_name_1_win':'event_name_1_win_1'},inplace=True)

        calendar_win['d'] = calendar_win['d'].apply(lambda x: x.split('_',2)[1])
        calendar_win['d'] = calendar_win['d'].astype(np.int16)

        calendar_win['event_name_1_win_1'] = calendar_win['event_name_1_win_1'].astype('category')
        grid_df = grid_df.merge(calendar_win, on='d', how='left')

        features_columns=features_columns+['event_name_1_win_1']

        lgb_params = lgb_others_params

 
    elif store_id=='WI_1':
        calendar_df = pd.read_csv(base_m5+'calendar.csv')

        calendar_win = calendar_df[['event_name_2','event_name_1','d']]
        calendar_win['d'] = calendar_win['d'].apply(lambda x: x.split('_',2)[1])


        even = calendar_df[['event_name_2','event_name_1']]
        even = even.fillna('')

        even1 = even.shift(-1)
        even1 = even1.fillna('')
        even2 = even.shift(-2)
        even2 = even2.fillna('')

        even_after = pd.DataFrame()
        even_after['event_name_2'] = even['event_name_2'].apply(lambda x: x+'_after' if x!='' else x)
        even_after['event_name_1'] = even['event_name_1'].apply(lambda x: x+'_after' if x!='' else x)

        even3 = even_after.shift(1)
        even3 = even3.fillna('')

        even_sum = even1+even2+even3
        even_sum['event_name_2'] = even_sum['event_name_2'].apply(lambda x: np.nan if x=='' else x)
        even_sum['event_name_1'] = even_sum['event_name_1'].apply(lambda x: np.nan if x=='' else x)

        calendar_win['event_name_1_win'] = even_sum['event_name_1']
        calendar_win['event_name_2_win'] = even_sum['event_name_2']
        calendar_win.drop(columns=['event_name_1','event_name_2'],inplace=True)

        calendar_win['event_name_1_win'] = calendar_win['event_name_1_win'].astype('category')
        calendar_win['event_name_2_win'] = calendar_win['event_name_2_win'].astype('category')
        calendar_win['d'] = calendar_win['d'].astype(np.int16)

        grid_df = grid_df.merge(calendar_win, on='d', how='left')

        features_columns=features_columns+['event_name_1_win','event_name_2_win']
        lgb_params = lgb_others_params


    elif store_id=='WI_3':

        calendar_win = pd.read_pickle(base_pkl+'WI_3_holidays.pkl')
        calendar_win['d'] = calendar_win['d'].apply(lambda x: x.split('_',2)[1])
        calendar_win['d'] = calendar_win['d'].astype(np.int16)

        calendar_win['event_name_1_win'] = calendar_win['event_name_1_win'].astype('category')
        grid_df = grid_df.merge(calendar_win, on='d', how='left')
        features_columns=features_columns+['event_name_1_win']
        lgb_params = lgb_others_params

    else:
        calendar_df = pd.read_csv(base_m5+'calendar.csv')
        calendar_win = calendar_df[['event_name_2','event_name_1','d']]
        calendar_win['d'] = calendar_win['d'].apply(lambda x: x.split('_',2)[1])


        even = calendar_df[['event_name_2','event_name_1']]
        even = even.fillna('')
        even1 = even.shift(-1)
        even1 = even1.fillna('')
        even2 = even.shift(-2)
        even2 = even2.fillna('')
        even_sum = even1+even2
        even_sum['event_name_2'] = even_sum['event_name_2'].apply(lambda x: np.nan if x=='' else x)
        even_sum['event_name_1'] = even_sum['event_name_1'].apply(lambda x: np.nan if x=='' else x)

        calendar_win['event_name_1_win'] = even_sum['event_name_1']
        calendar_win['event_name_2_win'] = even_sum['event_name_2']
        calendar_win.drop(columns=['event_name_1','event_name_2'],inplace=True)

        calendar_win['event_name_1_win'] = calendar_win['event_name_1_win'].astype('category')
        calendar_win['event_name_2_win'] = calendar_win['event_name_2_win'].astype('category')
        calendar_win['d'] = calendar_win['d'].astype(np.int16)

        grid_df = grid_df.merge(calendar_win, on='d', how='left')

        features_columns=features_columns+['event_name_1_win','event_name_2_win']
        lgb_params = lgb_others_params


    print(features_columns)
    print(grid_df.shape)
    train_mask = grid_df['d']<=END_TRAIN   

    valid_mask = train_mask&(grid_df['d']>(END_TRAIN-P_HORIZON))
    preds_mask = grid_df['d']>(END_TRAIN-100)
    train_data = lgb.Dataset(grid_df[train_mask][features_columns], 
                       label=grid_df[train_mask]['sales']) 
    valid_data = lgb.Dataset(grid_df[valid_mask][features_columns], 
                       label=grid_df[valid_mask]['sales'])
    
    grid_df = grid_df[preds_mask].reset_index(drop=True)

    keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]

    grid_df = grid_df[keep_cols]
    grid_df.to_pickle(save_dir+'test_'+store_id+'.pkl')
    del grid_df
    
    seed_everything(SEED)
    estimator = lgb.train(lgb_params, train_data, valid_sets = [valid_data], verbose_eval = 100)
    
    model_name = save_dir+'lgb_model_'+store_id+'_v'+str(VER)+'.bin'
    pickle.dump(estimator, open(model_name, 'wb'))

    del train_data, valid_data, estimator
    gc.collect()

    MODEL_FEATURES = features_columns

def make_lag_roll(LAG_DAY):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d', 'sales']]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])['sales'].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]] 


all_preds = pd.DataFrame()
base_test = get_base_store_test(AUX_MODELS, STORES_IDS, USE_AUX)
main_time = time.time()



for PREDICT_DAY in range(1,29):    
    print('Predict | Day:', PREDICT_DAY)
    start_time = time.time()

    grid_df = base_test.copy()
    grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll,ROLS_SPLIT)], axis=1)
    
    for store_id in STORES_IDS:

        model_path = 'lgb_model_'+store_id+'_v'+str(VER)+'.bin' 
        if USE_AUX:
            model_path = AUX_MODELS + model_path
        
        estimator = pickle.load(open(model_path, 'rb'))
        

        day_mask = base_test['d']==(END_TRAIN+PREDICT_DAY)

        store_mask = base_test['store_id']==store_id

        mask = (day_mask)&(store_mask)

        base_test['sales'][mask] = estimator.predict(grid_df[mask][MODEL_FEATURES])


    temp_df = base_test[day_mask][['id','sales']]
    temp_df.columns = ['id','F'+str(PREDICT_DAY)]
    if 'id' in list(all_preds):
        all_preds = all_preds.merge(temp_df, on=['id'], how='left')
    else:
        all_preds = temp_df.copy()
        
    print('#'*10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f min total |' % ((time.time() - main_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F'+str(PREDICT_DAY)].sum()))
    del temp_df
    

submission = pd.read_csv(base_m5+'sample_submission.csv')[['id']]
submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
submission.to_csv(sub_dir+'store.csv', index=False)








