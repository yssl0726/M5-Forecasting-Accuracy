import numpy as np
import pandas as pd
import gc
sub_dir='../sub/'
data_dir='../dataset/'
state = pd.read_csv(sub_dir+'state.csv')
store = pd.read_csv(sub_dir+'store.csv')

## We split the parts of each store separately
CA1_store_final = store[store.id.str.endswith('CA_1_evaluation')]
CA2_store_final = store[store.id.str.endswith('CA_2_evaluation')]
CA3_store_final = store[store.id.str.endswith('CA_3_evaluation')]
CA4_store_final = store[store.id.str.endswith('CA_4_evaluation')]

TX1_store_final = store[store.id.str.endswith('TX_1_evaluation')]
TX2_store_final = store[store.id.str.endswith('TX_2_evaluation')]
TX3_store_final = store[store.id.str.endswith('TX_3_evaluation')]

WI1_store_final = store[store.id.str.endswith('WI_1_evaluation')]
WI2_store_final = store[store.id.str.endswith('WI_2_evaluation')]
WI3_store_final = store[store.id.str.endswith('WI_3_evaluation')]

CA1_store_final.reset_index(drop=True,inplace=True)
CA2_store_final.reset_index(drop=True,inplace=True)
CA3_store_final.reset_index(drop=True,inplace=True)
CA4_store_final.reset_index(drop=True,inplace=True)
TX1_store_final.reset_index(drop=True,inplace=True)
TX2_store_final.reset_index(drop=True,inplace=True)
TX3_store_final.reset_index(drop=True,inplace=True)
WI1_store_final.reset_index(drop=True,inplace=True)
WI2_store_final.reset_index(drop=True,inplace=True)
WI3_store_final.reset_index(drop=True,inplace=True)

## We split the predicted by state into stores for separate processing
CA1_state_final = state[state.id.str.endswith('CA_1_evaluation')]
CA2_state_final = state[state.id.str.endswith('CA_2_evaluation')]
CA3_state_final = state[state.id.str.endswith('CA_3_evaluation')]
CA4_state_final = state[state.id.str.endswith('CA_4_evaluation')]
WI1_state_final = state[state.id.str.endswith('WI_1_evaluation')]
WI2_state_final = state[state.id.str.endswith('WI_2_evaluation')]
WI3_state_final = state[state.id.str.endswith('WI_3_evaluation')]

CA1_state_final.reset_index(drop=True,inplace=True)
CA2_state_final.reset_index(drop=True,inplace=True)
CA3_state_final.reset_index(drop=True,inplace=True)
CA4_state_final.reset_index(drop=True,inplace=True)
WI1_state_final.reset_index(drop=True,inplace=True)
WI2_state_final.reset_index(drop=True,inplace=True)
WI3_state_final.reset_index(drop=True,inplace=True)


## The prediction of by store and by state are weighted, and the weight is determined by their respective local scores

## We found that the weighted integration of local CA stores and WI stores was improved, while TX stores did not. 
## Therefore, we adopted the weighted integration of CA and WI

CA1_merge_final = CA1_store_final.copy()
CA1_merge_final[[f'F{col}' for col in range(1,29)]] = (0.47035/(0.47123+0.47035))*CA1_store_final[[f'F{col}' for col in range(1,29)]] + \
        (0.47123/(0.47123+0.47035))*CA1_state_final[[f'F{col}' for col in range(1,29)]]

CA3_merge_final = CA3_store_final.copy()
CA3_merge_final[[f'F{col}' for col in range(1,29)]] = (0.47100/(0.47059+0.47100))*CA3_store_final[[f'F{col}' for col in range(1,29)]] + \
        (0.47059/(0.47059+0.47100))*CA3_state_final[[f'F{col}' for col in range(1,29)]]
CA4_merge_final = CA4_store_final.copy()
CA4_merge_final[[f'F{col}' for col in range(1,29)]] = (0.47078/(0.47064+0.47078))*CA4_store_final[[f'F{col}' for col in range(1,29)]] + \
        (0.47064/(0.47064+0.47078))*CA4_state_final[[f'F{col}' for col in range(1,29)]]
WI1_merge_final = WI1_store_final.copy()
WI1_merge_final[[f'F{col}' for col in range(1,29)]] = (0.47114/(0.47114+0.47105))*WI1_store_final[[f'F{col}' for col in range(1,29)]] + \
        (0.47105/(0.47114+0.47105))*WI1_state_final[[f'F{col}' for col in range(1,29)]]
WI2_merge_final = WI2_store_final.copy()
WI2_merge_final[[f'F{col}' for col in range(1,29)]] = (0.4696/(0.4696+0.47010))*WI2_store_final[[f'F{col}' for col in range(1,29)]] + \
        (0.47010/(0.4696+0.47010))*WI2_state_final[[f'F{col}' for col in range(1,29)]]
WI3_merge_final = WI3_store_final.copy()
WI3_merge_final[[f'F{col}' for col in range(1,29)]] = (0.47084/(0.47222+0.47084))*WI3_store_final[[f'F{col}' for col in range(1,29)]] + \
        (0.47222/(0.47084+0.47222))*WI3_state_final[[f'F{col}' for col in range(1,29)]]

all_preds = pd.concat([CA1_merge_final,CA2_store_final,CA3_merge_final,CA4_merge_final,TX1_store_final,TX2_store_final,
                        TX3_store_final,WI1_merge_final,WI2_merge_final,WI3_merge_final],axis=0)
all_preds.reset_index(drop=True,inplace=True)  
##  
submission = pd.read_csv(data_dir+'sample_submission.csv')[['id']]
submission = submission.merge(all_preds, on=['id'], how='left').fillna(0)
submission.to_csv(sub_dir+'submission_last'+'.csv', index=False)