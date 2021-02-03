import pandas as pd
import os
import glob


rid_csv1 = '/home/bolero/.dc/dl/dataset/detection/idc_cancer/RID_dataset_idc_c16.csv'
rid_csv2 = '/home/bolero/.dc/dl/dataset/detection/idc_cancer/RID_dataset_newendo.csv'

col_names = ['RID', 'pT', 'GUBUN']

df1 = pd.read_csv(rid_csv1).loc[:, ['RID', 'TSTAGE', 'GUBUN']]
df1.columns = col_names

df2 = pd.read_csv(rid_csv2).loc[:, ['RID', 'pT', 'GUBUN']]
df2.columns = col_names

df_total = pd.concat([df1, df2], axis=0)

print(df_total)
df_total.to_csv("total.csv", index=False)
