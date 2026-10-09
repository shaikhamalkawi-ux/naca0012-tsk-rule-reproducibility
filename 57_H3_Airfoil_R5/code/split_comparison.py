from common import *
import pandas as pd,numpy as np
rows=[]
for f in ('F4','F5'):
 for model in ('linear_ridge','svr','extra_trees','xgboost','TSK-LS-oneSE','TSK-Ridge-oneSE'):
  pair={}
  for mode in ('group','row'):
   if model.startswith('TSK'):
    a=pd.read_csv(ROOT/'results'/'tsk'/f'{mode}_log_{f}'/'OUTER_PREDICTIONS.csv')
   else:
    a=pd.read_csv(ROOT/'results'/'baselines'/f'{mode}_log_{f}'/model/'OUTER_PREDICTIONS.csv')
   z=a[a.model==model];assert len(z)==1503
   pair[mode+'_row_RMSE']=row_rmse(z.y.to_numpy(),z.pred.to_numpy())
   pair[mode+'_equal_configuration_RMSE']=group_rmse(z.y.to_numpy(),z.pred.to_numpy(),z.group_id.to_numpy())
  rows.append({'feature':f,'model':model,**pair})
a=pd.DataFrame(rows);a.to_csv(ROOT/'results'/'TABLE_GROUP_VS_ROW_PROTOCOL.csv',index=False)
print(a.to_string(index=False))
