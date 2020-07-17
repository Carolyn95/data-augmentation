import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
dummy = np.load('dummy.npy', allow_pickle=True)


scp_perm = np.load('../split_4_rep/scp_permutation_series.npy')
rev_scp_perm = np.load('../split_4_rep/rev_scp_permutation_series.npy')

dummy_scp = dummy[scp_perm]
dummy_rev_scp = dummy[rev_scp_perm]

dummy_scp_json = np.array([i for i, l in enumerate(dummy_scp) if l == 'json'])
dummy_rev_scp_json = np.array([i for i, l in enumerate(dummy_rev_scp) if l == 'json'])

scp_result = pd.read_csv('../split_4_rep/scp_model/result.csv')
rev_scp_result = pd.read_csv('../split_4_rep/rev_scp_model/result.csv')

import pdb
# pdb.set_trace()
scp_json_result = scp_result.loc[dummy_scp_json]
rev_scp_json_result = rev_scp_result.loc[dummy_rev_scp_json]

scp_json_result.to_csv('./scp_json_result.csv')
rev_scp_json_result.to_csv('./rev_scp_json_result.csv')

print('scp')
print(accuracy_score(scp_json_result.GroundTruth, scp_json_result.Pred))
print(confusion_matrix(scp_json_result.GroundTruth, scp_json_result.Pred))
print('rev_scp')
print(accuracy_score(rev_scp_json_result.GroundTruth, rev_scp_json_result.Pred))
print(confusion_matrix(rev_scp_json_result.GroundTruth, rev_scp_json_result.Pred))
