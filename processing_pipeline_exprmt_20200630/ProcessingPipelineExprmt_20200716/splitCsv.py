## csv: new and update only
from EmailDataFactory import * 
save_file_prefix = './processed_data/dec_'  # all_org/all_noempty_
label_data = np.load(
    './data/jan_intentions.npy',
    allow_pickle=True) # dec_intentions
email_data = np.load(
    './data/jan_bodies.npy',
    allow_pickle=True) # dec_bodies

la = LabelArray(label_data)
idx_l = la.processLable()

ea = EmailArray(email_data)
idx_e = ea.preprocessEmail()

indexes_to_remove = np.array(list(set(idx_l + idx_e)))

label_array, email_array = removeProblemIndexes(la.labels,
                                                ea.emails, indexes_to_remove)
label_array, email_array = filterByLabel(save_file_prefix,
                                        label_array,
                                        email_array,
                                        labels=['update', 'new'])
## csv: 5 splits
for i in range(5):                                        
  split_data_path = './processed_data/split_' + str(i) + '/'
  prefix = 'jan_' # dec_
  balanceSplitDataset(split_data_path, prefix, label_array, email_array)
