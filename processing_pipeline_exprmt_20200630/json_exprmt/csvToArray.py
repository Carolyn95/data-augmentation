import numpy as np
import pdb
import pandas as pd
new_path = "NEW.csv"
update_path = "UPDATE.csv"

new_df = pd.read_csv(new_path)
update_df = pd.read_csv(update_path)

new_sents = np.array(new_df.EmailContent)
update_sents = np.array(update_df.EmailContent)

new_labels = np.array([l.lower() for l in new_df.EmailIntent])
update_labels = np.array([l.lower() for l in update_df.EmailIntent])

sents = np.concatenate((new_sents, update_sents))
labels = np.concatenate((new_labels, update_labels))

np.save("json_all_bodies.npy", sents)
np.save("json_all_labels.npy", labels)
