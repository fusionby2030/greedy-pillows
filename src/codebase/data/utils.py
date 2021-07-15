import pandas as pd # to read the initial filtered separatrix dataset
from sklearn.preprocessing import StandardScaler # to scale data between 0 and 1
import torch # For torch dataset
import os # to check if there is already a pickled form to skip the processing
import pickle # for pickling fun


class ANNtorchdataset(torch.utils.data.Dataset):
    def __init__(self, controls, targets):
        self.inputs = controls
        self.outputs = targets

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, item):
        return self.inputs[item], self.outputs[item]


filename = '/home/adam/data/seperatrix_dataset.csv'
main_eng = ['Ip(MA)', 'B(T)', 'a(m)', 'averagetriangularity',
                 'P_NBI(MW)', 'P_ICRH(MW)','P_TOTPNBIPohmPICRH-Pshi(MW)',
                 'plasmavolume(m3)', 'q95', 'gasflowrateofmainspecies1022(es)']

target = 'nepedheight1019(m-3)'
scale = True
n_samples = 15 # Number of samples for low and high neped in validation set
neped_split = 9.5 # where the high density begins and low density ends

def load_data_torch():
	if os.path.exists('./datasets.pickle'):
		with open('./datasets.pickle', 'rb') as file:
		    datasets = pickle.load(file)
	return datasets
	if scale:
		ss = StandardScaler()
		df[main_eng] = ss.fit_transform(df[main_eng])

	low_neped = df[df[target] < neped_split]
	high_neped = df[df[target] >= neped_split]

	# sample 15 shots from each set
	low_neped_sample = low_neped.sample(n_samples, random_state=42)
	high_neped_sample = high_neped.sample(n_samples, random_state=42)

	# remove those 15 shots from each set
	low_neped.drop(low_neped_sample.index, inplace=True)
	high_neped.drop(high_neped_sample.index, inplace=True)

	# combine removed samples to validation set
	df_sample = pd.concat([low_neped_sample, high_neped_sample])

	# now make into input and outputs set
	train_low_neped = ANNtorchdataset(low_neped[main_eng].to_numpy(np.float32), low_neped[target].to_numpy(np.float32))
	train_high_neped = ANNtorchdataset(high_neped[main_eng].to_numpy(np.float32), high_neped[target].to_numpy(np.float32))
	validation = ANNtorchdataset(df_sample[main_eng].to_numpy(np.float32), df_sample[target].to_numpy(np.float32))

	datasets = (train_low_neped, train_high_neped, validation)
	with open('./datasets.pickle', 'wb') as file:
		pickle.dump(datasets, file)
	return datasets
