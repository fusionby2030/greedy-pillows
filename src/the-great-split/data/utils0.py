import pandas as pd # to read the initial filtered separatrix dataset
from sklearn.preprocessing import StandardScaler # to scale data between 0 and 1
import torch # For torch dataset
import os # to check if there is already a pickled form to skip the processing
import pickle # for pickling fun
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'
class ANNtorchdataset(torch.utils.data.Dataset):
	def __init__(self, controls, targets):
		self.inputs = controls
		self.outputs = targets

	def __len__(self):
		return len(self.outputs)

	def __getitem__(self, item):
		return self.inputs[item], self.outputs[item]



 # Number of samples for low and high neped in validation set
 # where the high density begins and low density ends

def load_data_torch(random_st=42, neped_split = 9.5, n_samples = 15, scale = True, **kwargs):
	"""if os.path.exists('./data/datasets.pickle'):
		with open('./data/datasets.pickle', 'rb') as file:
			datasets = pickle.load(file)
		return datasets"""

	if kwargs.get('conditions'):
		target = kwargs['conditions']
	else:
		target = 'nepedheight1019(m-3)'


	if kwargs.get('main_engineering_inputs'):
		main_eng = kwargs['main_engineering_inputs']
	else:
		# print('No main_eng parameter supplied, using default')
		main_eng = ['Ip(MA)', 'B(T)', 'a(m)', 'averagetriangularity',
						 'P_NBI(MW)', 'P_ICRH(MW)','P_TOTPNBIPohmPICRH-Pshi(MW)',
						 'plasmavolume(m3)', 'q95', 'gasflowrateofmainspecies1022(es)']
	if kwargs.get('data_loc'):
		filename = kwargs['data_loc']
	else:
		filename = '/home/adam/data/seperatrix_dataset.csv'

	df = pd.read_csv(filename)
	if scale:
		ss = StandardScaler()
		# ss_2 = StandardScaler()
		df[main_eng] = ss.fit_transform(df[main_eng])
		# df[target] = ss_2.fit_transform(df[main_eng])

	# feature_space = df[main_eng].to_numpy()
	# target_space = df[target].to_numpy()
	# Min: 1.8494685
	# print('min', target_space.min())
	# Max: 11.737379
	# print('max', target_space.max())
	low_neped = df[df[target] < neped_split]
	high_neped = df[df[target] >= neped_split]

	# sample 15 shots from each set
	low_neped_sample = low_neped.sample(n_samples, random_state=random_st)
	high_neped_sample = high_neped.sample(n_samples, random_state=random_st)

	# remove those 15 shots from each set
	low_neped.drop(low_neped_sample.index, inplace=True)
	high_neped.drop(high_neped_sample.index, inplace=True)

	# combine removed samples to validation set
	df_sample = pd.concat([low_neped_sample, high_neped_sample])
	low_neped_set = (low_neped[main_eng].to_numpy(np.float32), low_neped[target].to_numpy(np.float32))
	high_neped_set = (high_neped[main_eng].to_numpy(np.float32), high_neped[target].to_numpy(np.float32))
	final_exam_set = (df_sample[main_eng].to_numpy(np.float32), df_sample[target].to_numpy(np.float32))
	# now make into input and outputs set
	# train_low_neped = ANNtorchdataset(low_neped[main_eng].to_numpy(np.float32), low_neped[target].to_numpy(np.float32))
	# train_high_neped = ANNtorchdataset(high_neped[main_eng].to_numpy(np.float32), high_neped[target].to_numpy(np.float32))
	# validation = ANNtorchdataset(df_sample[main_eng].to_numpy(np.float32), df_sample[target].to_numpy(np.float32))
	dataset = (low_neped_set, high_neped_set, final_exam_set)
	"""
	with open('./data/datasets.pickle', 'wb') as file:
		pickle.dump(dataset, file)

	"""
	return dataset, ss
