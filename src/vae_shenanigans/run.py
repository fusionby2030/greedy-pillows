import argparse
import numpy as np


from lightning_experiment import VAEXperiment

from models import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument("-bs", "--batch_size", help='batch size for training', type=int, default=300)
parser.add_argument("-ep", "--epochs", help='num epochs', type=int, default=50)
parser.add_argument("-lr","--learning_rate", help='learning rate', type=float, default=0.01)
parser.add_argument("-ld", "--latent_dim", help='Latent Dimensions of AE', default=2, type=int)
parser.add_argument('-hslist', '--hidden_dims', help='List of hidden layers [h1_size, h2_size, ..., ]', nargs='+', default=[150, 300], type=int)
parser.add_argument('-cond', '--conditional_inputs', help='List of condiitonal inputs to give, for longer list, see README.md', nargs='+', default=['nepedheight1019(m-3)'], type=str)
parser.add_argument('-main_eng', '--main_engineering_inputs', help='List of main eng. inputs to give that will be reconstructed, for longer list, see README.md', nargs='+', default=['Ip(MA)', 'B(T)', 'a(m)', 'averagetriangularity',
                 'P_NBI(MW)', 'P_ICRH(MW)','P_TOTPNBIPohmPICRH-Pshi(MW)',
                 'plasmavolume(m3)', 'q95', 'gasflowrateofmainspecies1022(es)'], type=str)
parser.add_argument('-data', '--data_loc', help='File location of data you wish to feed', type=str, default='/home/adam/data/seperatrix_dataset.csv')
parser.add_argument('-log', '--log_dir', help='Location of logging', type=str, default='./vae_exps')
parser.add_argument('-exp_name', '--experiment_name', help='Name of Experiment', type=str, default='STANDALONE')
parser.add_argument("-seed", "--torch_seed", help='Set manual seed for reproducability', default=42, type=int)
parser.add_argument("-vae", "--vae_type", help='Which VAE to use, choose VanillaVAE if you do not know', default='VanillaVAE', type=str)
parser.add_argument("-wd", "--weight_decay", help='Weight decay in optimizer', default=0.0, type=float)

args = parser.parse_args()
config = vars(args)
config['input_dim'] = len(config['main_engineering_inputs'])
config['cond_dim'] = len(config['conditional_inputs'])

model_param_keys = ["input_dim", "cond_dim", "hidden_dims", "latent_dim"]
model_params = {key: config[key] for key in model_param_keys}

trainer_params = {"max_epochs": args.epochs}
tt_logger = TestTubeLogger(save_dir = config['log_dir'], name=config['experiment_name'])
torch.manual_seed(config['torch_seed'])
np.random.seed(config['torch_seed'])

model = vae_models[config['vae_type']](**model_params)
model.double()
experiment = VAEXperiment(model, config)

runner = Trainer(logger=tt_logger, log_every_n_steps=5, **trainer_params)

print("================ Training =============")
runner.fit(experiment)
