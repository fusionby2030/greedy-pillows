"""
Here we want to do some searching for the optimal ANN
for transfer learning.


Requirements:
    1. CV - Shall split the low neped, and test for splits in high neped
        a. Low neped - l splits
        b. high neped - h splits
        results in l*h trained models
        ?? SHOULD WE BE GRADING ON THE FINAL EXAM, i.e., on transfer learning, the goal is to get best of both worlds??
        ?? MAYBE TRY BOTH, AT MOMENT IT IS JUST TRYING FOR THE OPTIMAL PREDICTIONS ON HIGH NEPED??

    2. Metrics
        a. low neped RMSE and MAE (2*l scores -> average with std)
        b. high neped RMSE and MAE (2*l*h scores -> average with std)
        c. final exam RMSE and MAE
            c1. for pre transfer
            c2. for post transfer

    3. Model
        a. Fitting function
            - cache best save and return at the end (Average Torch Regressor has this as default)
            - that way epochs is irrelevant
        b. layer names, '0.0' (layer 1), '1.0' (layer 2), '2.0' (layer 3), '3.0' (layer 4), 'out'

    4. Hyperparameters
        a. architectures (hidden_layer_sizes, list)
        b. which layer to freeze (e.g., '0.0' or 'out', string)
        c. number estimators (n_estimators, int)
        c. for low AND high neped models
            - batch_size (bs_l, bs_h, int)
            - learning_rate (lr_l, lr_h, float)
            - # splits (l_splits, h_splits, int)
"""

import torch
import torch.nn as nn
from peanuts.models.utils import set_module_torch, save_load_torch
from peanuts.models.torch_ensembles import AverageTorchRegressor
import numpy as np

# Data
from data import utils
from sklearn.model_selection import KFold

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import nni

# Misc
import argparse
import tqdm
import logging

logger = logging.getLogger("nni")

class FFNN_transfer(nn.Module):
    def __init__(self, hidden_layer_sizes, act_func_name=None):
        super(FFNN_transfer, self).__init__()
        # Take hidden layer sizes

        target_size = 1
        input_size = 10
        out_act = torch.nn.ReLU()
        act_func = set_module_torch.get_act_func(act_func_name)

        last_size = input_size

        self.hidden_layers = torch.nn.ModuleList()
        for size in hidden_layer_sizes:
            self.hidden_layers.append(self._fc_block(last_size, size, act_func))
            last_size = size

        self.out = self._fc_block(last_size, target_size, out_act)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        x = self.out(x)
        return x

    @staticmethod
    def _fc_block(in_c, out_c, act_func):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            act_func
        )
        return block

    def predict(self, X):
        self.eval()
        pred = None

        if isinstance(X, torch.Tensor):
            pred = self.forward(X)

        elif isinstance(X, np.ndarray):
            X = torch.Tensor(X)
            pred = self.forward(X)

        else:
            msg = 'The type of input to ensemble should be a torch.tensor or np.ndarray'
            raise ValueError(msg)

        return pred

def initalize_model(n_estimators, hidden_layer_sizes, learning_rate, state_dict=None):
    """
    Create a new average torch regressor using the state dict passed
    If no state dict is passed, i.e., None, model will be blank canvas AverageTorchRegressor (THIS SHOULD ONLY HAPPEN AT AT EACH CV low neped loop)

    param:
        state_dict: torch state dictionary for loading AverageTorchRegressor
        hidden_layer_sizes: [h1, h2, etc..,], list of ints
        n_estimators: number of estimators in avereage torch ensemble, int
    non-passed-params:
        base_model: FFNN_transfer (see above)

    returns:
        to return or not to return that is the question.
        initial idea is yes
    """
    model = AverageTorchRegressor(estimator=FFNN_transfer, n_estimators=n_estimators, estimator_args = {'hidden_layer_sizes': hidden_layer_sizes})
    if state_dict:
        save_load_torch.load_cache_save(model, state=state_dict)
    else:
        pass
    model.set_optimizer('Adam', lr=learning_rate)
    model.float()
    return model

def freeze_model(model, non_freeze):
    for name, param in model.named_parameters():
        if param.requires_grad and non_freeze in name:
            pass
        else:
            param.requires_grad = False

    model.float()

def pseudo_main(args):
    # Gather Data and prepare for CV split
    # 3 sets of data, low neped, high neped, and final exam (15 points from each)
    datasets = utils.load_data_torch()
    low_neped, high_neped, final_exam = datasets

    # initialize metric memory
    low_neped_scores_pre_transfer = np.zeros((2, args.l_splits))
    high_neped_scores = np.zeros((2, args.l_splits*args.h_splits))
    final_exam_pre_transfer = np.zeros((2, args.l_splits))
    final_exam_post_transfer= np.zeros((2, args.l_splits*args.h_splits))

    # Index of which split we are on
    h_split_index = l_split_index = 0


    # for loop with low neped spliting
    cv_low = KFold(n_splits=args.l_splits, shuffle=True, random_state=42)
    low_neped_iterator = tqdm.tqdm(enumerate(cv_low.split(low_neped[0])), desc='pre transfer learning')
    for num, (train, test) in low_neped_iterator:
        torch.manual_seed(10)
        # X_train, y_train = low_neped[0][train], low_neped[1][train]
        # X_test, y_test = low_neped[0][test], low_neped[1][test]
        train_set = utils.ANNtorchdataset(low_neped[0][train], low_neped[1][train])
        test_set = utils.ANNtorchdataset(low_neped[0][test], low_neped[1][test])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size_low, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_low, shuffle=True)

        # train a model on low neped split
        # Initalize Baby Model
        model = initalize_model(n_estimators= args.n_estimators, hidden_layer_sizes = args.hidden_layer_sizes, learning_rate=args.learning_rate_low)
        model_dict_cache = model.fit(train_loader=train_loader, test_loader=test_loader, epochs=args.epochs_low)
        # load cached model
        model = initalize_model(n_estimators= args.n_estimators, hidden_layer_sizes = args.hidden_layer_sizes, learning_rate=args.learning_rate_low, state_dict=model_dict_cache)
        # calc. low neped RMSE and MAE ->  top: RMSE, bot: MAE, low_neped_scores[0][l_split_index] = RMSE_l_split
        # calc. pre_transfer_final_exam
        low_neped_predictions = model.predict(test_set.inputs)
        low_neped_scores_pre_transfer[:, l_split_index] = mean_squared_error(y_true=test_set.outputs, y_pred=low_neped_predictions, squared=False), mean_absolute_error(y_true=test_set.outputs, y_pred=low_neped_predictions)

        final_exam_predictions = model.predict(final_exam[0])
        final_exam_pre_transfer[:, l_split_index] = mean_squared_error(y_true=final_exam[1], y_pred=final_exam_predictions, squared=False), mean_absolute_error(y_true=final_exam[1], y_pred=final_exam_predictions)

        nni.report_intermediate_result(-np.log10(final_exam_pre_transfer[:, l_split_index][0]))
        # plot_comparison(true_vals=test_set.outputs, predictions_low=low_neped_predictions, predictions_high=None)
        # plot_comparison(true_vals=high_neped[1], predictions_low=high_neped_predictions, predictions_high=None)
        # plot_comparison(true_vals=final_exam[1], predictions_low=final_exam_predictions, predictions_high=None)

        low_neped_iterator.set_postfix(scores=final_exam_pre_transfer[:, l_split_index])
        l_split_index += 1


        cv_high = KFold(n_splits=args.h_splits, shuffle=True, random_state=42)
        high_neped_iterator = tqdm.tqdm(enumerate(cv_high.split(high_neped[0])), desc='transfer learning')
        # for loop with high neped split
        for num_h, (train_h, test_h) in high_neped_iterator:
            # Set up high neped sets and loaders
            train_set_h = utils.ANNtorchdataset(high_neped[0][train_h], high_neped[1][train_h])
            test_set_h = utils.ANNtorchdataset(high_neped[0][test_h], high_neped[1][test_h])
            train_loader_h = torch.utils.data.DataLoader(train_set_h, batch_size=args.batch_size_high, shuffle=True)
            test_loader_h = torch.utils.data.DataLoader(test_set_h, batch_size=args.batch_size_high, shuffle=True)
            # Freeze model layers
            freeze_model(model, args.freeze_layer)
            # train on high neped split
            model_dict_cache_h = model.fit(train_loader=train_loader_h, test_loader=test_loader_h, epochs=args.epochs_high)

            model = initalize_model(n_estimators=args.n_estimators, hidden_layer_sizes=args.hidden_layer_sizes, learning_rate=args.learning_rate_high, state_dict=model_dict_cache_h)

            # calc high neped RMSE and MAE (same as low neped style, just with h_split_index)
            # calc. post_transfer_final_exam
            high_neped_predictions = model.predict(test_set_h.inputs)
            high_neped_scores[:, h_split_index] = mean_squared_error(y_true=test_set_h.outputs, y_pred=high_neped_predictions, squared=False), mean_absolute_error(y_true=test_set_h.outputs, y_pred=high_neped_predictions)

            final_exam_predictions = model.predict(final_exam[0])
            final_exam_post_transfer[:, h_split_index] = mean_squared_error(y_true=final_exam[1], y_pred=final_exam_predictions, squared=False), mean_absolute_error(y_true=final_exam[1], y_pred=final_exam_predictions)
            high_neped_iterator.set_postfix(scores=final_exam_post_transfer[:, h_split_index])
            # plot_comparison(true_vals=final_exam[1], predictions_low=final_exam_predictions, predictions_high=None)

            h_split_index += 1
            # reinitalize model from before transfer learning
            model = initalize_model(n_estimators= args.n_estimators, hidden_layer_sizes = args.hidden_layer_sizes, learning_rate=args.learning_rate_low, state_dict=model_dict_cache)


    # final result -> average low/high_neped_scores with np.mean(low/high_neped_scores, 1)
    final_exam_pre_tran_score = (np.mean(final_exam_pre_transfer, 1), np.std(final_exam_pre_transfer, 1))
    final_exam_post_tran_score = (np.mean(final_exam_post_transfer, 1), np.std(final_exam_post_transfer, 1))
    msg = "Pre Transfer RMSE: {:.4} +- {:.4} \n Post Transfer RMSE {:.4} +/- {:.4}".format(final_exam_pre_tran_score[0][0], final_exam_pre_tran_score[1][0], final_exam_post_tran_score[0][0], final_exam_pre_tran_score[1][0])
    # save result and relevant hyperparams to a dataframe and export to csv?
    print(msg)
    scores_dict = {'RMSE_pre_transfer': final_exam_pre_tran_score, 'RMSE_post_transfer': final_exam_post_tran_score}
    nni_dict= {'default': -np.log10(final_exam_post_tran_score[0][0]), 'pre-trans': -np.log10(final_exam_pre_tran_score[0][0])}
    nni.report_final_result(nni_dict)

def plot_comparison(true_vals, predictions_low, predictions_high, RMSE_dict = None, output_loc=None):
    import matplotlib.pyplot as plt
    SMALL_SIZE = 40
    MEDIUM_SIZE = 45
    BIGGER_SIZE = 50

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axs = plt.subplots(1, 1, figsize=(18, 18))
    axs.scatter(true_vals, predictions_low, s=100)
    axs.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--')
    # axs.scatter(true_vals, predictions_high, s=100, label='After Transfer: {:.4}'.format(RMSE_dict['RMSE_high']))
    axs.set(title='Transfer Learning on split $n_e^{ped} \geq 9.5 x 10^{21}$', xlabel='True $n_e^{ped} (10^{21}$m$^{-3})$', ylabel='Predicted')
    plt.legend()
    plt.show()
    if output_loc is not None:
        file_name = output_loc + "transfer-learning_trial"  + '.png'
        # print(file_name)
        plt.savefig(file_name)
    else:
        plt.show()

def update_args(args, new_params):
    pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs_l", '--batch_size_low', help='batch size during initial training on low neped', type=int, default=1)
    parser.add_argument("-bs_h", '--batch_size_high', help='batch size during transfer training on high neped', type=int, default=1)
    parser.add_argument('-lr_l', "--learning_rate_low", help='learning rate for initial training on low neped', type=float, default=0.004)
    parser.add_argument('-lr_h', "--learning_rate_high", help='learning rate for transfer learning on high neped', type=float, default=0.00001)
    parser.add_argument('-ep_l', "--epochs_low", help='Number of epochs in low neped training (pre-transfer)', type=int,default=150)
    parser.add_argument('-ep_h', "--epochs_high", help='Number of epochs in low neped training (pre-transfer)', type=int,default=150)
    parser.add_argument('-ls', '--l_splits', help='Number of CV splits for low neped training (pre transfer)', default=4, type=int)
    parser.add_argument('-hs', '--h_splits', help='Number of CV splits for high neped training (transfer)', default=4, type=int)

    parser.add_argument('-hslist', '--hidden_layer_sizes', help='List of hidden layers [h1_size, h2_size, ..., ]', nargs='+', default=[100, 100, 100, 100], type=int)
    parser.add_argument('-n_est', '--n_estimators', help='Number of ANNs in ensemble, 1 is default ANN',type=int, default=1)
    parser.add_argument('-fl', '--freeze_layer', help='Which layer to NOT freeze', default='out', const='out', nargs='?', choices=['5.0', '4.0', '3.0', '2.0', '1.0', '0.0', 'out'])
    args = parser.parse_args()

    try:
        updated_params = nni.get_next_parameter()
        d = vars(args)
        d.update(updated_params)

        pseudo_main(args)
    except Exception as e:
        raise e
