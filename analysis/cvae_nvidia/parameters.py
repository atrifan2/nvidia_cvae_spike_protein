import argparse
import yaml

# NOTE: new arguments must only use dashes, not underscores. 
# If adding a new argument, best to follow an example from below.
def fcn_parameters():

    p = argparse.ArgumentParser(description='Fully Convolutional VAE',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, 
            default=10, help='batch size')
    p.add_argument('--epochs', type=int,
            default=1500, help='number of epochs')
    p.add_argument('--optimizer', type=str.lower, choices=['adam', 'sgd'],
            default='adam', help='choice of optimizer')
    p.add_argument('--learning-rate', type=float,
            default=1e-5, help='learning rate')
    p.add_argument('--profile',
            action='store_true', help='flag to enable profiling')
    p.add_argument('--profile-filename', type=str,
            default='timeline.tGRU.json', help='filename for profile')
    p.add_argument('--hidden-dim', type=int,
            default=150, help='width of the hidden dim (use default for now!)')
    p.add_argument('--activation', type=str,
            default='relu', help='non-linear activation type')
    p.add_argument('--enc-kernel-size', type=int,
            default=5, help='conv kernel size for encoder')
    p.add_argument('--dec-kernel-size', type=int,
            default=5, help='conv kernel size for decoder')
    p.add_argument('--dec-filters', type=int,
            default=1200, help='number of conv filters for decoder')
    p.add_argument('--dec-filter-growth-fac', type=float,
            default=1.0, help='growth factor (do not tune for now)')
    p.add_argument('--dec-reslayers', type=int,
            default=3, help='number of res layer blocks for decoder')
    p.add_argument('--MAX-LEN', type=int,
            default=1200, help='length of one-hot SMILES string (do not change)')
    p.add_argument('--NCHARS', type=int,
            default=1200, help='length of one-hot SMILES dict (do not change)')
    p.add_argument('--random-seed', type=int,
            default=42, help='random seed')
    p.add_argument('--kl-start-ramp', type=int,
            default=10, help='epoch number to begin linear ramping of KL loss')
    p.add_argument('--kl-final-ramp', type=int,
            default=30, help='epoch number to end linear ramping of KL loss')
    p.add_argument('--kl-loss-weight', type=float, 
            default=0.03, help='weight value of KL loss')
    p.add_argument('--kl-distribution-width', type=float,
            default=1.0, help='width of the sample distribution in the variational layer')
    p.add_argument('--hdf5-file-path', type=str,
            default='/projects/data/production.h5', help='')
    p.add_argument('--tag', type=str,
            default='17Sept2018_ChEMBL_37', help='HDF5 base tag')
    p.add_argument('--checkpoint-path',
            default='/tmp', help='path to directory for model checkpoints')
    p.add_argument('--checkpoint-period', type=int,
            default=0, help='number of epochs between saves. zero disables.')
    p.add_argument('--overwrite-checkpoint', action='store_true', 
            help='flag to continue if checkpoint directory exists')
    p.add_argument('--saved-encoder-weights', 
            default=False, help='encoder weight file to reload')
    p.add_argument('--saved-decoder-weights', 
            default=False, help='decoder weight file to reload')
    p.add_argument('--saved-prop-pred-weights', 
            default=False, help='property predictor weight file to reload')
    p.add_argument('--previous-epochs', type=int,
            default=0, help='start value for epoch (if reloading model)')
    p.add_argument('--enable-tensorboard', action='store_true', 
            help='enables tensorboard')
    p.add_argument('--lr-schedule-prop', default='val_loss',
            help='property monitored by ReduceLRCallback')
    p.add_argument('--lr-schedule-patience', type=int, default=0,
            help='number of epochs to wait before reducing LR. 0 disables')
    p.add_argument('--lr-schedule-min', type=float,
            default=0.0, help='minimum learning rate for scheduler')
    p.add_argument('--lr-schedule-cooldown', type=int, 
            default=0, help='cool-down time for lr scheduler')
    p.add_argument('--do-prop-pred', action='store_true',
            help='do property prediction task')
    p.add_argument('--prop-conv-layers', type=int,
            default=3, help='not implemented')
    p.add_argument('--prop-dense-layers', type=int,
            default=3, help='number of dense layers in property predictor')
    p.add_argument('--prop-kernel-size', type=int,
            default=5, help='not implemented')
    p.add_argument('--prop-growth-factor', type=float,
            default=1.5, help='not implemented')
    p.add_argument('--prop-dense-layer-size', type=int,
            default=256, help='width of dense layers in prop predictor')
    p.add_argument('--prop-activation', 
            default='relu', help='activation for property predictor')
    p.add_argument('--prop-dropout-prob', type=float,
            default=0.0, help='dropout probability for property predictor. (0 disables)')
    p.add_argument('--prop-pred-start-ramp', type=int,
            default=50, help='epoch number to begin linear ramping of prop pred')
    p.add_argument('--prop-add', action='append',
            help='regression property string corresponding to named series in HDF data')
    p.add_argument('--prop-pred-final-ramp', type=int,
            default=4000, help='epoch number of top of linear ramp of prop pred')
    p.add_argument('--prop-pred-weight', type=float,
            default=1.0, help='final loss weight of prop predictor')
    p.add_argument('--prop-pred-initial-weight', type=float,
            default=0.0, help='initial loss weight of prop predictor')
    p.add_argument('--decoder-sampling-temperature', type=float,
            default=0.0, help='add non-deterministic sampling to decoder')
    p.add_argument('--random-train-data', action='store_true',
            help='use random data for training')
    p.add_argument('--yaml-dump-then-exit', action='store_true',
            help='write out yaml, exit without training')
    p.add_argument('--hc-penalty-path', type=str,
            default='', help='path to hit counts txt file')
    p.add_argument('--hc-scale-fac', type=float,
            default=240.0, help='scale factor to divide wc for penalty VAE')
    p.add_argument('--random-train-size', type=int,
            default=100, help='specifies x in train.shape(x, 1200, 1200)')

    return p


#------------------------ print parameters to yaml file------------------------#
def dump_yaml(params, filename='input_parameters.yaml'):
    with open(filename, 'w') as fp:
        yamlstring = yaml.dump(vars(params), default_flow_style=False)
        fp.write(yamlstring)


def load_yaml(filename):
    with open(filename, 'r') as fp:
        yamldict = yaml.load(fp)
    return yamldict


def prepend_key_to_all_vals(arg_list, key, val):
    first_pass=1
    for v in val:
        if not first_pass:
            arg_list.append(key)
        else:
            first_pass=0
        arg_list.append(v)


def generate_argparse_string(dict_in):
    arg_list = []
    for key, val in dict_in.items():
        key = '--'+key.replace('_','-')
        if str(val).upper()=='FALSE': continue
        arg_list.append(key)
        if str(val).upper()=='TRUE': continue

        # handle lists (like for prop_add)
        if isinstance(val, list):
            prepend_key_to_all_vals(arg_list, key, val)
        else:
            arg_list.append(str(val))
    return arg_list


def load_params_from_yaml(filename):
    p = fcn_parameters()
    yamldict = load_yaml(filename)
    arg_list = generate_argparse_string(yamldict)
    params = p.parse_args(arg_list)
    return params
