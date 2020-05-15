import sys
import numpy as np

#------------------------ print out all arguments -----------------------------#
def print_line(key, val):

    line_max=40
    keylen = len(key)
    remaining_space = line_max - keylen - 1
    print('{0} {2:>{1}}'.format(key, remaining_space, str(val)))

def print_params(params):

    print('----------- Input Parameters -----------')
    params_dict = vars(params)
    for key, val in params_dict.items():
        if isinstance(val, list):
            for subval in val:
                print_line(key, subval)
        else:
            print_line(key, val)
    print('----------------------------------------')


def print_cl(argv):
    
    print('------------------------- Command Line ----------------------------')
    line_len=0
    line_max=65
    for tok in argv:
        if line_len + len(tok) > line_max:
            sys.stdout.write('\\\n' + tok + ' ')
            line_len = len(tok)
        else:
            sys.stdout.write(tok + ' ')
            line_len += len(tok) + 1
    print('')
    print('-------------------------------------------------------------------')
    

#----------------------------------- load model from yaml ---------------------#
def load_model_from_yaml(filename):
    try:
        from parameters import load_params_from_yaml
        from model import fcn_vae
    except ImportError:
        from .parameters import load_params_from_yaml
        from .model import fcn_vae

    params = load_params_from_yaml(filename)
    return fcn_vae(params)


def print_hc(hc, params, rank):
    if rank==0:
        print('')
        print('penalty function enabled')
        print('------------------------')
        print('hit counts file:', params.hc_penalty_path)
        print('hit counts array:')
        for f in hc:
            print(f)
        print('weight (wc) array:')
        _l = np.log(hc)
        wc = (np.sum(_l) / _l) / params.hc_scale_fac
        for f in wc:
            print(f)

