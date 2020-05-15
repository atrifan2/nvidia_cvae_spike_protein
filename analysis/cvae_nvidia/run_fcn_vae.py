################################################################################
# Fully convolutional VAE 
#
# note, the below uses data generators that do not exist in this project
################################################################################

import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from datetime import datetime
import argparse

from keras.optimizers import Adam, SGD
from keras import backend as K
from keras import objectives

import tensorflow as tf
from tensorflow.python.client import timeline

import horovod.keras as hvd

from model import fcn_vae
from callbacks import setup_kl_callback, setup_hvd_callbacks, setup_callbacks
from utils import print_params, print_cl
from parameters import fcn_parameters

################################################################################
# Parameters
################################################################################

p = fcn_parameters()

################################################################################
# Horovod: initialize Horovod.
################################################################################

hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
tfconfig =  tf.compat.v1.ConfigProto() #tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.gpu_options.visible_device_list = str(hvd.local_rank())
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tfconfig))

################################################################################
# Argument handling
################################################################################

params = p.parse_args()

if hvd.rank()==0:
    print_cl(sys.argv)
    print_params(params)

if params.yaml_dump_then_exit:
    sys.exit(0)

#-------------------------------- optimizer -----------------------------------#
if params.optimizer=='adam':
    params.optimizer = Adam(lr=params.learning_rate * hvd.size())
elif params.optimizer=='sgd':
    params.optimizer = SGD(lr=params.learning_rate * hvd.size())

params.optimizer = hvd.DistributedOptimizer(params.optimizer)

#------------------------------- model reloading ------------------------------#

reloading_model = False
if params.saved_decoder_weights or \
   params.saved_encoder_weights or \
   params.previous_epochs:
       reloading_model = True
       assert params.saved_decoder_weights and \
              params.saved_encoder_weights and \
              params.previous_epochs, 'if reloading a model: encoder '+ \
                    'weights, decoder weights, and previous epochs '+   \
                    'must all be specified'

################################################################################
# Setup data generators
################################################################################

if not params.random_train_data:
    #train_gen, valid_gen = setup_generators(params)
    print('not implemented: please provide your data here')
    print('please run with --random-train-data')
    sys.exit(1)
elif params.random_train_data:
    #randx, randy = random_data(params)
    n_steps = params.random_train_size
    n_atoms = 1200
    n_coords = 3
    hidden_dim = 150
    xyz = np.random.random((n_steps, n_atoms, n_coords))
    dismat = np.array([ distance_matrix(i, i) for i in xyz ])
    y_true = np.zeros((n_steps, hidden_dim*2))
    if hvd.rank()==0:
        print('Initialized synthetic data shaped: ', dismat.shape)

################################################################################
# Profiling
################################################################################
if params.profile:
    trace_level=tf.compat.v1.RunOptions.FULL_TRACE
else:   
    trace_level=tf.compat.v1.RunOptions.NO_TRACE
#tf.compat.v1.disable_eager_execution()
run_options = tf.compat.v1.RunOptions(trace_level=trace_level)
run_metadata = tf.compat.v1.RunMetadata()

################################################################################
# Load model
################################################################################

print('Initializing Horovod rank', hvd.rank())

model, encoder, decoder, prop_pred = fcn_vae(params)

if reloading_model:
    encoder.load_weights(params.saved_encoder_weights)
    decoder.load_weights(params.saved_decoder_weights)
    if params.saved_prop_pred_weights:
        params.do_prop_pred=True
        prop_pred.load_weights(params.saved_prop_pred_weights)

if hvd.rank()==0:
    encoder.summary()
    decoder.summary()
    if params.do_prop_pred:
        prop_pred.summary()
    model.summary()

################################################################################
# Callbacks
################################################################################

callbacks = []

if hvd.size()>1:
    setup_hvd_callbacks(params, callbacks, encoder, decoder, prop_pred)

kl_loss_var, kl_callback = setup_kl_callback(params)
callbacks.append(kl_callback)

if params.do_prop_pred:
    prop_pred_var, prop_pred_ramp_callback = setup_prop_pred_callback(params)
    callbacks.append(prop_pred_ramp_callback)

setup_callbacks(params, callbacks, encoder, decoder, prop_pred)

################################################################################
# Compile model
################################################################################

#--------------------- model losses and loss weights --------------------------#
def custom_loss(hc, scale_fac):
    
    # compute wc
    # need to make these tf.tensors
    # batch nor MAX_LEN dims do not need to be included - broadcasting rules
    # take care of this
    _l = np.log(hc)
    wc = tf.constant(np.sum(_l) / _l) / tf.constant(scale_fac)
    
    def penalized_crossent_loss(target, output, axis=-1):
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        
        # convert 
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        
        # compute cross-entropy, scaled by wc computed by function factory
        return -tf.reduce_sum(wc * target * tf.log(output), axis)
    
    return penalized_crossent_loss


def KL_loss(y_true, mu_l_sigma_output):
    mu, l_sigma = tf.split(mu_l_sigma_output, 2, axis=1)
    KL = -0.5 * K.mean(1 + l_sigma - K.square(mu) - K.exp(l_sigma), axis=-1)
    return KL

def recon_loss(y_true, y_pred):
    return objectives.mean_squared_error(y_true, y_pred)

def reg_loss(y_true, y_pred):
    return objectives.mean_squared_error(y_true, y_pred)

if not params.hc_penalty_path:
    # no penalty path specified, use regular recon loss
    model_losses = [ recon_loss, KL_loss ]
else:
    # load the hc weights
    hc = np.loadtxt(params.hc_penalty_path, dtype=np.float32)
    print_hc(hc, params, hvd.rank())
    model_losses = [ custom_loss(hc, params.hc_scale_fac), KL_loss ]

loss_weights = [ 1.0, kl_loss_var ]
if params.do_prop_pred:
    model_losses.append(reg_loss)
    loss_weights.append(prop_pred_var)    

#------------------------------- metrics --------------------------------------#
# metric functions to track annealing variables
def kl_weight_anneal(y_true, y_pred): return kl_loss_var
def  reg_prop_anneal(y_true, y_pred): return prop_pred_var

metrics={'x_pred': ['categorical_accuracy', kl_weight_anneal]}
if params.do_prop_pred:
    metrics['reg_prop_pred'] = reg_prop_anneal

#----------------------------- model compile ----------------------------------#
model.compile(loss=model_losses,
              loss_weights=loss_weights,
              options=run_options,                  # PROFILING
              run_metadata=run_metadata,            # PROFILING
              metrics=metrics,
              optimizer=params.optimizer)

################################################################################
# Fit simple model
################################################################################

if not params.random_train_data:
    model.fit_generator(generator=train_gen,
                        verbose=False,
                        steps_per_epoch=len(train_gen)// hvd.size(),
                        callbacks=callbacks,
                        epochs=params.epochs,
                        initial_epoch=params.previous_epochs,
                        validation_data=valid_gen,
                        validation_steps=max(1, len(valid_gen)//hvd.size()),
                        use_multiprocessing=False)
elif params.random_train_data:
              model.fit(x=dismat, 
                        y=[dismat, y_true],
                        batch_size=params.batch_size,
                        epochs=params.epochs,
                        callbacks=callbacks,
                        verbose=hvd.rank()==0)

################################################################################
# Profiling: Final writeout
################################################################################
if params.profile:
    # get the graph from the tf session
    graph = K.get_session().graph
    trace = timeline.Timeline(step_stats=run_metadata.step_stats,
                              graph=graph)
    if hvd.rank()==0:
        with open(params.profile_filename,'w') as f:
            f.write(trace.generate_chrome_trace_format(show_memory=True))

if hvd.rank()==0:
    print('completed gracefully')
    print(datetime.now())
