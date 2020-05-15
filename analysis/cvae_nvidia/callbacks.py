from keras.callbacks import Callback
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras import backend as K
from datetime import datetime
import sys
import os

class print_loss(Callback):
    '''callback to print some stats during training'''
    def __init__(self, nepochs=1):
        super(print_loss, self).__init__()
        self.nepochs=nepochs
        super().__init__()
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch%self.nepochs==0:
            print('\n---        epoch {:4d}       ---'.format(epoch))
            print('--- ',datetime.now())
            for key in logs:
                print('{:9.7f}  {}'.format(logs[key], key))
            sys.stdout.flush()


class ramp_schedule(object):
    def __init__(self, start_epoch, start_weight, final_epoch, final_weight):
        '''
                      (final_epoch, final_weight)         (N, final_weight)
                    /----------------
                   /
        ----------/ (start_epoch, start_weight) 
        '''
        self.start_epoch = start_epoch    
        self.start_weight = float(start_weight)
        self.final_epoch = final_epoch
        self.final_weight = float(final_weight)
        self.slope = (final_weight - start_weight) / (final_epoch - start_epoch) 

    def __call__(self, epoch):
        if epoch < self.start_epoch:
            return self.start_weight
        elif epoch < self.final_epoch:
            return self.start_weight + (epoch - self.start_epoch)*self.slope
        else:
            return self.final_weight


class generic_scheduler(Callback):
    def __init__(self, var, schedule, verbose=True, varname='var'):
        super(generic_scheduler, self).__init__()
        self.var = var
        self.schedule = schedule
        self.verbose = verbose
        self.varname = varname

    def on_epoch_begin(self, epoch, logs=None):
        new_weight = self.schedule(epoch)
        K.set_value(self.var, new_weight)
        if self.verbose:
            print(self.varname + ' annealer:', new_weight)


class model_checkpoint(Callback):
    def __init__(self, encoder, decoder, prop_pred, 
                 savepath, nepochs=1, overwrite=True):
        super(model_checkpoint, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prop_pred = prop_pred
        self.savepath = savepath
        self.nepochs = nepochs

        if os.path.exists(savepath):
            if not overwrite: 
                raise ValueError('checkpoint directory already exists. ' +\
                                 'Consider option --overwrite-checkpoint')
        else:
            os.makedirs(savepath)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch%self.nepochs==0:
            encoder_ffn = os.path.join(self.savepath, 
                                       'encoder_{:05d}'.format(epoch))
            decoder_ffn = os.path.join(self.savepath, 
                                       'decoder_{:05d}'.format(epoch))
            prop_pred_ffn = os.path.join(self.savepath, 
                                       'prop_pred_{:05d}'.format(epoch))
            self.encoder.save_weights(encoder_ffn)
            self.decoder.save_weights(decoder_ffn)
            if self.prop_pred:
                self.prop_pred.save_weights(prop_pred_ffn)


def setup_kl_callback(params):

    kl_loss_var = K.variable(params.kl_loss_weight)

    kl_loss_schedule = ramp_schedule(start_epoch=params.kl_start_ramp,
                                    start_weight=0.0,
                                    final_epoch=params.kl_final_ramp,
                                    final_weight=params.kl_loss_weight)

    kl_callback = generic_scheduler(var=kl_loss_var,
                                    varname='kl_loss',
                                    schedule=kl_loss_schedule,
                                    verbose=0)

    return kl_loss_var, kl_callback


def setup_prop_pred_callback(params):

    prop_pred_var = K.variable(params.prop_pred_weight)

    pp_loss_schedule = ramp_schedule(start_epoch=params.prop_pred_start_ramp,
                                   start_weight=params.prop_pred_initial_weight,
                                   final_epoch=params.prop_pred_final_ramp,
                                   final_weight=params.prop_pred_weight)

    pp_callback = generic_scheduler(var=prop_pred_var,
                                   varname='prop_pred_loss',
                                   schedule=pp_loss_schedule,
                                   verbose=hvd.rank()==0)


    return prop_pred_var, pp_callback


def setup_hvd_callbacks(params, callbacks, encoder, decoder, prop_pred):
    import horovod.keras as hvd

    # Horovod: broadcast initial variable states
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

    # Horovod: average metrics among workers at the end of every epoch.
    callbacks.append(hvd.callbacks.MetricAverageCallback())

    # Horovod: Scale the learning rate * hvd.size()`
    callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5,
                                                      verbose=(hvd.rank()==0)))


def setup_callbacks(params, callbacks, encoder, decoder, prop_pred):
    import horovod.keras as hvd

    # model checkpointing
    if params.checkpoint_period and hvd.rank()==0:
            model_checkpoint_callback = model_checkpoint(encoder, 
                                     decoder,
                                     prop_pred,
                                     params.checkpoint_path,
                                     nepochs=params.checkpoint_period,
                                     overwrite=params.overwrite_checkpoint)
            callbacks.append(model_checkpoint_callback)

    # LR scheduler
    if params.lr_schedule_patience:
        lr_callback = ReduceLROnPlateau(monitor=params.lr_schedule_prop, 
                                     factor=0.5, 
                                     patience=params.lr_schedule_patience,
                                     min_lr=params.lr_schedule_min * hvd.size(),
                                     cooldown=params.lr_schedule_cooldown,
                                     verbose=(hvd.rank()==0))
        callbacks.append(lr_callback)

    if hvd.rank()==0:
        callbacks.append(print_loss())
        if params.enable_tensorboard:
            callbacks.append(TensorBoard(params.checkpoint_path))




