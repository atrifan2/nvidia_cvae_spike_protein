import os 
import time
# import sys, errno
import argparse 
import json 
from cvae.CVAE import train_cvae, predict_from_cvae
import horovod.tensorflow as hvd
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--h5_file", dest="f", default='cvae_input.h5', 
    help="Input: contact map h5 file")
parser.add_argument(
    "-d", "--dim", default=3, 
    help="Number of dimensions in latent space"
    )
parser.add_argument("-g", "--gpu", default=0, help="gpu_id") 
parser.add_argument(
    "-b", "--batch_size", default=1000, 
    help="Batch size for CVAE training") 
parser.add_argument(
    "-n", "--num_frame", default="all", 
    help="Number of frames used for training") 
parser.add_argument(
    "-e", "--epochs", default=100, 
    help="Number of epochs for CVAE training")

args = parser.parse_args()

cvae_input = args.f
hyper_dim = int(args.dim) 
gpu_id = args.gpu
batch_size = int(args.batch_size) 
num_frame = args.num_frame 
epochs = int(args.epochs)

if not os.path.exists(cvae_input):
    raise IOError('Input file doesn\'t exist...')


if __name__ == '__main__':
    cvae_info_dict = {} 

    # Train CVAE model 
    start_train = time.time()
    cvae, train_time = train_cvae(
        gpu_id, cvae_input, 
        input_size=num_frame, 
        hyper_dim=hyper_dim, 
        epochs=epochs, 
        batch_size=batch_size,
        )
    end_train = time.time()

    # Collect information for performance assessment of CVAE 
    cvae_info_dict['train_time'] = train_time  
    #cvae_info_dict['loss'] = cvae.history.losses 
    print('Training time is ', train_time)

    # Save trained models information
    #if hvd.rank() == 0: 
    model_weight = 'cvae_weight_%d.h5'%hvd.rank()
    model_file = 'cvae_model_%d.h5'%hvd.rank() 
    cvae.model.save_weights(model_weight)
    cvae.save(model_file)

    # Prediction using trained CVAE 
    start_predict = time.time() 
    cm_embeded, predict_time = predict_from_cvae(cvae, cvae_input, num_frame)
    end_predict = time.time() 

    # Collect information for performance assessment of CVAE 
    cvae_info_dict['predict_time'] = predict_time  
    print('Predicting time is ', predict_time)

    #if hvd.rank() == 0: 
    cvae_info_file = 'time_%d_%dg_%db_%de.json'%(hvd.rank(), hvd.size(), batch_size, epochs) 
    with open(cvae_info_file, 'w') as fp: 
        json.dump(cvae_info_dict, fp)

    
