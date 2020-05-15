import h5py, warnings
import argparse, os
import numpy as np
from glob import glob
from utils import cm_to_cvae, read_h5py_file
from test_tfrs import convert_to_tfr_2
import numpy as np
from glob import glob
from utils import cm_to_cvae, read_h5py_file
from test_tfrs import convert_to_tfr_2
import sklearn
from sklearn.model_selection import train_test_split
import json

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--sim_path", dest='f', help="Input: OpenMM simulation path")
parser.add_argument("-o", help="Output: CVAE 2D contact map h5 input file")

# Let's say I have a list of h5 file names
args = parser.parse_args()

#cm_filepath = os.path.abspath(os.path.join(args.f, 'omm*/*_cm.h5'))

cm_file = 'contact_maps_sprs.h5'
cvae_input = read_h5py_file(cm_file)
train_data, val_data = train_test_split(cvae_input[:], test_size=.20, random_state=42)
train_len = dict(
                train = int(len(train_data))
                            )
with open('./train_len.json', 'w') as f:
    json.dump(train_len, f)

    #convert_to_tfr_1(train_data, './float-train.tfrecords')
    # convert_to_tfr_1(val_data, './float-val.tfrecords')

convert_to_tfr_2(train_data, './bytes-train.tfrecords')
convert_to_tfr_2(val_data, './bytes-val.tfrecords')
