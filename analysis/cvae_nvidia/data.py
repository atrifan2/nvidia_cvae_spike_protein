from loaders.types import SplitTypes
import horovod.keras as hvd
from loaders.hdf5_generator3 import DatasetGeneratorFast
import numpy as np

def setup_generators(params):

    # train/valid splits
    train_split = '{}/index/{}/{}'.format(params.tag, 'LogD', SplitTypes.train)
    valid_split = '{}/index/{}/{}'.format(params.tag, 'LogD', SplitTypes.valid)
    
    # outputs
    if params.do_prop_pred:
        normalize_y=True
        regression_prediction_columns = params.prop_add
        output_datasets = ['{}/data/values/{}'.format(params.tag, x) \
                               for x in regression_prediction_columns]
        if hvd.rank()==0:
            for ix, val in enumerate(output_datasets):
                print('regression output:', val)
    else:
        normalize_y=[]
        regression_prediction_columns = []
        output_datasets = []
        
    # inputs
    input_datasets = ['{}/data/one_hot/{}'.format(params.tag, x) \
                                              for x in ['smiles']]

    # setup generators
    train_gen = DatasetGeneratorFast(h5store=params.hdf5_file_path,
                                    vae_params={'hidden_dim':params.hidden_dim},
                                    batch_size=params.batch_size,
                                    xlabel=input_datasets,
                                    ylabel=output_datasets,
                                    normalize_X=False,
                                    normalize_y=normalize_y,
                                    splitlabel=train_split)

    valid_gen = DatasetGeneratorFast(h5store=params.hdf5_file_path,
                                    vae_params={'hidden_dim':params.hidden_dim},
                                    batch_size=params.batch_size,
                                    xlabel=input_datasets,
                                    ylabel=output_datasets,
                                    normalize_X=False,
                                    normalize_y=normalize_y,
                                    splitlabel=valid_split)
    return train_gen, valid_gen        


def random_data(params):

    examples = int(117000.0 // params.batch_size) * params.batch_size
    randx = np.random.rand(examples, params.MAX_LEN, params.NCHARS)
    z_mean_log_var = np.ones((examples, 592))
    return randx, [ randx, z_mean_log_var ]

