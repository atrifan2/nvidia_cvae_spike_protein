# Fully Convolutional Variational Autoencoder

### Getting Started

Assuming data and code volumes are mounted in /projects/data and 
/projects/DOW-Chemical/..., the following will run a single GPU job.

```bash
CUDA_VISISBLE_DEVICES=0 \
python /projects/DOW-Chemical/libraries/models/fcn_vae/run_fcn_vae.py \
--hdf5-file-path /projects/data/chembl/v20_dow/17Sept2018_ChEMBL_37.h5 \
--batch-size 512 \
--checkpoint-period 1 \
--checkpoint-path /projects/checkpoints/rb03b \
--overwrite-checkpoint
```

Kicking off a parallel job in an interactive session is accomplished simply by
prepending an _mpirun_ command to the normal mode of execution.
```bash
mpirun -np 8 \
python /projects/DOW-Chemical/libraries/models/fcn_vae/run_fcn_vae.py \
--hdf5-file-path /projects/data/chembl/v20_dow/17Sept2018_ChEMBL_37.h5 \
--batch-size 512 \
--checkpoint-period 1 \
--checkpoint-path /projects/checkpoints/rb03b \
--overwrite-checkpoint
```

Reloading weights from a previous run and setting up a learning rate scheduler
is illustrated here:
```bash
mpirun -np 8 \
python /projects/DOW-Chemical/libraries/models/fcn_vae/run_fcn_vae.py \
--hdf5-file-path /projects/data/chembl/v20_dow/17Sept2018_ChEMBL_37.h5 \
--batch-size 512 \
--checkpoint-period 100 \
--checkpoint-path /projects/checkpoints/rb03c \
--overwrite-checkpoint \
--saved-encoder-weights /projects/checkpoints/fcn_neuron/encoder_00650 \
--saved-decoder-weights /projects/checkpoints/fcn_neuron/decoder_00650 \
--previous-epochs 650 \
--reduce-lr-callback-patience 50 \
--reduce-lr-callback-property val_loss &> runlog.log &
```

Using the property predictor requires including the flag `--do-prop-pred` 
along with one or more `--prop-add` to specify a property.  Each Property must
be added individually with a separate `--prop-add` option.  See the following
example.
```bash
EXESTRING="python \
/projects/DOW-Chemical/libraries/models/fcn_vae/run_fcn_vae.py \
--hdf5-file-path /projects/data/chembl/v20_dow/17Sept2018_ChEMBL_37.h5 \
--epochs 30000 \
--batch-size 128 \
--optimizer adam \
--hidden-dim 296 \
--learning-rate 1e-5 \
--lr-schedule-patience 500 \
--lr-schedule-prop loss \
--checkpoint-period 100 \
--checkpoint-path $CHKPOINT \
--overwrite-checkpoint \
--enable-tensorboard \
--kl-loss-weight 0.08 \
--kl-start-ramp 30 \
--kl-final-ramp 100 \
--kl-distribution-width .3 \
--dec-filters 37 \
--dec-reslayers 5 \
--do-prop-pred \
--prop-pred-start-ramp 0 \
--prop-pred-final-ramp 5 \
--prop-dropout-prob .5 \
--prop-dense-layer-size 128 \
--prop-add ALogP \
--prop-add LogD \
--prop-add Molecular_SAVol \
--prop-add BIC \
--prop-add CHI_0 \
--prop-add Molecular_Weight
```

### Detailed Help

This help is available from the command-line with the _-h_ flag.

```bash
usage: run_fcn_vae.py [-h] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                      [--optimizer {adam,sgd}] [--learning-rate LEARNING_RATE]
                      [--profile] [--profile-filename PROFILE_FILENAME]
                      [--hidden-dim HIDDEN_DIM] [--activation ACTIVATION]
                      [--enc-kernel-size ENC_KERNEL_SIZE]
                      [--dec-kernel-size DEC_KERNEL_SIZE]
                      [--dec-filters DEC_FILTERS]
                      [--dec-filter-growth-fac DEC_FILTER_GROWTH_FAC]
                      [--dec-reslayers DEC_RESLAYERS] [--MAX-LEN MAX_LEN]
                      [--NCHARS NCHARS] [--random-seed RANDOM_SEED]
                      [--kl-start-ramp KL_START_RAMP]
                      [--kl-final-ramp KL_FINAL_RAMP]
                      [--kl-loss-weight KL_LOSS_WEIGHT]
                      [--kl-distribution-width KL_DISTRIBUTION_WIDTH]
                      [--hdf5-file-path HDF5_FILE_PATH] [--tag TAG]
                      [--checkpoint-path CHECKPOINT_PATH]
                      [--checkpoint-period CHECKPOINT_PERIOD]
                      [--overwrite-checkpoint]
                      [--saved-encoder-weights SAVED_ENCODER_WEIGHTS]
                      [--saved-decoder-weights SAVED_DECODER_WEIGHTS]
                      [--saved-prop-pred-weights SAVED_PROP_PRED_WEIGHTS]
                      [--previous-epochs PREVIOUS_EPOCHS]
                      [--enable-tensorboard]
                      [--lr-schedule-prop LR_SCHEDULE_PROP]
                      [--lr-schedule-patience LR_SCHEDULE_PATIENCE]
                      [--do-prop-pred] [--prop-conv-layers PROP_CONV_LAYERS]
                      [--prop-dense-layers PROP_DENSE_LAYERS]
                      [--prop-kernel-size PROP_KERNEL_SIZE]
                      [--prop-growth-factor PROP_GROWTH_FACTOR]
                      [--prop-dense-layer-size PROP_DENSE_LAYER_SIZE]
                      [--prop-activation PROP_ACTIVATION]
                      [--prop-dropout-prob PROP_DROPOUT_PROB]
                      [--prop-pred-start-ramp PROP_PRED_START_RAMP]
                      [--prop-add PROP_ADD]
                      [--prop-pred-final-ramp PROP_PRED_FINAL_RAMP]
                      [--prop-pred-weight PROP_PRED_WEIGHT]

Fully Convolutional VAE

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size (default: 256)
  --epochs EPOCHS       number of epochs (default: 1500)
  --optimizer {adam,sgd}
                        choice of optimizer (default: sgd)
  --learning-rate LEARNING_RATE
                        learning rate (default: 1e-05)
  --profile             flag to enable profiling (default: False)
  --profile-filename PROFILE_FILENAME
                        filename for profile (default: timeline.tGRU.json)
  --hidden-dim HIDDEN_DIM
                        width of the hidden dim (use default for now!)
                        (default: 296)
  --activation ACTIVATION
                        non-linear activation type (default: relu)
  --enc-kernel-size ENC_KERNEL_SIZE
                        conv kernel size for encoder (default: 5)
  --dec-kernel-size DEC_KERNEL_SIZE
                        conv kernel size for decoder (default: 5)
  --dec-filters DEC_FILTERS
                        number of conv filters for decoder (default: 37)
  --dec-filter-growth-fac DEC_FILTER_GROWTH_FAC
                        growth factor (do not tune for now) (default: 1.0)
  --dec-reslayers DEC_RESLAYERS
                        number of res layer blocks for decoder (default: 7)
  --MAX-LEN MAX_LEN     length of one-hot SMILES string (do not change)
                        (default: 296)
  --NCHARS NCHARS       length of one-hot SMILES dict (do not change)
                        (default: 37)
  --random-seed RANDOM_SEED
                        random seed (default: 42)
  --kl-start-ramp KL_START_RAMP
                        epoch number to begin linear ramping of KL loss
                        (default: 10)
  --kl-final-ramp KL_FINAL_RAMP
                        epoch number to end linear ramping of KL loss
                        (default: 30)
  --kl-loss-weight KL_LOSS_WEIGHT
                        weight value of KL loss (default: 0.03)
  --kl-distribution-width KL_DISTRIBUTION_WIDTH
                        width of the sample distribution in the variational
                        layer (default: 1.0)
  --hdf5-file-path HDF5_FILE_PATH
  --tag TAG             HDF5 base tag (default: 17Sept2018_ChEMBL_37)
  --checkpoint-path CHECKPOINT_PATH
                        path to directory for model checkpoints (default:
                        /tmp)
  --checkpoint-period CHECKPOINT_PERIOD
                        number of epochs between saves. zero disables.
                        (default: 0)
  --overwrite-checkpoint
                        flag to continue if checkpoint directory exists
                        (default: False)
  --saved-encoder-weights SAVED_ENCODER_WEIGHTS
                        encoder weight file to reload (default: False)
  --saved-decoder-weights SAVED_DECODER_WEIGHTS
                        decoder weight file to reload (default: False)
  --saved-prop-pred-weights SAVED_PROP_PRED_WEIGHTS
                        property predictor weight file to reload (default:
                        False)
  --previous-epochs PREVIOUS_EPOCHS
                        start value for epoch (if reloading model) (default:
                        0)
  --enable-tensorboard  enables tensorboard (default: False)
  --lr-schedule-prop LR_SCHEDULE_PROP
                        property monitored by ReduceLRCallback (default:
                        val_loss)
  --lr-schedule-patience LR_SCHEDULE_PATIENCE
                        number of epochs to wait before reducing LR. 0
                        disables (default: 0)
  --do-prop-pred        do property prediction task (default: False)
  --prop-conv-layers PROP_CONV_LAYERS
                        not implemented (default: 3)
  --prop-dense-layers PROP_DENSE_LAYERS
                        number of dense layers in property predictor (default:
                        3)
  --prop-kernel-size PROP_KERNEL_SIZE
                        not implemented (default: 5)
  --prop-growth-factor PROP_GROWTH_FACTOR
                        not implemented (default: 1.5)
  --prop-dense-layer-size PROP_DENSE_LAYER_SIZE
                        width of dense layers in prop predictor (default: 256)
  --prop-activation PROP_ACTIVATION
                        activation for property predictor (default: relu)
  --prop-dropout-prob PROP_DROPOUT_PROB
                        dropout probability for property predictor. (0
                        disables) (default: 0.0)
  --prop-pred-start-ramp PROP_PRED_START_RAMP
                        epoch number to begin linear ramping of prop pred
                        (default: 50)
  --prop-add PROP_ADD   regression property string corresponding to named
                        series in HDF data (default: None)
  --prop-pred-final-ramp PROP_PRED_FINAL_RAMP
                        epoch number of top of linear ramp of prop pred
                        (default: 4000)
  --prop-pred-weight PROP_PRED_WEIGHT
                        final loss weight of prop predictor (default: 1.0)
```
