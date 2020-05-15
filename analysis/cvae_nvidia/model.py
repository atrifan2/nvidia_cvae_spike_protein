from keras.layers import Dense, Input, Flatten, Conv1D, Activation, Dropout
from keras.layers import Lambda, Reshape, BatchNormalization, Add, Concatenate
from keras.layers import UpSampling1D, Layer
from keras.models import Model
from keras import backend as K
import numpy as np

################################################################################
# Derived parameters
################################################################################
def calculate_derived_parameters(params):
    '''
    these are architecture-specific derived parameters which 
    are not user-settable.  
    '''
    # calculate the number of layers required in the encoder.
    # This is a function of MAX_LEN.  We are computing the number
    # of times we need to divide MAX_LEN by two before we get to one.
    # i.e., solving 2^x = MAX_LEN for x.
    params.enc_reslayers = int(np.ceil(np.log(params.MAX_LEN) / np.log(2)))

    # the number of starting filters we use for the first
    # Conv1D.  Subsequent number of filters are computed
    # from the growth factor, enc_filter_growth_fac
    params.enc_filters = params.NCHARS
    
    # calculate the growth factor required to get to desired 
    # hidden dim as a function of enc_reslayers and NCHARS.
    # i.e., solving: NCHARS * x^enc_reslayers = hidden_dim; for x
    ratio = params.hidden_dim / params.NCHARS
    params.enc_filter_growth_fac = ratio**(1.0 / (params.enc_reslayers - 1))


    # think about upsampling / downsampling in the decoder
    # initialize these flags to zero
    params.upsample_rounds = 0
    params.shrink_rounds = 0

    # assert that MAX_LEN is a multiple of hidden_dim. 
    # this is prevent the need for zero-padding.  We will
    # just use Upsampling.
    if params.hidden_dim < params.MAX_LEN:
        err_msg = 'MAX_LEN must be a multiple of hidden_dim'
        assert params.MAX_LEN % params.hidden_dim == 0, err_msg

        params.upsample_rounds = params.MAX_LEN / params.hidden_dim - 1

    # if we choose a larger hidden_dim, then we must be able
    # to get back to MAX by using a strided conv.  So we 
    # must confirm that MAX_LEN is a multiple of hidden_dim
    if params.hidden_dim > params.MAX_LEN:
        err_msg = 'hidden_dim must be a multiple of MAX_LEN'
        assert params.hidden_dim % params.MAX_LEN == 0, err_msg

        params.shrink_rounds = params.hidden_dim / params.MAX_LEN




################################################################################
# Residual layer definition
################################################################################
def Residual1DConv(x, 
                   filters, 
                   kernel_size, 
                   activation='relu',
                   name='res1', 
                   shrink=False, 
                   kfac=2):

    res = Conv1D(filters,
               kernel_size=1,
               padding='same',
               name=name+'_1x1')(x)

    x = BatchNormalization(name=name+'_bn1')(x)
    x = Activation(activation, name=name+'_act1')(x)
    x = Conv1D(filters,
               kernel_size,
               padding='same',
               name=name+'_conv1D1')(x)

    x = BatchNormalization(name=name+'_bn2')(x)
    x = Activation(activation, name=name+'_act2')(x)
    x = Conv1D(filters,
               kernel_size,
               padding='same',
               name=name+'_conv1D2')(x)
    x = Add(name=name+'_add')([x, res])

    if shrink:
        x = Conv1D(filters=filters,
               kernel_size=kfac,
               strides=kfac,
               padding='same',
               activation=activation,
               name=name+'strided')(x)

    return x


################################################################################
# Gaussian sampling layer definition
################################################################################
class GaussianNoiseSampling(Layer):
    """Apply additive zero-centered Gaussian noise
    during decoding (not during training).
    This is a modified version (basically the opposite) 
    of Keras version GaussianNoise layer.
    # Arguments
        stddev: float, standard deviation of the noise distribution.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """
    def __init__(self, stddev, **kwargs):
        super(GaussianNoiseSampling, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            return inputs + K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return K.in_train_phase(inputs, noised, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianNoiseSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


################################################################################
# model definition
################################################################################

#------------------------------ encoder model ---------------------------------#
def encoder_model(params):

    x_in = Input(shape=(params.MAX_LEN, params.NCHARS), 
               name='input_smiles')

    x = Conv1D(filters=params.enc_filters,
               kernel_size=5,
               strides=1,
               dilation_rate=1,
               padding='same',
               activation=None,
               use_bias=True)(x_in)

    for lidx in range(params.enc_reslayers):
        
        filters = params.enc_filters * params.enc_filter_growth_fac**lidx
        filters = int(round(filters))

        x = Residual1DConv(x, 
               filters, 
               params.enc_kernel_size, 
               name='res'+str(lidx), 
               shrink=True)

    encoder_h = Flatten(name='encoder_hidden')(x)

    l_sigma = Lambda(K.identity, name='sigma')(encoder_h)

    mu = Dense(params.hidden_dim, 
                    activation='linear',
                    name='mu')(encoder_h)

    return Model(x_in, [mu, l_sigma], name='encoder')


#------------------------------ middle layers ---------------------------------#
def variational_layer(params, mu, l_sigma):

    def sample_z(args):
        mu, l_sigma = args
        eps = K.random_normal(shape=(params.batch_size, 
                                     params.hidden_dim), 
                              mean=0., stddev=1.)
        z_rand = mu + K.exp(l_sigma / 2) * eps * params.kl_distribution_width
        return K.in_train_phase(z_rand, mu)

    # save for kl loss for output. We won't do anything else with this. 
    mu_l_sigma_output = Concatenate(name='z_mean_log_var')([mu, l_sigma])

    z = Lambda(sample_z, output_shape = (params.hidden_dim, ))([mu, l_sigma])

    return z, mu_l_sigma_output

#----------------------------- decoder layers ---------------------------------#
def decoder_model(params):

    z = Input(shape=(params.hidden_dim,),
              name='decoder_in')

    x = Reshape((params.hidden_dim, 1))(z)

    x = GaussianNoiseSampling(stddev=params.decoder_sampling_temperature)(x)

    for lidx in range(params.dec_reslayers):
        
        filters = params.dec_filters * params.dec_filter_growth_fac**lidx
        filters = int(round(filters))

        if params.shrink_rounds:
            params.shrink_rounds -= 1

        x = Residual1DConv(x, 
               filters, 
               params.dec_kernel_size, 
               shrink=params.shrink_rounds,
               name='res'+str(lidx+params.enc_reslayers))

        if params.upsample_rounds:

            x = UpSampling1D()(x)

            params.upsample_rounds -= 1

    recon = Conv1D(params.NCHARS,
               kernel_size=params.dec_kernel_size,
               name='x_pred',
               padding='same',
               activation='relu')(x)

    return Model(z, recon, name='decoder')

#----------------------------- property predictor -----------------------------#
def prop_predictor_definition(params):

    n_outputs = len(params.prop_add)

    z = Input(shape=(params.hidden_dim,),
              name='prop_in')

    x = z

    for _ in range(params.prop_dense_layers):
        size = params.prop_dense_layer_size
        activation = params.prop_activation
        x = Dense(size, activation=activation)(x)

        if params.prop_dropout_prob:
            x = Dropout(params.prop_dropout_prob)(x)

    x = Dense(n_outputs)(x)

    return Model(z, x, name='reg_prop_pred')

#------------------------------ fcn_vae Model ---------------------------------#
def fcn_vae(params):

    calculate_derived_parameters(params)

    encoder = encoder_model(params)

    x_in = encoder.inputs[0]
    mu, l_sigma = encoder(x_in)

    z, mu_l_sigma_output = variational_layer(params, mu, l_sigma)

    decoder = decoder_model(params)
    recon = decoder(z)
    
    recon = Lambda(K.identity, name='x_pred')(recon)
    model_outputs = [recon, mu_l_sigma_output]

    if params.do_prop_pred:
        prop_pred_model = prop_predictor_definition(params)
        regression_out = prop_pred_model(z)
        model_outputs.append(regression_out)
    else:
        prop_pred_model = None

    fcn_model = Model(x_in, model_outputs, name='fcn_vae')

    return fcn_model, encoder, decoder, prop_pred_model
