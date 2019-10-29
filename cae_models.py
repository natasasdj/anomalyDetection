from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import LeakyReLU, Activation
from keras.layers import concatenate, Flatten, Reshape
from keras.models import Model
from keras import regularizers

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def inception_layer(x, filters):
    # 1x1 convolution
    x0 = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x0 = BatchNormalization()(x0)
    x0 = LeakyReLU(alpha=0.1)(x0)
    # 3x3 convolution
    x1 = Conv2D(filters, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)
    # 5x5 convolution
    x2 = Conv2D(filters, (5,5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(alpha=0.1)(x2)
    # Max Pooling
    x3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    x3 = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x3)
    x3 = BatchNormalization()(x3)
    x3 = LeakyReLU(alpha=0.1)(x3)
    output = concatenate([x0, x1, x2, x3], axis = 3)
    return output


##### Inception-like Convolutional AutoEncoder #####

def inceptionCAE(img_dim, filters):
    # input
    input_img = Input(shape=img_dim) # adapt this if using `channels_first` image data format
    # encoder
    x = inception_layer(input_img, filters[0])
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = inception_layer(x, filters[1])
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = inception_layer(x, filters[2])
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = x
    #decoder
    x = inception_layer(x, filters[2])
    x = UpSampling2D((2, 2))(x)
    x = inception_layer(x, filters[1])
    x = UpSampling2D((2, 2))(x)
    x = inception_layer(x, filters[0])
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(img_dim[2], (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    decoded = x
    # model
    autoencoder = Model(input_img, decoded)
    return autoencoder


def baselineCAE(img_dim):
    #input
    input_img = Input(shape=img_dim)
    # encoder
    encoding_dim = 128
    x = Conv2D(32, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(encoding_dim, kernel_regularizer=regularizers.l2(1e-6))(x)
    x = LeakyReLU(alpha=0.1)(x)
    encoded = x
    #decoder
    x = Reshape((4,4,encoding_dim//16))(x)
    x = Conv2D(128, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(img_dim[2], (5, 5), padding='same', kernel_regularizer=regularizers.l2(1e-6))(x)           
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    decoded = x
    # model
    autoencoder = Model(input_img, decoded)
    return autoencoder
