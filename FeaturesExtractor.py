from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from cae_models import baselineCAE, inceptionCAE
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def raw(x_train_normal, x_test):
    features_train = x_train_normal
    features_test = x_test
    return features_test, features_train


def cae(cae_type, x_train_normal, x_test_normal, x_test):
    logger.info('building autoencoder model: ' + cae_type)
    img_dim = x_test.shape[1:]       
    if cae_type == 'inception':
        filters = [8, 16, 32]
        autoencoder = inceptionCAE(img_dim, filters)
        extractionLayer = 45; gap = True
    else:
        autoencoder = baselineCAE(img_dim)
        extractionLayer = 15; gap = False

    # fitting the model
    logger.info('fitting autoencoder model')
    autoencoder.compile(optimizer=Adam(lr = 1e-4),  loss='mse')

    #e1=250; e2=100
    e1=1; e2=1
    autoencoder.fit(x_train_normal, x_train_normal, 
                    epochs= e1, batch_size=200, shuffle=True,
                    validation_data=(x_test_normal, x_test_normal), verbose=1)

    K.set_value(autoencoder.optimizer.lr, 1e-5)

    autoencoder.fit(x_train_normal, x_train_normal, 
                    epochs= e2, batch_size=200, shuffle=True,
                    validation_data=(x_test_normal, x_test_normal), verbose=1)

    logger.info('extracting features from autoencoder model')
    f = autoencoder.layers[extractionLayer].output
    if gap: f = GlobalAveragePooling2D()(f)

    feature_model = Model(inputs=autoencoder.input, outputs=f)


    ##### Testing: NND Anomaly Detection #####

    # extract train and test image features from the CAE's bottleneck layer
    features_train = feature_model.predict(x_train_normal)
    start = time.time()
    features_test = feature_model.predict(x_test)
    test_runtime = time.time() - start
    return features_train, features_test, test_runtime
        

