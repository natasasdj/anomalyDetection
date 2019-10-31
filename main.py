import os
import argparse
import pandas as pd
import FeaturesExtractor
import AnomalyDetection
from numpy.random import seed
from tensorflow import set_random_seed
from load_data import load_data, get_normal_data
from sklearn.metrics import roc_auc_score

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# input:
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="type of datasets: mnist, fmnist, cifar10, cifar100", required=True, choices=['mnist', 'fmnist', 'cifar10', 'cifar100'])
ap.add_argument("-nc", "--normal_class", help="normal class, if dataset is 'cifar100' it is in range 0..19, otherwise it is in range 0..9", required=True, type=int)
ap.add_argument("-fe", "--features_extractor", help="type of features extractor: raw (raw images) or cae (convolutional autoencoder)", choices=['raw','cae'], default='cae')
ap.add_argument("-cae", "--cae_type", help="type of CAE (Convolutional Autoencoder): baseline, inception", choices=['baseline','inception'], default='inception')
ap.add_argument("-ad", "--anomaly_detection", help="anomaly detection method: ocsvm - One Class Support Vector Machines, nnd - exact distance-based, qnnd - approximated distance-based with product quantization", choices=['ocsvm','nnd', 'qnnd'], default='qnnd')
ap.add_argument("-odir", "--output_dir", help="directory to store the output results", required=True)
ap.add_argument("-rs", "--random_seed", help="random seed", type=int, default=0)
ap.add_argument("-k", "--param_k", help="nnd and qnnd parameter k", type=int, default=1)
ap.add_argument("-m", "--param_m", help="qnnd parameter m", type=int, default=1, choices = [1,2,4,8,16,32,64,128])
ap.add_argument("-c", "--param_c", help="qnnd parameter c", type=int, default=1, choices = [1,2,3,4,5,6,7,8])


args = vars(ap.parse_args())

# check value of the normal_class argument

if args['dataset'] == 'cifar100':
    if (args['normal_class']<0) or (args['normal_class']>19):
        ap.error('Normal class must be in range 0..19')
else:
    if (args['normal_class']<0) or (args['normal_class']>9):
        ap.error('Normal class must be in range 0..9')

        
dataset = args ['dataset']
normal_class = args ['normal_class']
features_extractor = args ['features_extractor']
cae_type = args ['cae_type']
anomaly_detection = args ['anomaly_detection']
fdir = args ['output_dir']
random_seed = args ['random_seed']
k = args['param_k']
m = args['param_m']
c = args['param_c']


# set random seed
seed(random_seed); set_random_seed(random_seed)

### data load ###
logger.info('load data')
(x_train, y_train), (x_test, y_test) = load_data(dataset)

# create the training and test data with normal class only for training CAE's 
x_train_normal, x_test_normal = get_normal_data(x_train, y_train, x_test, y_test, normal_class)

### make features_extractor model and extract features for training and test dataset images
logger.info('extract features')
if features_extractor == 'cae':
    features_train, features_test, featuresExtractTime  = FeaturesExtractor.cae(cae_type, x_train_normal, x_test_normal, x_test)
else:
    featuresExtractTime = 0
    features_train, features_test = FeaturesExtractor.raw(x_train_normal, x_test)

### anomaly detection
logger.info('anomaly detection: calculate anomaly scores and auc') 
if anomaly_detection == 'ocsvm': 
    scores, anomalyDetectTime = AnomalyDetection.ocsvm(features_train, features_test)
    labels_test = y_test.flatten() == normal_class
elif anomaly_detection == 'nnd':
    scores, anomalyDetectTime = AnomalyDetection.nnd(features_train, features_test, k)
    labels_test = y_test.flatten() != normal_class
elif anomaly_detection == 'qnnd':
    scores, anomalyDetectTime = AnomalyDetection.qnnd(features_train, features_test, k, m, c)
    labels_test = y_test.flatten() != normal_class

# calculate auc    
auc_ =  100*roc_auc_score(labels_test, scores)


### write auc into output file
logger.info('write auc results into an output file')
# create resulting output directory if not exists
if not os.path.exists(fdir):
    os.makedirs(fdir)
    
# create pd.DataFrame for writing auc results or read auc results from the file if it already exists
if features_extractor == 'raw':
    fname = os.path.join(fdir, 'auc' + '_fe:'+ features_extractor + '_ad:' + anomaly_detection + '_data:' + dataset)
else:
    fname = os.path.join(fdir, 'auc' + '_fe:cae:' + cae_type + '_ad:' + anomaly_detection + '_data:' + dataset)

logger.info('write result into file: '+ fname)
       
if os.path.isfile(fname):
    auc = pd.read_csv(fname)
elif anomaly_detection == 'ocsvm':
    auc = pd.DataFrame(columns = ['class', 'auc', 'featuresExtractTime', 'anomalyDetectTime', 'randomSeed'])
else:
    auc = pd.DataFrame(columns = ['class', 'm', 'c', 'k', 'auc', 'featuresExtractTime', 'anomalyDetectTime', 'randomSeed'])
  
if anomaly_detection == 'ocsvm':
    auc = auc.append({'class':normal_class, 'auc':auc_, 'featuresExtractTime':featuresExtractTime, 'anomalyDetectTime':anomalyDetectTime,
                      'randomSeed':random_seed}, ignore_index=True)
elif anomaly_detection == 'nnd':
    auc = auc.append({'class':normal_class, 'm':0, 'c':0, 'k':k, 'auc':auc_, 'featuresExtractTime':featuresExtractTime, 'anomalyDetectTime':anomalyDetectTime,
                      'randomSeed':random_seed}, ignore_index=True)
elif anomaly_detection == 'qnnd':
    auc = auc.append({'class':normal_class, 'm':m, 'c':c, 'k':k, 'auc':auc_, 'featuresExtractTime':featuresExtractTime, 'anomalyDetectTime':anomalyDetectTime,
                      'randomSeed':random_seed}, ignore_index=True)

auc.to_csv(fname,index=False)

logger.info('END')
