# anomalyDetection

This is the code used to produce the main results in the paper:

Sarafijanovic-Djukic N, Davis J. Fast Distance-Based Anomaly Detection in Images Using an Inception-Like Autoencoder. InInternational Conference on Discovery Science 2019 Oct 28 (pp. 493-508). Springer, Cham.

@inproceedings{sarafijanovic2019fast,\
title={Fast Distance-Based Anomaly Detection in Images Using an Inception-Like Autoencoder},\
author={Sarafijanovic-Djukic, Natasa and Davis, Jesse},\
booktitle={International Conference on Discovery Science},\
pages={493--508},\
year={2019},\
organization={Springer}\
}\

Usage example:

python main.py --dataset cifar100 --normal_class 11 --features_extractor cae --cae_type inception --anomaly_detection qnnd --output_dir results2 --random_seed 15 --param_k 1 --param_m 2 --param_c 3

Parameters for main.py:

"-d" or "--dataset"; meaning: type of datasets; choices: ['mnist', 'fmnist', 'cifar10', 'cifar100']; required parameter \
"-nc" or "--normal_class"; meaning: class for normal images; choices: if dataset is 'cifar100' it is in range 0..19, otherwise it is in range 0..9"; required parameter \
"-fe" or "--features_extractor"; meaning: if we use raw images or low-dimensional representation obtained by convolutional auto-encoder (CAE); choices: ['raw','cae']; not required - defau=cae' \
"-cae" or "--cae_type",  meaning: the type of CAE used; choices=['baseline','inception']; not required - default='inception'\
"-ad" or "--anomaly_detection"; meaning: anomaly detection method: ocsvm - One Class Support Vector Machines, nnd - exact distance-based, qnnd - approximated distance-based with product quantization"; choices=['ocsvm','nnd', 'qnnd'], not required - default='qnnd' \
"-odir" or "--output_dir"; meaning:directory to store the output results; required parameter \
"-rs" or "--random_seed"; meaning: random seed, not required - default=0 \
"-k" or "--param_k"; meaning: nnd and qnnd parameter k"; not required - default=1 \
"-m" or "--param_m"; meaning: qnnd parameter m; choices = [1,2,4,8,16,32,64,128]; not required - default=1 \
"-c" or "--param_c"; meaning: qnnd parameter c"; choices = [1,2,3,4,5,6,7,8]; not required - default=1 \