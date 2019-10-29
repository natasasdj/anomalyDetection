from keras.datasets import mnist, fashion_mnist, cifar100, cifar10
from keras.backend import cast_to_floatx
import numpy as np

### data load ####

def normalize_fn(data):
    return data/255.

def get_channels_axis():
    import keras
    idf = keras.backend.image_data_format()
    if idf == 'channels_first':
        return 1
    assert idf == 'channels_last'
    return 3

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = normalize_fn(cast_to_floatx(np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_train = np.expand_dims(X_train, axis=get_channels_axis())
    X_test = normalize_fn(cast_to_floatx(np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_test = np.expand_dims(X_test, axis=get_channels_axis())
    return (X_train, y_train), (X_test, y_test)

def load_fashion_mnist():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = normalize_fn(cast_to_floatx(np.pad(X_train, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_train = np.expand_dims(X_train, axis=get_channels_axis())
    X_test = normalize_fn(cast_to_floatx(np.pad(X_test, ((0, 0), (2, 2), (2, 2)), 'constant')))
    X_test = np.expand_dims(X_test, axis=get_channels_axis())
    return (X_train, y_train), (X_test, y_test)

def load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = normalize_fn(cast_to_floatx(X_train))
    X_test = normalize_fn(cast_to_floatx(X_test))
    return (X_train, y_train), (X_test, y_test)


def load_cifar100(label_mode='coarse'):
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)
    X_train = normalize_fn(cast_to_floatx(X_train))
    X_test = normalize_fn(cast_to_floatx(X_test))
    return (X_train, y_train), (X_test, y_test)

def load_data(dataset):
    if dataset == 'mnist':
        load_dataset_fn = load_mnist
    elif dataset == 'fmnist':
        load_dataset_fn = load_fashion_mnist
    elif dataset == 'cifar10':
        load_dataset_fn = load_cifar10
    elif dataset == 'cifar100':
        load_dataset_fn = load_cifar100
    return load_dataset_fn()

def get_normal_data(x_train, y_train, x_test, y_test, normal_class):
    x_train_normal = x_train[y_train.flatten() == normal_class] # normal train one class
    x_test_normal = x_test[y_test.flatten() == normal_class]     # normal test one class
    return x_train_normal, x_test_normal
 

