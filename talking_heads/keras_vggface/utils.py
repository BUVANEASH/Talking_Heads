'''VGGFace models for Keras.

# Notes:
- Utility functions are modified versions of Keras functions [Keras](https://keras.io)

'''
import numpy as np
from keras import backend as K
from keras.utils.data_utils import get_file

V1_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v1.npy'
V2_LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels_v2.npy'

VGG16_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_vgg16.h5'
VGG16_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5'


RESNET50_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_resnet50.h5'
RESNET50_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5'

SENET50_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_senet50.h5'
SENET50_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_senet50.h5'


VGGFACE_DIR = 'model/vggface'

# Global tensor of imagenet mean for preprocessing symbolic inputs
_IMAGENET_MEAN = None

def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        mean = [93.5940, 104.7624, 129.1863]
    elif version == 2:
        mean = [91.4953, 103.8827, 131.0912]
    else:
        raise NotImplementedError

    if data_format == 'channels_first':
        x_temp = x_temp[:, ::-1, ...]
        x_temp[:, 0, :, :] -= mean[0]
        x_temp[:, 1, :, :] -= mean[1]
        x_temp[:, 2, :, :] -= mean[2]
    else:
        x_temp = x_temp[..., ::-1]
        x_temp[..., 0] -= mean[0]
        x_temp[..., 1] -= mean[1]
        x_temp[..., 2] -= mean[2]

    return x_temp

def preprocess_tf_input(x, data_format=None, version=1):
    global _IMAGENET_MEAN
    
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        mean = [93.5940, 104.7624, 129.1863]
    elif version == 2:
        mean = [91.4953, 103.8827, 131.0912]
    else:
        raise NotImplementedError

    std = None

    if _IMAGENET_MEAN is None:
        _IMAGENET_MEAN = K.constant(-np.array(mean))

    # Zero-center by mean pixel
    if K.dtype(x) != K.dtype(_IMAGENET_MEAN):
        x = K.bias_add(
            x, K.cast(_IMAGENET_MEAN, K.dtype(x)),
            data_format=data_format)
    else:
        x = K.bias_add(x, _IMAGENET_MEAN, data_format)
    if std is not None:
        x /= std
   
    return x

def decode_predictions(preds, top=5):
    LABELS = None
    if len(preds.shape) == 2:
        if preds.shape[1] == 2622:
            fpath = get_file('rcmalli_vggface_labels_v1.npy',
                             V1_LABELS_PATH,
                             cache_subdir=VGGFACE_DIR)
            LABELS = np.load(fpath)
        elif preds.shape[1] == 8631:
            fpath = get_file('rcmalli_vggface_labels_v2.npy',
                             V2_LABELS_PATH,
                             cache_subdir=VGGFACE_DIR)
            LABELS = np.load(fpath)
        else:
            raise ValueError('`decode_predictions` expects '
                             'a batch of predictions '
                             '(i.e. a 2D array of shape (samples, 2622)) for V1 or '
                             '(samples, 8631) for V2.'
                             'Found array with shape: ' + str(preds.shape))
    else:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 2622)) for V1 or '
                         '(samples, 8631) for V2.'
                         'Found array with shape: ' + str(preds.shape))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [[str(LABELS[i].encode('utf8')), pred[i]] for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results
