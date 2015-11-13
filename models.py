# Models that work with setup_caffe_network.py
import numpy as np
import setup_caffe_network as su


class NetModels:
    @staticmethod
    def setup_places_model():
        # From the MIT Places paper: B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva.
        # Learning Deep Features for Scene Recognition using Places Database.
        # Advances in Neural Information Processing Systems 27 (NIPS), 2014
        # AKA the "NIPS 2014 Paper".  GoogleNet trained on places images.
        prototxt_path = 'models/googlenet_places205/deploy_places205.protxt'
        caffemodel_path = 'models/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel'  # From the model zoo
        pixel_mean = np.float32([104.0, 116.0, 122.0])  # WRONG training set! ImageNet mean, training set dependent
        height = 224
        width = 224

        caff = su.SetupCaffe(prototxt_path, caffemodel_path, pixel_mean, height, width)
        return caff.get_network()

    @staticmethod
    def setup_googlenet_model():
        prototxt_path = 'models/bvlc_googlenet/deploy.prototxt'
        caffemodel_path = 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'  # this model comes with caffe
        pixel_mean = np.float32([104.0, 116.0, 122.0])  # ImageNet mean, training set dependent
        height = 224
        width = 224

        caff = su.SetupCaffe(prototxt_path, caffemodel_path, pixel_mean, height, width)
        return caff.get_network()
