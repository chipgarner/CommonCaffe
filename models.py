# Models that work with setup_caffe_network.py
# Using common .caffemodel files (these are large) but allows other .prototxt files if desired.
import numpy as np
import setup_caffe_network as su


class NetModels:
    @staticmethod
    def setup_places_model(path_to_models):
        # From the MIT Places paper: B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva.
        # Learning Deep Features for Scene Recognition using Places Database.
        # Advances in Neural Information Processing Systems 27 (NIPS), 2014
        # AKA the "NIPS 2014 Paper".  GoogleNet trained on places images.
        prototxt_path = path_to_models + 'googlenet_places205/deploy_places205.protxt'
        caffemodel_path = '../CommonCaffe/TrainedModels/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel'  # From the model zoo
        pixel_mean = np.float32([104.0, 116.0, 122.0])  # WRONG training set! ImageNet mean, training set dependent
        height = 224
        width = 224

        caff = su.SetupCaffe(prototxt_path, caffemodel_path, pixel_mean, height, width)
        return caff.get_network()

    @staticmethod
    def setup_googlenet_model(path_to_models):
        prototxt_path = path_to_models + 'bvlc_googlenet/deploy.prototxt'
        caffemodel_path = '../CommonCaffe/TrainedModels/bvlc_googlenet/bvlc_googlenet.caffemodel'  # this model comes with caffe
        pixel_mean = np.float32([104.0, 116.0, 122.0])  # ImageNet mean, training set dependent
        height = 224
        width = 224

        caff = su.SetupCaffe(prototxt_path, caffemodel_path, pixel_mean, height, width)
        return caff.get_network()

    @staticmethod
    def setup_bvlc_ref_model(path_to_models):
        prototxt_path = path_to_models + 'models/bvlc_reference_caffenet/deploy.prototxt'
        caffemodel_path = path_to_models + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
        pixel_mean = np.float32([104.0, 116.0, 122.0])  # WRONG mean ImageNet mean, training set dependent
        height = 227
        width = 227

        caff = su.SetupCaffe(prototxt_path, caffemodel_path, pixel_mean, height, width)
        return caff.get_network()

    @staticmethod
    def setup_flickr_model(path_to_models):
        prototxt_path = path_to_models + 'finetune_flickr_style/deploy.prototxt'
        caffemodel_path = '../CommonCaffe/TrainedModels/finetune_flickr_style/finetune_flickr_style.caffemodel'  # this model comes from bvlc flickr example
        pixel_mean = np.float32([104.0, 116.0, 122.0])  # Wrong mean
        height = 227
        width = 227

        caff = su.SetupCaffe(prototxt_path, caffemodel_path, pixel_mean, height, width)
        return caff.get_network()

    @staticmethod
    def setup_vgg(path_to_models):
        prototxt_path = path_to_models + 'VGG19/deploy.prototxt'
        caffemodel_path = '../CommonCaffe/TrainedModels/VGG19/VGG_ILSVRC_19_layers.caffemodel'  # this model comes from  ksimonyan on github
        pixel_mean = np.float32([103.9, 116.8, 123.7])  # Training dependent
        height = 224
        width = 224

        caff = su.SetupCaffe(prototxt_path, caffemodel_path, pixel_mean, height, width)
        return caff.get_network()
