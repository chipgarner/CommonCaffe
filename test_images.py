from unittest import TestCase
import images
import cv2


class TestImages(TestCase):
    def test_resize_image_tall(self):
        im = images.Images()
        img = cv2.imread('../CamDreams/Paintings/wondering.jpg')

        width = 640
        height = 480
        image = im.resize_image(height, width, img)

        h, w = image.shape[:2]

        assert h == height
        assert w == width

        # cv2.imshow('Test Image', image)
        # cv2.waitKey(0)

    def test_resize_image_wide(self):
        im = images.Images()
        img = cv2.imread('../../Pictures/1920x1080.jpg')

        width = 640
        height = 480
        image = im.resize_image(height, width, img)

        h, w = image.shape[:2]

        assert h == height
        assert w == width

        # cv2.imshow('Test Image', image)
        # cv2.waitKey(0)

    def test_resize_image_same_size(self):
        im = images.Images()
        img = cv2.imread('../CamDreams/Paintings/floating2.jpg')

        width = 640
        height = 480
        image = im.resize_image(height, width, img)

        h, w = image.shape[:2]

        assert h == height
        assert w == width

        # cv2.imshow('Test Image', image)
        # cv2.waitKey(0)


