import cv2


# noinspection PyClassHasNoInit,PyPep8Naming,PyPep8Naming,PyPep8Naming
class Display:
    def showResultCV(self, vis):
        self.__showOpenCV(vis)

    @staticmethod
    def showResultSmall(vis):
        cv2.imshow(vis)
        cv2.waitKey(0)  # It will not display without this
        cv2.destroyAllWindows()

    @staticmethod
    def __showOpenCV(image):
        # cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # I think this is being converted both ways ...
        cv2.imshow("test", bgr)
        cv2.waitKey(0)  # It will not display without this
        cv2.destroyAllWindows()
