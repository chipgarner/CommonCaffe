import PIL.Image
import cv2


# noinspection PyClassHasNoInit,PyPep8Naming,PyPep8Naming,PyPep8Naming
class Display:
    def showResultCV(self, vis):
        self.__showOpenCV(vis)

    @staticmethod
    def showResultPIL(vis):
        pimg = PIL.Image.fromarray(vis)
        pimg.show(),

    @staticmethod
    def __showOpenCV(image):
        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # I think this is being converted both ways ...
        cv2.imshow("test", bgr)
        cv2.waitKey(0)  # Scripting languages are weird, It will not display without this
        cv2.destroyAllWindows()
