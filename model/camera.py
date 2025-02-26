# import cv2 as cv

# class Camera:
#     def __init__(self, device=0, width=960, height=540):
#         self.cap = cv.VideoCapture(device)
#         self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
#         self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

#     def get_frame(self):
#         return self.cap.read()

#     def release(self):
#         self.cap.release()

import cv2 as cv

class Camera:
    def __init__(self, device=0, width=960, height=540):
        self.cap = cv.VideoCapture(device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

def main():
    camera = Camera()
    while True:
        ret, frame = camera.get_frame()
        if not ret:
            break

        cv.imshow('Camera', frame)
        if cv.waitKey(1) & 0xFF == 27:  # 按 ESC 键退出
            break

    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
