import cv2 as cv
import mediapipe as mp
import copy
import itertools
import numpy as np

class HandDetector:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=static_image_mode,
                                         max_num_hands=1,
                                         min_detection_confidence=min_detection_confidence,
                                         min_tracking_confidence=min_tracking_confidence)

    def process(self, image):
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        return results

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        # 转换为相对坐标
        base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
        for idx, point in enumerate(temp_landmark_list):
            temp_landmark_list[idx][0] -= base_x
            temp_landmark_list[idx][1] -= base_y
        # 一维化
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list)))
        temp_landmark_list = [n / max_value for n in temp_landmark_list]
        return temp_landmark_list

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]
        temp_point_history = copy.deepcopy(point_history)
        base_x, base_y = temp_point_history[0][0], temp_point_history[0][1]
        for idx, point in enumerate(temp_point_history):
            temp_point_history[idx][0] = (point[0] - base_x) / image_width
            temp_point_history[idx][1] = (point[1] - base_y) / image_height
        temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
        return temp_point_history

# 独立函数：计算包围手部的矩形框
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for landmark in landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [[x, y]], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

# 独立函数：提取每个关键点坐标
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_list = []
    for landmark in landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append([x, y])
    return landmark_list
