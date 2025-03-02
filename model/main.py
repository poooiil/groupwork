import os
import cv2 as cv
import copy
import argparse
import socket
import time

from keypoint_classifier.keypoint_classifier import KeyPointClassifier

from camera import Camera
from hand_detector import HandDetector, calc_bounding_rect, calc_landmark_list
from data_logger import logging_csv
from draw_utils import draw_bounding_rect, draw_landmarks, draw_info_text, draw_info
from utils import CvFpsCalc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # number 0~9
        number = key - 48
    if key == 110:  # n 
        mode = 0
    if key == 107:  # k 
        mode = 1
    if key == 104:  # h 
        mode = 2
    return number, mode

def main():
    args = get_args()

    # Initialize the camera
    camera = Camera(device=args.device, width=args.width, height=args.height)

    # Initialize MediaPipe hand detector
    detector = HandDetector(
        static_image_mode=args.use_static_image_mode,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    # Compute the absolute path of the CSV file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    keypoint_label_path = os.path.join(base_dir, 'keypoint_classifier_label.csv')
    model_path = os.path.join(base_dir, 'pytorch_mlp_model.pth')
    scaler_path = os.path.join(base_dir, 'scaler.save')

    # Initialize classifier
    keypoint_classifier = KeyPointClassifier(
        label_csv=keypoint_label_path,
        model_path=model_path,
        scaler_path=scaler_path,
        load_existing_model=True
    )

    # Read labels
    keypoint_classifier_labels = keypoint_classifier.label_dict

    fps_calc = CvFpsCalc(buffer_len=10)
    use_brect = True
    mode = 0

    # Create connection to UDP socket
    server_address = ('127.0.0.1', 65432)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Variables for prediction stability
    last_pred = None
    curr_pred = None
    inter_pred = None
    pred_time = None
    stability_duration = 2

    while True:
        fps = fps_calc.get()

        key = cv.waitKey(10)
        if key == 27:  
            break
        number, mode = select_mode(key, mode)

        ret, image = camera.get_frame()
        if not ret:
            break
        image = cv.flip(image, 1)  
        debug_image = copy.deepcopy(image)

        # Detect hands
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = detector.process(image_rgb)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = detector.pre_process_landmark(landmark_list)

                if pre_processed_landmark_list:
                    logging_csv(number, mode, pre_processed_landmark_list, None)
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    label_text = keypoint_classifier_labels[hand_sign_id] if hand_sign_id >= 0 else "Unknown"
                    print(f"Detected hand_sign_id: {hand_sign_id}, Label: {label_text}")
                else:
                    hand_sign_id = -1
                    label_text = "Unknown"

                if curr_pred == label_text: 
                    # If prediction is the same, calculate stability
                    if pred_time is None:
                        pred_time = time.time()
                    elif time.time() - pred_time >= stability_duration:
                        # Check prediction is different from last prediction, or intermediate prediction exists
                        if curr_pred != last_pred or inter_pred is not None:
                            last_pred = curr_pred
                            inter_pred = None

                            # Send prediction to socket
                            try:
                                message = curr_pred.encode('utf-8')
                                print(f'Sent hand_sign_label: {message}, to {server_address}')
                                udp_socket.sendto(message, server_address)
                            except socket.timeout:
                                print('Socket timeout')
                else:
                    # Reset timer if prediction changes, and track intermediate prediction
                    pred_time = None
                    if curr_pred != last_pred:
                        inter_pred = curr_pred
                    curr_pred = label_text

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness, label_text, "")

        debug_image = draw_info(debug_image, fps, mode, number)
        cv.imshow('Hand Gesture Recognition', debug_image)

    # Close the socket
    udp_socket.close()

    camera.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
