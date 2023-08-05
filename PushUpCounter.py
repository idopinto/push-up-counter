# import cv2
import math

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

file_name = 'pushups2.mp4'
cap = cv2.VideoCapture(file_name)

# Meta.
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Video writer.
video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)
pu_counter = 0
form = "UP"


def normalized_to_pixel_coords(normalized_coord, shape):
    x_px = min(math.floor(normalized_coord.x * shape[1]), shape[1] - 1)
    y_px = min(math.floor(normalized_coord.y * shape[0]), shape[0] - 1)
    return x_px, y_px


def main():
    global pu_counter
    global form
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            shape = frame.shape[:2]
            results = pose.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Use lm and lmPose as representative of the following methods.
            lm = results.pose_landmarks
            if lm:
                lmPose = mp_pose.PoseLandmark
                # Left shou lder.
                l_shldr_x, l_shldr_y = normalized_to_pixel_coords(lm.landmark[lmPose.LEFT_SHOULDER], shape)
                r_shldr_x, r_shldr_y = normalized_to_pixel_coords(lm.landmark[lmPose.RIGHT_SHOULDER], shape)
                l_elbow_x, l_elbow_y = normalized_to_pixel_coords(lm.landmark[lmPose.LEFT_ELBOW], shape)
                r_elbow_x, r_elbow_y = normalized_to_pixel_coords(lm.landmark[lmPose.RIGHT_ELBOW], shape)

                if form == "UP" and l_shldr_x <= l_elbow_x and l_shldr_y > l_elbow_y and r_elbow_x <= r_shldr_x and r_shldr_y > r_elbow_y:
                    pu_counter += 1
                    form = "DOWN"

                if form == "DOWN" and l_shldr_x <= l_elbow_x and l_shldr_y < l_elbow_y and r_elbow_x <= r_shldr_x and r_shldr_y < r_elbow_y:
                    form = "UP"
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # put push up counter text
            cv2.putText(frame, f"{str(pu_counter)}", org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=(0, 0, 255), thickness=5)
            cv2.imshow('Image', frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
