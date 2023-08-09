import math
import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("Videos\pullup.mp4") #captureDevice = camera

def normalized_to_pixel_coords(normalized_coord, shape):
    x_px = min(math.floor(normalized_coord.x * shape[1]), shape[1] - 1)
    y_px = min(math.floor(normalized_coord.y * shape[0]), shape[0] - 1)
    return x_px, y_px



def main():
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            shape = frame.shape[:2]

            # Get Pose Estimation Key Points from the current frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Store the landmarks of the given key points
            lm = results.pose_landmarks
            # Draw the 33 pose landmarks found
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.imshow('Pull-Up Tracker', frame)

            # Quit - 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    cap.release()
if __name__ == '__main__':
    main()