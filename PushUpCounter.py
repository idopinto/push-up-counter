# import cv2
import math
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

file_name = 'pushups2.mp4'
cap = cv2.VideoCapture(0)

# Meta.
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Video writer.
video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)
pu_counter = 0
set_counter = 0
form = "UP"
rest = 30
sets = 3
reps = 5
start_time = None


def normalized_to_pixel_coords(normalized_coord, shape):
    x_px = min(math.floor(normalized_coord.x * shape[1]), shape[1] - 1)
    y_px = min(math.floor(normalized_coord.y * shape[0]), shape[0] - 1)
    return x_px, y_px


def main():
    # Global Variables
    global pu_counter
    global set_counter
    global form
    global rest
    global sets
    global start_time

    # Get input from the user ( reps X sets, rest duration)
    reps, sets = map(int,
                     input("How many sets and reps would you like to do? write in REPSXSETS format.\n").split(sep='X'))
    rest = int(input("Enter rest duration in seconds.\n"))
    print("GOOD LUCK!")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # Read the current frame
            ret, frame = cap.read()
            shape = frame.shape[:2]

            # Get Pose Estimation Key Points from the current frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Store the landmarks of the given key points
            lm = results.pose_landmarks

            # check if there are any landmarks and that the timer is off -> start push-ups set
            if lm and start_time is None:
                lmPose = mp_pose.PoseLandmark
                # get the x,y coordinates of the shoulders and elbows
                l_shldr_x, l_shldr_y = normalized_to_pixel_coords(lm.landmark[lmPose.LEFT_SHOULDER], shape)
                r_shldr_x, r_shldr_y = normalized_to_pixel_coords(lm.landmark[lmPose.RIGHT_SHOULDER], shape)
                l_elbow_x, l_elbow_y = normalized_to_pixel_coords(lm.landmark[lmPose.LEFT_ELBOW], shape)
                r_elbow_x, r_elbow_y = normalized_to_pixel_coords(lm.landmark[lmPose.RIGHT_ELBOW], shape)

                # check 'UP' mode to ensure we count the rep once
                # and that the shoulders are between and beneath the elbows to ensure good rep
                if form == "UP" and l_shldr_x <= l_elbow_x and l_shldr_y > l_elbow_y \
                        and r_elbow_x <= r_shldr_x and r_shldr_y > r_elbow_y:
                    pu_counter += 1
                    form = "DOWN"

                # Check 'DOWN' mode and that the shoulders
                # are between and above the elbows so we can prepare for a new rep
                if form == "DOWN" and l_shldr_x <= l_elbow_x and l_shldr_y < l_elbow_y \
                        and r_elbow_x <= r_shldr_x and r_shldr_y < r_elbow_y:
                    form = "UP"

                # Draw the 33 pose landmarks found
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Write informative message when the workout is finished
                if set_counter == sets:
                    cv2.putText(frame, "Workout Completed! Well Done.", org=(50, 150),
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=1,
                                color=(51, 51, 161), thickness=3)
                    cv2.putText(frame, "(press 'r' to restart)", org=(50, 200),
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                                color=(51, 51, 161), thickness=3)
                    cv2.putText(frame, "(press 'q' to quit)", org=(50, 250),
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
                                color=(51, 51, 161), thickness=3)
                else:  # Not finished yet. show how many reps and sets we did so far.
                    cv2.putText(frame, f"reps: {str(pu_counter)}", org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(51, 51, 161), thickness=3)
                    cv2.putText(frame, f"sets: {str(set_counter)}", org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(51, 51, 161), thickness=3)

                # Update variables when we finish one set and start timer for rest time
                if pu_counter == reps:
                    pu_counter = 0
                    set_counter += 1
                    if set_counter < sets:
                        start_time = datetime.now()

            # If timer is ON -> Update timer on screen and turn it off when the time is up
            if start_time:
                elapsed_time = datetime.now() - start_time
                elapsed_seconds = elapsed_time.total_seconds()  # Convert timedelta to seconds
                if elapsed_seconds <= rest:
                    timer_text = f"{elapsed_seconds:.2f}"
                    cv2.putText(frame, f"rest: {timer_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                elif elapsed_seconds <= rest + 5:
                    cv2.putText(frame, f"Oh shit, here we go again!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (51, 51, 161), 2)
                else:
                    start_time = None

            # Show the current frame on the screen
            cv2.imshow('Push-Up Tracker', frame)

            # Handle Keyboard input
            key = cv2.waitKey(10) & 0xFF
            # Restart - 'r'
            if key == ord('r'):
                pu_counter = 0
                set_counter = 0
            # Quit - 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
