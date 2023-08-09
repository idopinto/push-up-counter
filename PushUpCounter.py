# import cv2
import math
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pygame as pygame

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# file_name = 'Videos/pushups2.mp4'
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # captureDevice = camera
cap_type = int(input("0 -> Live Stream\n1 -> Video\n"))
if cap_type == 0:
    cap = cv2.VideoCapture(0)  # captureDevice = camera
else:
    file_name = input("Enter filename (with ext.)\n")
    cap = cv2.VideoCapture(f"Videos\{file_name}")  # captureDevice = camera

# Initialize Pygame mixer for audio playback
pygame.mixer.init()
# Load sound effect files
rep_sound = pygame.mixer.Sound('Assets/ping.mp3')
set_sound = pygame.mixer.Sound('Assets/nice_sound.mp3')
shit_sound = pygame.mixer.Sound('Assets/shit_sound.mp3')

# Meta.
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Video writer.
video_output = cv2.VideoWriter('out/output.mp4', fourcc, fps, frame_size)
exercise = None
pu_counter = 0
set_counter = 0
form = None
rest = 30
sets = 3
reps = 5
start_time = None


def normalized_to_pixel_coords(normalized_coord, shape):
    x_px = min(math.floor(normalized_coord.x * shape[1]), shape[1] - 1)
    y_px = min(math.floor(normalized_coord.y * shape[0]), shape[0] - 1)
    return x_px, y_px


def count_push_ups(lm, frame_shape, pu_counter, form):
    lmPose = mp_pose.PoseLandmark
    # get the x,y coordinates of the shoulders and elbows
    l_shldr_x, l_shldr_y = normalized_to_pixel_coords(lm.landmark[lmPose.LEFT_SHOULDER], frame_shape)
    r_shldr_x, r_shldr_y = normalized_to_pixel_coords(lm.landmark[lmPose.RIGHT_SHOULDER], frame_shape)
    l_elbow_x, l_elbow_y = normalized_to_pixel_coords(lm.landmark[lmPose.LEFT_ELBOW], frame_shape)
    r_elbow_x, r_elbow_y = normalized_to_pixel_coords(lm.landmark[lmPose.RIGHT_ELBOW], frame_shape)

    # check 'UP' mode to ensure we count the rep once
    # and that the shoulders are between and beneath the elbows to ensure good rep
    if form == "UP" and l_shldr_x <= l_elbow_x and l_shldr_y > l_elbow_y \
            and r_elbow_x <= r_shldr_x and r_shldr_y > r_elbow_y:
        pu_counter += 1
        form = "DOWN"
        rep_sound.play()

    # Check 'DOWN' mode and that the shoulders
    # are between and above the elbows so we can prepare for a new rep
    if form == "DOWN" and l_shldr_x <= l_elbow_x and l_shldr_y < l_elbow_y \
            and r_elbow_x <= r_shldr_x and r_shldr_y < r_elbow_y:
        form = "UP"
    return pu_counter, form


def count_pull_ups(lm, frame_shape, pu_counter, form):
    lmPose = mp_pose.PoseLandmark
    # get the x,y coordinates of the shoulders and elbows
    l_shldr_x, l_shldr_y = normalized_to_pixel_coords(lm.landmark[lmPose.LEFT_SHOULDER], frame_shape)
    r_shldr_x, r_shldr_y = normalized_to_pixel_coords(lm.landmark[lmPose.RIGHT_SHOULDER], frame_shape)
    l_elbow_x, l_elbow_y = normalized_to_pixel_coords(lm.landmark[lmPose.LEFT_ELBOW], frame_shape)
    r_elbow_x, r_elbow_y = normalized_to_pixel_coords(lm.landmark[lmPose.RIGHT_ELBOW], frame_shape)
    print(f"shoulder coords:\n"
          f"\t left: {l_shldr_x}, {l_shldr_y}\n"
          f"\t right: {r_shldr_x}, {r_shldr_y}\n")
    print(f"elbow coords:\n"
          f"\t left: {l_elbow_x}, {l_elbow_y}\n"
          f"\t right: {r_elbow_x}, {r_elbow_y}\n")
    print(form)
    # check 'UP' mode to ensure we count the rep once
    # and that the shoulders are between and beneath the elbows to ensure good rep
    if form == "DOWN" and l_shldr_y + 60 < l_elbow_y and r_shldr_y + 60 < r_elbow_y:
        pu_counter += 1
        rep_sound.play()
        form = "UP"
    # Check 'DOWN' mode and that the shoulders
    # are between and above the elbows so we can prepare for a new rep
    if form == "UP" and l_shldr_y > l_elbow_y + 60 and r_shldr_y > r_elbow_y + 60:
        form = "DOWN"
    return pu_counter, form


def main():
    # Global Variables
    global pu_counter
    global set_counter
    global form
    global rest
    global sets
    global start_time
    global exercise

    # Get input from the user ( reps X sets, rest duration)
    exercise = int(input("choose exercise:\n"
                         "\t 1 -> push-ups\n"
                         "\t 2 -> pull-ups\n"))
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
                if exercise == 1:
                    if not form:
                        form = "UP"
                    pu_counter, form = count_push_ups(lm, shape, pu_counter, form)
                if exercise == 2:
                    if not form:
                        form = "DOWN"
                    pu_counter, form = count_pull_ups(lm, shape, pu_counter, form)

                # Draw the 33 pose landmarks found
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Write informative message when the workout is finished
                if set_counter == sets:
                    # end_sound.play()
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
                    cv2.putText(frame, f"reps: {str(pu_counter)}", org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(51, 51, 161), thickness=3)
                    cv2.putText(frame, f"sets: {str(set_counter)}", org=(10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(51, 51, 161), thickness=3)

                # Update variables when we finish one set and start timer for rest time
                if pu_counter == reps:
                    pu_counter = 0
                    set_counter += 1
                    set_sound.play()
                    if set_counter < sets:
                        start_time = datetime.now()

            # If timer is ON -> Update timer on screen and turn it off when the time is up
            if start_time:
                elapsed_time = datetime.now() - start_time
                elapsed_seconds = elapsed_time.total_seconds()  # Convert timedelta to seconds
                if elapsed_seconds <= rest:
                    timer_text = f"{elapsed_seconds:.2f}"
                    cv2.putText(frame, f"rest: {timer_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                elif elapsed_seconds <= rest + 4:
                    cv2.putText(frame, f"Ready?", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (51, 51, 161), 2)
                    cv2.putText(frame, f"{int(rest + 4 - elapsed_seconds)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (51, 51, 161), 2)
                else:
                    shit_sound.play()
                    start_time = None

            # Show the current frame on the screen
            cv2.imshow('Tracker', frame)

            # Handle Keyboard input
            key = cv2.waitKey(10) & 0xFF
            # Restart - 'r'
            if key == ord('r'):
                pu_counter = 0
                set_counter = 0
            # Quit - 'q'
            if key & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
