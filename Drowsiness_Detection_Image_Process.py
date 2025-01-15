from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import pyttsx3
from pygame import mixer
import time
import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

mixer.init()
mixer.music.load("music.wav")

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    engine = pyttsx3.init()

    while alarm_status:
        print('call')
        engine.say(msg)
        engine.runAndWait()

    if alarm_status2:
        print('call')
        saying = True
        engine.say(msg)
        engine.runAndWait()
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def calculate_kss(ear, mar):
    if ear < 0.2 and mar > 0.5:
        return 9
    elif ear < 0.2 and mar > 0.4:
        return 8
    elif ear < 0.2:
        return 7
    elif ear < 0.25 and mar > 0.4:
        return 6
    else:
        return 5

def display_kss_level(frame, kss_level):
    if kss_level >= 6:
        warning_text = "Warning: Drowsiness Detected! KSS Level: {}".format(kss_level)
        cv2.putText(frame, warning_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    kss_text = "KSS Level: {}".format(kss_level)
    cv2.putText(frame, kss_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

# Monitor the process resources
process = psutil.Process()
start_time = time.time()
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = cv2.VideoCapture(0)

# Initialize FPS calculation
fps_start_time = time.time()
fps = 0
frame_count = 0

# Initialize Matplotlib figure and axes
fig, ax = plt.subplots()
bars = ax.bar(['EAR', 'Yawn'], [0, 0], color=['blue', 'red'])
ax.set_ylim(0, 100)  
canvas = FigureCanvas(fig)

def update_bar_chart(ear, yawn):
    bars[0].set_height(ear * 100)  
    bars[1].set_height(yawn)
    canvas.draw()

def plot_to_image():
    canvas.draw()
    buf = canvas.buffer_rgba()
    ncols, nrows = buf.shape[1], buf.shape[0]
    return np.frombuffer(buf, np.uint8).reshape(nrows, ncols, 4)

while True:
    ret, frame = vs.read()
    if not ret:
        break
    original_frame = frame.copy() 
    frame = imutils.resize(frame, width=450)
    original_frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted_frame = cv2.bitwise_not(gray)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # Draw contours on the inverted frame in blue
        cv2.drawContours(inverted_frame, [leftEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(inverted_frame, [rightEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(inverted_frame, [lip], -1, (255, 0, 0), 1)

        # Draw facial landmark points
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.circle(inverted_frame, (x, y), 1, (0, 0, 255), -1)

        kss_level = calculate_kss(ear, distance)
        display_kss_level(frame, kss_level)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.deamon = True
                    t.start()
                    mixer.music.play()

                cv2.putText(frame, "--*--*--*--*CHU Y!--*--*--*--*", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False

        if (distance > YAWN_THRESH):
                cv2.putText(frame, "--*--*--*--*CHU Y!--*--*--*--*", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if alarm_status2 == False and saying == False:
                    alarm_status2 = True
                    t = Thread(target=alarm, args=('take some fresh air sir',))
                    t.deamon = True
                    t.start()
                    mixer.music.play()
        else:
            alarm_status2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, frame.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # FPS calculation
    frame_count += 1
    if (time.time() - fps_start_time) >= 1:
        fps = frame_count
        frame_count = 0
        fps_start_time = time.time()

    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Update the bar chart with current EAR and Yawn values
    #update_bar_chart(ear, distance)

    # Convert the plot to an image
    #plot_img = plot_to_image()
    #plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

    # Resize the plot image to fit the specified rectangular area
    #plot_height, plot_width = plot_img.shape[:2]
    #desired_width = 180  # Set the desired width for the chart
    #desired_height = 290  # Set the desired height for the chart
    #plot_img = cv2.resize(plot_img, (desired_width, desired_height))

    # Overlay the plot image on the frame
    #x_offset = 10
    #y_offset = frame.shape[0] - desired_height - 10
    #frame[y_offset:y_offset + plot_img.shape[0], x_offset:x_offset + plot_img.shape[1]] = plot_img

    cv2.imshow("Frame", frame)
    cv2.imshow("Original Frame", original_frame)
    cv2.imshow("Inverted Frame", inverted_frame) 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()
