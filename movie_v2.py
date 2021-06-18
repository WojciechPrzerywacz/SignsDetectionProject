import time

import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
from playsound import playsound


def nothing(x):
    pass

def openFile():
    filepath = fd.askopenfilename(initialdir="C:\\Users\\twoto\\PycharmProjects\\SignsDetectionProject",
                                          title="Open file okay?",
                                          filetypes= (("videos","*.mp4"),
                                          ("all files","*.*")))
    return filepath


# 21.jpg big.png 20.jpeg sunny.jpeg
#filename = "C:\\Users\\twoto\\PycharmProjects\\SignsDetectionProject\\2.jpg"

cv2.namedWindow("Color based detection", cv2.WINDOW_AUTOSIZE)


end_val = 0
# Stale dla kolorow czerwony i zolty
l_h = 5
l_v = 30
u_s = 255
u_v = 255

vidcap1 = cv2.VideoCapture('test2.mp4')
count = 0
while (count<1):
    _, frame1 = vidcap1.read()
    #cv2.imshow('frame', frame1)
    k = cv2.waitKey(5) & 0xFF
    count += 1
    if k == 27:
        break

vidcap = cv2.VideoCapture('test6.mp4')
img_height = 640
img_width = 360
font = cv2.FONT_HERSHEY_COMPLEX
while (1):
    time.sleep(0.08)
    _, frame2 = vidcap.read()

    cv2.getTrackbarPos("", "Trackbar window")
    #frame2 = cv2.imread(filename, 1)
    crop_size = 89 #tu ustawiamy
    frame = frame2[0:int(img_height-(crop_size*img_height/200)), 0:img_width]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #i tu też
    l_s = 168
    u_h = 33
    size_modifier = 8

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.erode(mask, kernel)
    #stały rozmiar okna (cv2.resize(mask, (960, 540))
    cv2.imshow("Color based detection", mask)
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == 32:
        filename = openFile();
        continue
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print(len(contours))
    suspectsCounter = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        #print(x, " + ",y)
        # area depends on image size
        perfect_size = (img_height * img_width) * 0.004 * size_modifier / 100
        #
        if area > perfect_size:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 3) #grubość lini

            if len(approx) == 3:
                suspectsCounter = suspectsCounter + 1
                cv2.putText(frame, "Suspect", (x, y), font, 1, color=(255, 28, 43))
                image = frame
                out = np.zeros_like(image)  # Extract out the object and place into output image
                out[mask == 255] = image[mask == 255]
                #współrzędne gdzie maska jest biała
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))

                x, y, w, h = cv2.boundingRect(cnt)
                print("Detected:", x, " + ", y, " + ",w, " + ", h)
                crop_img = out[y:y + h, x:x + w]

                cv2.imshow('Suspect', crop_img)

    #print(suspectsCounter)
    #if suspectsCounter == 1:
        #playsound("C:\\Users\\twoto\\PycharmProjects\\SignsDetectionProject\\beep.wav")

    #cv2.namedWindow('Detection', cv2.WINDOW_AUTOSIZE)
    frame = cv2.putText(frame, "Click ESC to quit", org=(20, 20), fontFace=1, fontScale=2, color=(0, 0, 0), thickness=2)
    #img_concate=np.concatenate(((cv2.inRange((cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)), lower_red, upper_red)),frame),axis=0)
    cv2.imshow('Det', frame)


cv2.destroyAllWindows()
