import cv2
import numpy as np
import time
import HandTracking as ht
import autopy

### Variables Declaration
pTime = 0               # Used to calculate frame rate
width = 640             # Width of Camera
height = 480            # Height of Camera
frameR = 100            # Frame Rate
smoothening = 8         # Smoothening Factor
prev_x, prev_y = 0, 0   # Previous coordinates
curr_x, curr_y = 0, 0   # Current coordinates

cap = cv2.VideoCapture(0)   # Getting video feed from the webcam
cap.set(3, width)           # Adjusting size
cap.set(4, height)

tracker = ht.HandTracking()  # Instantiate the HandTracking class
screen_width, screen_height = autopy.screen.size()  # Getting the screen size
while True:
    success, img = cap.read()

    dot_center = tracker.detect_color(img)

    if dot_center is not None:
        cv2.circle(img, dot_center, 15, (255, 0, 255), cv2.FILLED)

        x1, y1 = dot_center
        x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
        y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

        curr_x = prev_x + (x3 - prev_x) / smoothening
        curr_y = prev_y + (y3 - prev_y) / smoothening

        autopy.mouse.move(screen_width - curr_x, curr_y)  # Moving the cursor
        cv2.circle(img, dot_center, 7, (255, 0, 255), cv2.FILLED)
        prev_x, prev_y = curr_x, curr_y
    else:
        # Do nothing when the green dot is not detected
        pass

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)