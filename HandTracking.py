import cv2
import numpy as np
import time
import autopy

class HandTracking:
    def __init__(self, detectionCon=0.5, trackCon=0.5):
        self.detectionCon = detectionCon
        self.trackCon = trackCon

    def detect_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center = None
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)

        return center

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = np.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def track_green_dot(self, img):
        return self.detect_color(img)

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    tracker = HandTracking()

    while True:
        success, img = cap.read()

        dot_center = tracker.track_green_dot(img)

        if dot_center is not None:
            cv2.circle(img, dot_center, 15, (255, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()