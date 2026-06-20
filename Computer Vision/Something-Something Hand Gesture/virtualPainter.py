import cv2 as cv
import numpy as np
import os
import handTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)

header = overlayList[0]
drawColor = (219, 104, 190)
brushThickness = 15
eraserThickness = 50

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0

imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. Import image
    _, img = cap.read()
    img = cv.flip(img, 1)

    # 2. Find Hand Landmarks
    img, hands = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # 3. Check which fingers are up
        fingers = detector.fingersUp(0)
        print(fingers)

    # 4. If selection mode
        if fingers[1] and fingers[2] and fingers[3] == False and fingers[4] == False and fingers[0] == False:
            xp, yp = 0, 0

            # Checking for the click
            if y1 < 125:
                if 25 < x1 < 175:
                    header = overlayList[0]
                    drawColor = (219, 104, 190)
                elif 235 < x1 < 380:
                    header = overlayList[1]
                    drawColor = (87, 210, 123)
                elif 910 < x1 < 1075:
                    header = overlayList[2]
                    drawColor = (51, 51, 224)
                elif 1115 < x1 < 1270:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv.rectangle(img, (x1, y1-20), (x2, y2+20), drawColor, cv.FILLED)

    # 5. If drawing mode
        if fingers[1] and fingers[2] == False and fingers[3] == False and fingers[4] == False and fingers[0] == False:
            cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)

    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:100, 0:1280] = header
    img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv.imshow("Video", img)
    cv.waitKey(1)
