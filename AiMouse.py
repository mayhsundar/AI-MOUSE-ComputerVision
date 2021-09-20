import cv2
import numpy as np
import handTrackerMod as htm
import pyautogui as mouse
# url = "{IP}/video"
# capture = cv2.VideoCapture(1)
# capture.open(url)
camH, camW = 480, 680
capture = cv2.VideoCapture(0)
capture.set(3, camW)
capture.set(4, camH)
# capture.set(cv2.CAP_PROP_BRIGHTNESS, 0.1)
# time.sleep(2)

# capture.set(15, 1.0)
myW, myH = mouse.size()

# for mouse movement
smoothenValue = 6

pmX, pmY = 0, 0
cmX, xmY = 0, 0
frameR = 100


detector = htm.HandTracker(max_hands=1, det_conf=0.7)

while True:
    success, img = capture.read()
    cv2.rectangle(img, (frameR, frameR-50), (camW-frameR, camH-frameR-50), (255, 0, 255), 3)
    img = cv2.flip(img, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = detector.find_hands(img)
    ordinates = detector.findcordinates(img, draw=False)
    if len(ordinates) > 0:

        fingersUp = detector.fingers_up()
        distance, img, points = detector.findDistance(8, 12, img, r=5, draw=False)
        x1, y1 = points[:2]
        xc, yc = points[4:6]
        xp = np.interp(x1, (frameR, camW-frameR), (0, myW))
        yp = np.interp(y1, (frameR-50, camH-frameR-50), (0, myH))
        # mouse.FAILSAFE = False

        if fingersUp[1] and fingersUp[2] and sum(fingersUp) == 2:
            if distance < 30:
                mouse.click()
                # time.sleep(0.5)
        elif fingersUp[1] and sum(fingersUp) == 1:
            cmX = pmX + (xp-pmX)/smoothenValue
            cmY = pmY + (yp-pmY)/smoothenValue
            mouse.moveTo(cmX, cmY)
            pmX, pmY = cmX, cmY
            cv2.circle(img, (x1, y1), 9, (0, 255, 0), cv2.FILLED)

    cv2.imshow("IMG:", img)

    if cv2.waitKey(1) == ord('q'):
        break
