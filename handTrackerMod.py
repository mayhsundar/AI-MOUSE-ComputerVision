import math

import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, mode=False, max_hands=2, det_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.detConf = det_conf
        self.trackConf = track_conf
        self.hands = mp.solutions.hands.Hands(self.mode, self.maxHands, self.detConf, self.trackConf)  # change the args also for some modification
        self.mpCalc = mp.solutions.drawing_utils  # would be used to draw
        self.tipsIds = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        self.current_hands = results.multi_hand_landmarks

        if self.current_hands and draw:
            for cHand in self.current_hands:
                self.mpCalc.draw_landmarks(img, cHand, mp.solutions.hands.HAND_CONNECTIONS)
        return img

    def findcordinates(self, img, hand_number=0, draw=True):
        self.cords = []
        if self.current_hands:
            selected_hand = self.current_hands[hand_number]
            h, w, c = img.shape
            for ix, loc in enumerate(selected_hand.landmark):
                x, y = int(w*loc.x), int(h*loc.y)
                self.cords.append([ix, x, y])
                if draw:
                    cv2.circle(img, (x, y), 7, (0, 0, 255), cv2.FILLED)
        return self.cords

    def fingers_up(self):
        fingers = []
        if len(self.cords) > 0:
            # checking if the existing one right hand or left hand
            right_hand = True
            if self.cords[self.tipsIds[4]][1] < self.cords[self.tipsIds[0]][1]:
                right_hand = False

            if right_hand:
                # print("RIGHT HAND IS ACTIVATED")
                if self.cords[self.tipsIds[0]][1] > self.cords[self.tipsIds[0]-1][1]:
                    fingers.append(0)
                else:
                    fingers.append(1)
            else:
                # print("LEFT HAND IS ACTIVATED")
                if self.cords[self.tipsIds[0]][1] < self.cords[self.tipsIds[0]-1][1]:
                    fingers.append(0)
                else:
                    fingers.append(1)

            # then 4 fingers
            for _id in range(1, 5):
                if self.cords[self.tipsIds[_id]][2] > self.cords[self.tipsIds[_id]-2][2]:
                    fingers.append(0)
                else:
                    fingers.append(1)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.cords[p1][1:]
        x2, y2 = self.cords[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

