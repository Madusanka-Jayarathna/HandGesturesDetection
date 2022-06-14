import cv2
import mediapipe as mp
import os

fingerTips = [8,12,16,20]   #finger tips from openCV
overLay = []

imageList = os.listdir("FingerImages")  #images from the source_folder
imageList.sort()                        #sorter out images

for imgNo in imageList:
    image = cv2.imread(f'FingerImages/{imgNo}')
    overLay.append(image)

#open video camera and set the width and height
stream = cv2.VideoCapture(0)
stream.set(3,1280)
stream.set(4,720)


hands = mp.solutions.hands
hand = hands.Hands(max_num_hands=1)
handDraw = mp.solutions.drawing_utils


while True:
    stat, img = stream.read()              #read image from stream
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #convert BGR color image to RGB
    results = hand.process(imgRGB)

    if results.multi_hand_landmarks:    #if hand detected
        lmList = []
        myHand = results.multi_hand_landmarks[0]
        for lmks in results.multi_hand_landmarks:
            handDraw.draw_landmarks(img, lmks, hands.HAND_CONNECTIONS,
                                    handDraw.DrawingSpec(color=(102,0,102),thickness=3,circle_radius=3),
                                    handDraw.DrawingSpec(color=(0,255,255),thickness=2,circle_radius=2))

        for no, lmk in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lmk.x * w), int(lmk.y * h)
            lmList.append([no, cx, cy])

        sequence = []
        if len(lmList) != 0:
            if lmList[4][1] > lmList[3][1]:
                sequence.append(1)
            else:
                sequence.append(0)

            for tip in fingerTips:
                if lmList[tip][2] < lmList[tip-2][2]:
                    sequence.append(1)
                else:
                    sequence.append(0)

        outSequence = "".join(map(str,sequence))
        value = sequence.count(1)
        show = int(outSequence,2)

        img[0:360, 920:1280] = overLay[show]
        cv2.rectangle(img,(920,0),(1280,360),(0,255,255),5)
        cv2.rectangle(img,(1010,400),(1190,580),(0,255,255),cv2.FILLED)
        cv2.putText(img, str(value),(1050,540),cv2.FONT_HERSHEY_SIMPLEX,5,(102,0,102), 20)

    cv2.imshow("LIVE", img)
    cv2.waitKey(1)
