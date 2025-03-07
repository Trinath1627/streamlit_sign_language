import streamlit as st
import numpy as np
import math
import cv2
import os
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
from PIL import Image

# Initialize hand detectors
hd = HandDetector(maxHands=1, detectionCon=0.8)
hd2 = HandDetector(maxHands=1, detectionCon=0.8)
offset = 29

# Set CUDA environment variable (optional)
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

class Application:
    def _init_(self):
        self.vs = cv2.VideoCapture(0)
        self.model = load_model('cnn8grps_rad1_model.h5')
        
        self.ct = {'blank': 0}
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = " "
        self.count = -1
        self.ten_prev_char = [""] * 10
        self.str = ""
        self.current_symbol = "C"

        for i in ascii_uppercase:
            self.ct[i] = 0
        st.write("Loaded model from disk")

    def video_loop(self):
        st.title("Sign Language to Text Conversion")
        
        # Placeholder for video feed and hand skeleton
        video_placeholder = st.empty()
        hand_placeholder = st.empty()
        char_placeholder = st.empty()
        sentence_placeholder = st.empty()
        
        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            clear_button = st.button("Clear")
        with col2:
            quit_button = st.button("Quit")

        while True:
            ok, frame = self.vs.read()
            if not ok:
                st.write("Error: Could not read from webcam.")
                break

            cv2image = cv2.flip(frame, 1)
            hands = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy = np.array(cv2image)
            cv2image_rgb = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)

            # Display video feed
            video_placeholder.image(cv2image_rgb, channels="RGB", caption="Video Feed")

            if hands:
                hand = hands[0]
                map = hand[0]
                x, y, w, h = map['bbox']
                image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                white = cv2.imread("white.jpg")
                if image.size > 0:
                    handz = hd2.findHands(image, draw=False, flipType=True)
                    if handz:
                        hand = handz[0]
                        handmap = hand[0]
                        self.pts = handmap['lmList']

                        if len(self.pts) == 21:
                            os_x = ((400 - w) // 2) - 15
                            os_y = ((400 - h) // 2) - 15
                            for t in range(0, 4):
                                cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                         (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(5, 8):
                                cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                         (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(9, 12):
                                cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                         (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(13, 16):
                                cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                         (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(17, 20):
                                cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                         (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[5][0] + os_x, self.pts[5][1] + os_y),
                                     (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[9][0] + os_x, self.pts[9][1] + os_y),
                                     (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[13][0] + os_x, self.pts[13][1] + os_y),
                                     (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y),
                                     (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y),
                                     (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)

                            for i in range(21):
                                cv2.circle(white, (self.pts[i][0] + os_x, self.pts[i][1] + os_y), 2, (0, 0, 255), 1)

                            res = white
                            self.predict(res)

                            # Display hand skeleton
                            hand_placeholder.image(res, channels="BGR", caption="Hand Skeleton")
                            char_placeholder.write(f"Character: {self.current_symbol if self.current_symbol not in [' ', 'next', 'Backspace'] else ''}")
                            sentence_placeholder.write(f"Sentence: {self.str}")

            # Handle button actions
            if clear_button:
                self.clear_fun()
                sentence_placeholder.write(f"Sentence: {self.str}")
            if quit_button:
                self.destructor()
                st.stop()

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def clear_fun(self):
        self.str = ""

    def predict(self, test_image):
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')

        confidence_threshold = 0.7
        if np.max(prob) < confidence_threshold:
            ch1 = -1
        else:
            ch1 = np.argmax(prob, axis=0)
            prob[ch1] = 0
            ch2 = np.argmax(prob, axis=0)
            prob[ch2] = 0
            ch3 = np.argmax(prob, axis=0)

            pl = [ch1, ch2]

            # Condition for [Aemnst]
            l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                 [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                 [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
            if pl in l:
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 0

            # Condition for [o][s]
            l = [[2, 2], [2, 1]]
            if pl in l:
                if (self.pts[5][0] < self.pts[4][0]):
                    ch1 = 0

            # Condition for [c0][aemnst]
            l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                    ch1 = 2

            # Condition for [c0][aemnst]
            l = [[6, 0], [6, 6], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[8], self.pts[16]) < 52:
                    ch1 = 2

            # Condition for [gh][bdfikruvw]
            l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                    ch1 = 3

            # Con for [gh][l]
            l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][0] > self.pts[0][0]:
                    ch1 = 3

            # Con for [gh][pqz]
            l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[2][1] + 15 < self.pts[16][1]:
                    ch1 = 3

            # Con for [l][x]
            l = [[6, 4], [6, 1], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[4], self.pts[11]) > 55:
                    ch1 = 4

            # Con for [l][d]
            l = [[1, 4], [1, 6], [1, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.distance(self.pts[4], self.pts[11]) > 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 4

            # Con for [l][gh]
            l = [[3, 6], [3, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[4][0] < self.pts[0][0]):
                    ch1 = 4

            # Con for [l][c0]
            l = [[2, 2], [2, 5], [2, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[1][0] < self.pts[12][0]):
                    ch1 = 4

            # Con for [gh][z]
            l = [[3, 6], [3, 5], [3, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]:
                    ch1 = 5

            # Con for [gh][pq]
            l = [[3, 2], [3, 1], [3, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                    ch1 = 5

            # Con for [l][pqz]
            l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][0] > self.pts[0][0]:
                    ch1 = 5

            # Con for [pqz][aemnst]
            l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                    ch1 = 5

            # Con for [pqz][yj]
            l = [[5, 7], [5, 2], [5, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[3][0] < self.pts[0][0]:
                    ch1 = 7

            # Con for [l][yj]
            l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] < self.pts[8][1]:
                    ch1 = 7

             # Con for [x][yj]
            l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[18][1] > self.pts[20][1]:
                    ch1 = 7

            # Condition for [x][aemnst]
            l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] > self.pts[16][0]:
                    ch1 = 6

            # Condition for [yj][x]
            l = [[7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[18][1] < self.pts[20][1]:
                    ch1 = 6

            # Condition for [c0][x]
            l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[8], self.pts[16]) > 50:
                    ch1 = 6

            # Con for [l][x]
            l = [[4, 6], [4, 2], [4, 1], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[4], self.pts[11]) < 60:
                    ch1 = 6

            # Con for [x][d]
            l = [[1, 4], [1, 6], [1, 0], [1, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                    ch1 = 6

            # Con for [b][pqz]
            l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
                 [6, 3], [6, 4], [7, 5], [7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            # Con for [f][pqz]
            l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
                 [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                        self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                        self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            # Con for [d][pqz]
            fg = 19
            l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                     self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                    ch1 = 1

            l = [[4, 1], [4, 2], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.distance(self.pts[4], self.pts[11]) < 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 1

            l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                     self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                    ch1 = 1

            l = [[6, 6], [6, 4], [6, 1], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                    ch1 = 1

            # Con for [i][pqz]
            l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                     self.pts[18][1] > self.pts[20][1])):
                    ch1 = 1

            # Con for [yj][bfdi]
            l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[4][0] < self.pts[5][0] + 15) and (
                (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                    ch1 = 7

            # Con for [uvr]
            l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                     self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                    ch1 = 1

            # Con for [w]
            fg = 13
            l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                        self.pts[0][0] + fg < self.pts[20][0]) and not (
                        self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                    ch1 = 1

            # Con for [w]
            l = [[5, 0], [5, 5], [0, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                    ch1 = 1

            # Subgroup conditions
            if ch1 == 0:
                ch1 = 'S'
                if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                    ch1 = 'A'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                    ch1 = 'T'
                if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                    ch1 = 'E'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                    ch1 = 'M'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                    ch1 = 'N'

            if ch1 == 2:
                if self.distance(self.pts[12], self.pts[4]) > 42:
                    ch1 = 'C'
                else:
                    ch1 = 'O'

            if ch1 == 3:
                if (self.distance(self.pts[8], self.pts[12])) > 72:
                    ch1 = 'G'
                else:
                    ch1 = 'H'

            if ch1 == 7:
                if self.distance(self.pts[8], self.pts[4]) > 42:
                    ch1 = 'Y'
                else:
                    ch1 = 'J'

            if ch1 == 4:
                ch1 = 'L'

            if ch1 == 6:
                ch1 = 'X'

            if ch1 == 5:
                if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                    if self.pts[8][1] < self.pts[5][1]:
                        ch1 = 'Z'
                    else:
                        ch1 = 'Q'
                else:
                    ch1 = 'P'

            if ch1 == 1:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'B'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'D'
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'F'
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'I'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'W'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                    ch1 = 'K'
                if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'U'  
                #####                                                                                                                                                                                                                                                                                                                                  
                if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                    ch1 = 'V'
                if (self.pts[8][0] > self.pts[12][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'R'

            if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = " "

            if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
                if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = "next"

            if ch1 in ['next', 'B', 'C', 'H', 'F', 'X']:
                if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                    ch1 = 'Backspace'

            if ch1 == "next" and self.prev_char != "next":
                if self.ten_prev_char[(self.count - 2) % 10] != "next":
                    if self.ten_prev_char[(self.count - 2) % 10] == "Backspace":
                        self.str = self.str[:-1] if self.str else self.str
                    else:
                        self.str += self.ten_prev_char[(self.count - 2) % 10]
                else:
                    self.str += self.ten_prev_char[self.count % 10]

            if ch1 == " " and self.prev_char != " ":
                self.str += " "

            self.prev_char = ch1 if ch1 != -1 else ""
            self.current_symbol = ch1 if ch1 != -1 else ""
            if ch1 != -1:
                self.count += 1
                self.ten_prev_char[self.count % 10] = ch1

    def destructor(self):
        self.vs.release()
        cv2.destroyAllWindows()

if __name__ == "_main_":
    st.write("Starting Application...")
    app = Application()
    app.video_loop()