
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

image = cv2.imread('workspace/demo_images/01두1316.jpg')

ocr = PaddleOCR(use_angle_cls=True, lang='korean')
result = ocr.ocr(image, cls=True)

plate_text = ''
if result:
    for line in result:
        for idx in range(len(line)):
            # print("bbox:", line[idx][0]) # <class 'list'> [[155.0, 59.0], [269.0, 78.0], [262.0, 116.0], [148.0, 98.0]]
            # print("text:", line[idx][1][0])
            # print("confidence:", line[idx][1][1])
            plate_text += line[idx][1][0]

print(plate_text)
