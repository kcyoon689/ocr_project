import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

image = cv2.imread('workspace/demo_images/01두1316.jpg')

ocr = PaddleOCR(use_angle_cls=True, lang='korean')
result = ocr.ocr(image, cls=True)


font_path = 'workspace/font/HYheadline_m-yoond1004.ttf'
font = ImageFont.truetype(font_path, 20)

plate_text = ''
if result:
    for line in result:
        for idx in range(len(line)):
            # print("bbox:", line[idx][0]) # <class 'list'> [[155.0, 59.0], [269.0, 78.0], [262.0, 116.0], [148.0, 98.0]]
            # print("text:", line[idx][1][0])
            # print("confidence:", line[idx][1][1])
            plate_text += line[idx][1][0]

print(plate_text)

# 이미지에 번호판 문자 출력
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(image_pil)

for line in result:
    for idx in range(len(line)):
        bbox = [tuple(point) for point in line[idx][0]]
        draw.polygon(bbox, outline=(0, 255, 0), width=2)
        draw.text((bbox[0][0], bbox[0][1] - 25), line[idx][1][0], font=font, fill=(0, 255, 0))

image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# 결과 이미지 저장
result_dir = 'workspace/result_images'
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, '01두1316_result.jpg')
cv2.imwrite(result_path, image)
