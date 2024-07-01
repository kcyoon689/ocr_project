import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

def get_files(path):
    file_list = []

    files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    files.sort()
    abspath = os.path.abspath(path)
    for file in files:
        file_path = os.path.join(abspath, file)
        file_list.append(file_path)

    return file_list, len(file_list)

if __name__ == '__main__':
    
    font_path = 'workspace/font/HYheadline_m-yoond1004.ttf'
    font = ImageFont.truetype(font_path, 20)
    
    ocr = PaddleOCR(use_angle_cls=True, lang='korean')
    
    files, count = get_files('./workspace/demo_images')
    
    
    for idx, img_file in enumerate(files):
        plate_text = ''
        result = ocr.ocr(img_file, cls=True)

        if result:
            for line in result:
                if line:  # Check if the line is not empty
                    for idx in range(len(line)):
                        # print("bbox:", line[idx][0]) # <class 'list'> [[155.0, 59.0], [269.0, 78.0], [262.0, 116.0], [148.0, 98.0]]
                        # print("text:", line[idx][1][0])
                        # print("confidence:", line[idx][1][1])
                        plate_text += line[idx][1][0]

        # print(plate_text)

        image = cv2.imread(img_file)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        if result:
            for line in result:
                if line:  # Check if the line is not empty
                    for idx in range(len(line)):
                        bbox = [tuple(point) for point in line[idx][0]]
                        draw.polygon(bbox, outline=(0, 255, 0), width=2)
                        draw.text((bbox[0][0], bbox[0][1] - 25), line[idx][1][0], font=font, fill=(0, 255, 0))

        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        result_dir = 'workspace/result/paddle_ocr'
        img_file_name = os.path.basename(img_file)
        os.makedirs(result_dir, exist_ok=True)
        
        result_path = os.path.join(result_dir, img_file_name)
        cv2.imwrite(result_path, image)
