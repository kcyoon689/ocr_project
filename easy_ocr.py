import os
import cv2
import numpy as np
from easyocr.easyocr import Reader
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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

    # # Using default model
    reader = Reader(['ko'], gpu=True)

    # # Using custom model
    # reader = Reader(['ko'], gpu=True,
    #                 model_storage_directory='./workspace/user_network_dir',
    #                 user_network_directory='./workspace/user_network_dir',
    #                 recog_network='custom')
    
    
    font_path = 'workspace/font/HYheadline_m-yoond1004.ttf'
    font = ImageFont.truetype(font_path, 20)

    files, count = get_files('./workspace/demo_images')

    for idx, file in tqdm(enumerate(files), total=count, desc="Processing Images"):
        filename = os.path.basename(file)
        result = reader.readtext(file)
        
        image = cv2.imread(file)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)


        # ./easyocr/utils.py 733 lines
        # result[0]: bbox
        # result[1]: string
        # result[2]: confidence
        for (bbox, string, confidence) in result:
            # print("filename: '%s', confidence: %.4f, string: '%s'" % (filename, confidence, string))

            bbox = [tuple(point) for point in bbox]
            draw.polygon(bbox, outline=(0, 255, 0), width=2)
            draw.text((bbox[0][0], bbox[0][1] - 25), string, font=font, fill=(0, 255, 0))
        
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        result_dir = 'workspace/result/tqdm_test'
        os.makedirs(result_dir, exist_ok=True)
        
        result_path = os.path.join(result_dir, filename)
        cv2.imwrite(result_path, image)