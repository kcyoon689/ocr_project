import os
import cv2
import json
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

    font_path = 'workspace/font/HYheadline_m-yoond1004.ttf'
    font = ImageFont.truetype(font_path, 20)
    
    custom_model = False
    result_dir = 'result/custom_easyocr'
    dict_result_dir = 'result/dict'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(dict_result_dir, exist_ok=True)
    
    

    files, count = get_files('./workspace/demo_images')
    
    positive_cnt = 0
    negative_cnt = 0
    
    predict_dict = {
        'positive': [],
        'negative': [],
    }
    
    if custom_model:
        # Using custom model
        reader = Reader(['ko'], gpu=True,
                        model_storage_directory='./workspace/user_network_dir',
                        user_network_directory='./workspace/user_network_dir',
                        recog_network='custom')
    else:
        # Using default model
        reader = Reader(['ko'], gpu=True)
    

    for idx, file in tqdm(enumerate(files), total=count, desc="Processing Images"):
        filename = os.path.basename(file)
        name = filename.split('.')[0]
        if "-" in name:
            name = name.split("-")[0]
            # print(name)
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
            
            if " " in string:
                string = string.replace(" " , "")
            if "[" in string:
                string = string.replace("[", "")
            if "]" in string:    
                string = string.replace("]", "")                
                
        if name == string:
            positive_cnt += 1
            # filename, confidence, string save to dict
            predict_dict['positive'].append({
                'filename': filename,
                'confidence': confidence,
                'string': string
            })
        else:
            negative_cnt += 1
            # filename, confidence, string save to dict
            predict_dict['negative'].append({
                'filename': filename,
                'confidence': confidence,
                'string': string
            })
            
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        result_path = os.path.join(result_dir, filename)
        cv2.imwrite(result_path, image)

    print("Total: %d" % count)
    print("Positive: %d, Negative: %d" % (positive_cnt, negative_cnt))
    predict_dict['total'] = count
    predict_dict['positive_cnt'] = positive_cnt
    predict_dict['negative_cnt'] = negative_cnt
    
    with open(os.path.join(dict_result_dir, 'predict.json'), 'w', encoding="UTF-8") as f:
        json.dump(predict_dict, f, ensure_ascii=False, indent=4)
    
    print("Done")
        