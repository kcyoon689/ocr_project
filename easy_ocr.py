import os
import cv2
import json
import numpy as np
from easyocr.easyocr import Reader
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse


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

def init_dir(args):
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.result_dir + '/positive', exist_ok=True)
    os.makedirs(args.result_dir + '/negative', exist_ok=True)
    os.makedirs(args.dict_result_dir, exist_ok=True)

def arg_parser():

    parser = argparse.ArgumentParser(description='EasyOCR')
    parser.add_argument('--font_path', type=str, default='workspace/fonts/HYheadline_m-yoond1004.ttf', help='Font path')
    parser.add_argument('--demo_images', type=str, default='test_img', help='Demo images directory')
    parser.add_argument('--result_dir', type=str, default='results/easyocr/images', help='Result directory')
    parser.add_argument('--dict_result_dir', type=str, default='results/easyocr/dict', help='Dict result directory')
    # parser.add_argument('--demo_images', type=str, default='workspace/demo', help='Demo images directory')
    parser.add_argument('--custom_model', type=str, default='/workspace/OCR_src/workspace/models/user_network_dir', help='Model type')
    parser.add_argument('--custom_network', type=str, default='/workspace/OCR_src/workspace/models/user_network_dir', help='Model type')
    parser.add_argument('--custom', type=bool, default=False, help='Using custom model')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = arg_parser()

    font = ImageFont.truetype(args.font_path, 20)
    files, count = get_files(args.demo_images)
    init_dir(args)
    
    positive_cnt, negative_cnt = 0, 0
    
    # stop_words = [" ", '[', ']', ':', '-', '|',"'", '}', '{', '(', ')', '!', '?', '.', ',']
    
    predict_dict = {
        'positive': [],
        'negative': [],
    }
    
    if args.custom:
        # Using custom model
        reader = Reader(['ko'], gpu=True,
                        model_storage_directory=args.custom_model,
                        user_network_directory=args.custom_network,
                        recog_network='custom')
    else:
        # Using default model
        reader = Reader(['ko'], gpu=True)
    

    for idx, file in tqdm(enumerate(files), total=count, desc="Processing Images"):
        filename = os.path.basename(file)
        name = filename.split('.')[0]
        if "-" in name:
            name = name.split("-")[0]
        
        result = reader.readtext(file)
        
        predict_str = ''
        strings = []
        
        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
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
                for string in string.split(" "):
                    strings.append(string)
            else:
                strings.append(string)
        # print(strings)
    #     # for stop_word in stop_words:
    #     #     strings = [string.replace(stop_word, "") for string in strings]
    #     # print(strings)
    #     sorted_strings = sorted(strings[:2], key=lambda x: (x.isdigit(), x))
    #     predict_str = " ".join(sorted_strings)
    #     # print(sorted_strings)

    # #     # print(predict_str)
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        if name == predict_str:
            positive_cnt += 1
            # filename, confidence, string save to dict
            predict_dict['positive'].append({
                'filename': filename,
                'confidence': confidence,
                'string': predict_str
            })
            # "result/easyocr/images/positive"
            # cv2.imwrite(os.path.join(args.result_dir, 'positive', filename), image)
        else:
            negative_cnt += 1
    #         # filename, confidence, string save to dict
            predict_dict['negative'].append({
                'filename': filename,
                'confidence': confidence,
                'string': predict_str
            })
            # "result/easyocr/images/negative"
            # cv2.imwrite(os.path.join(args.result_dir, 'negative', filename), image)
            
        cv2.imwrite(os.path.join(args.result_dir, filename), image)

    print(f"Total: {count}, Positive: {positive_cnt}, Negative: {negative_cnt}")

    predict_dict['total'] = count
    predict_dict['positive_cnt'] = positive_cnt
    predict_dict['negative_cnt'] = negative_cnt
    
    with open(os.path.join(args.dict_result_dir, 'predict.json'), 'w', encoding="UTF-8") as f:
        json.dump(predict_dict, f, ensure_ascii=False, indent=4)
    
    print("Done")
        