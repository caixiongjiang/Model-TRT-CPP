import yaml
import os
import numpy as np

def config_read():
    config_path = "./config/config.yaml"
    fo = open(config_path, 'r', encoding='utf-8')
    res = yaml.load(fo, Loader=yaml.FullLoader) 

    return res


def return_config():
    res = config_read()
    return res["LOG"], res["MODEL"], res["SERVER"]

# 创建文件夹
def create_dir(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"文件夹 '{folder_path}' 创建成功。")
        except OSError as e:
            print(f"创建文件夹 '{folder_path}' 失败: {e}")
    else:
        print(f"文件夹 '{folder_path}' 已存在，无需创建。")


# 从文件获取类别
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def gen_predict_results(top_label, top_conf, top_boxes, class_names, origin_image):
        res = {
            "results": []
        }

        for i, c in list(enumerate(top_label)):
            predicted_class = class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = int(max(1, np.floor(top)))
            left    = int(max(1, np.floor(left)))
            bottom  = int(min(origin_image.size[1] - 1, np.floor(bottom)))
            right   = int(min(origin_image.size[0] - 1, np.floor(right)))

            box_p = [left, top, right, bottom] # 左上右下 

            temp_result = {
                "class": predicted_class,
                "score": round(score.astype('float'), 2),
                "box": box_p
            }
            res["results"].append(temp_result)
        
        return res