import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 定义一个函数来生成高亮文本
def highlight_text(text, color="red"):
    if color == "red":
        color_code = 31
    elif color == "green":
        color_code = 32
    elif color == "yellow":
        color_code = 33
    else:
        raise ValueError(f"Unknown color: {color}")
    
    return f"\033[{color_code}m{text}\033[0m"

# 灰度图转化为彩图
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    
# 对图片进行resize
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


# 获得先验框
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

# 从文件获取类别
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

# 图像绘制
def draw_boxes(top_label, class_names, top_boxes, top_conf, image, colors):
    # 设置字体和大小
    font = ImageFont.truetype(font='./data/material/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness   = int(max((image.size[0] + image.size[1]) // np.mean([640, 640]), 1))

    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box             = top_boxes[i]
        score           = top_conf[i]

        top, left, bottom, right = box

        top     = max(1, np.floor(top).astype('int32'))
        left    = max(1, np.floor(left).astype('int32'))
        bottom  = min(image.size[1] - 1, np.floor(bottom).astype('int32'))
        right   = min(image.size[0] - 1, np.floor(right).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        # 在左上角开始画文字
        x1, y1, x2, y2 = draw.textbbox((left, top), label, font=font)
        label_size = (y2 - y1, x2 - x1)
        label = label.encode('utf-8')
        # print(label, top, left, bottom, right)
        
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c]) 
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    return image