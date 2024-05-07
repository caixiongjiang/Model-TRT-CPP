from PIL import Image
import numpy as np

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
    

def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh

# 从txt文件加载类别和RGB
def load_class_rgb(filename):
    colors = []
    classname = []
    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            color = tuple(map(int, data[:3]))
            name = data[3].strip()
            colors.append(color)
            classname.append(name)
    return colors, classname