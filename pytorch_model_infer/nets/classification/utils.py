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

# 从文件获取类别
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)