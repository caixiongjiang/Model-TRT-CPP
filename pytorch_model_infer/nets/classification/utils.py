


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



