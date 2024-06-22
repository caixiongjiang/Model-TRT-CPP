import numpy as np

from .utils import get_iou, get_inter, xywh2xyxy


def std_output(pred):
    """
    将模型的预测结果中的最后四个元素提取出来，并计算它们的最大值，然后将这个最大值插入到原始预测结果的倒数第四个位置。
    这样做的目的可能是为了在后处理阶段为每个样本添加一个置信度得分，以提供关于预测结果的可靠性的信息。
    将(1,11,8400)处理成(8400,12)  12 = box:4 + conf:1 + cls:7
    """
    pred = np.squeeze(pred)  # 去掉batch
    pred = np.transpose(pred, (1, 0)) # 处理成（8400，11）
    pred_class = pred[..., 4:] # 将维度2的第5个开始的所有值取到
    pred_conf = np.max(pred_class, axis=-1) # 做一次取最大
    pred = np.insert(pred, 4, pred_conf, axis=-1) # 然后插入到原来索引为4的位置
    return pred  #（8400，12）




def nms(pred, conf_thres, iou_thres): 
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True] 
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    total_cls = list(set(cls))  
    output_box = []  
    for i in range(len(total_cls)):
        clss = total_cls[i] 
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]  
        box_conf_sort = np.argsort(box_conf) 
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box) 
        cls_box = np.delete(cls_box, 0, 0) 
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]  
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]  
                interArea = get_inter(max_conf_box, current_box)  
                iou = get_iou(max_conf_box, current_box, interArea)  
                if iou > iou_thres:
                    del_index.append(j)  
            cls_box = np.delete(cls_box, del_index, 0)  
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box



def cod_trf(result, pre, after):
    """
    因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上
    Args:
        result:  [x,y,w,h,conf(最大类别概率),class]
        pre:    原尺寸图像
        after:  经过letterbox处理后的图像
    Returns: 坐标变换后的结果,并将xywh转换为左上角右下角坐标x1y1x2y2
    """
    res = np.array(result)
    x, y, w, h, conf, cls = res.transpose((1, 0)) # (12, 8400)
    x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)  # 左上角点和右下角的点
    h_pre, w_pre, _ = pre.shape
    h_after, w_after, _ = after.shape
    scale = max(w_pre/w_after, h_pre/h_after)  # 缩放比例
    h_pre, w_pre = h_pre/scale, w_pre/scale  # 计算原图在等比例缩放后的尺寸
    x_move, y_move = abs(w_pre-w_after)//2, abs(h_pre-h_after)//2  # 计算平移的量
    ret_x1, ret_x2 = (x1 - x_move) * scale, (x2 - x_move) * scale
    ret_y1, ret_y2 = (y1 - y_move) * scale, (y2 - y_move) * scale
    ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
    return ret  # x1y1x2y2


def gen_detect_results(ret, origin_image, class_names):
        res = {
            "results": []
        }

        # ret: []
        for r in ret:
            x1 = max(1, r[0])
            x2 = max(origin_image.size[0] - 1, r[1])
            y1 = min(origin_image.size[1] - 1, r[2])
            y2 = min(1, r[3])

            predicted_class = class_names[int(r[5])]
            score = round(float(r[4]), 2)
            box_p = [x1, y1, x2, y2] # 左上右下 

            temp_result = {
                "class": predicted_class,
                "score": score,
                "box": box_p
            }
            res["results"].append(temp_result)
        
        return res