import numpy as np
import difflib
import time

from infer_cs import ClintInterface, Bbox

class foodEncoder:
    def __init__(self, my_car):
        self.ocrService = ClintInterface('ocr')
        self.my_car = my_car

    def orderedRec(self, dets=None):
        """
        有序识别函数：传入的dets中包含两个text_det，按照y_c从小到大将识别结果放入result列表中
        Args:
            dets: 检测结果列表，每个元素格式为 [det_cls_id, det_id, det_label, det_score, det_bbox]
        Returns:
            result: 按y_c坐标从小到大排序的OCR识别结果列表
        """
        result = []
        
        # 过滤出text_det检测结果
        text_dets = []
        for det in dets:
            det_cls_id, det_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
            if det_cls_id == 0:  # text_det的类别ID为0
                text_dets.append(det)
        
        # 如果没有检测到text_det或数量不足，返回空结果
        if len(text_dets) < 2:
            return result
        
        # 按照y_c坐标从小到大排序
        text_dets.sort(key=lambda x: x[4:][1])  # x[4:][1]是det_bbox中的y_c坐标
        
        # 获取一次完整图像
        img = self.my_car.cap_side.read()
        img_h, img_w = img.shape[:2]
        
        # 对排序后的每个text_det进行OCR识别
        for det in text_dets:
            det_cls_id, det_id, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
            x_c, y_c, w, h = det_bbox
            
            # 使用与detCheck.py相同的坐标转换方式
            x1 = img_w * (1 + x_c) / 2 - img_w * w / 4
            x2 = x1 + img_w * w / 2
            y1 = img_h * (1 + y_c) / 2 - img_h * h / 4
            y2 = y1 + img_h * h / 2
            
            # 边界检查和整数转换
            x1 = max(0, min(int(x1), img_w-1))
            y1 = max(0, min(int(y1), img_h-1))
            x2 = max(0, min(int(x2), img_w-1))
            y2 = max(0, min(int(y2), img_h-1))
            
            # 确保裁剪区域有效
            if x2 <= x1 or y2 <= y1:
                result.append("")
                continue
            
            # 裁剪文本区域
            img_txt = img[y1:y2, x1:x2]
            
            # OCR识别
            try:
                text = self.ocrService(img_txt)
                result.append(text)
            except Exception as e:
                print(f"OCR识别失败: {e}")
                result.append("")
        
        return result