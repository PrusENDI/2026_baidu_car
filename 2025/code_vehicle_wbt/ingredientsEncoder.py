import numpy as np

class ingEncoder:
    def __init__(self):
        pass

    def encode(self, dets):
        x_c = {}
        y_c = {}
        det_id = {}

        for i, det in enumerate(dets):
            det_id[i], det_width, det_label, det_score, det_bbox = det[0], det[1], det[2], det[3], det[4:]
            x_c[i], y_c[i], w, h = det_bbox
        
        # 如果检测到的目标少于等于1个，无法进行分组
        if len(dets) <= 1:
            return []
        
        # 排除y_c最小的项目
        min_y_idx = max(y_c.keys(), key=lambda k: y_c[k])
        remaining_indices = [i for i in range(len(dets)) if i != min_y_idx]
        
        if len(remaining_indices) == 0:
            return []
        
        # 提取剩余项目的坐标
        remaining_x = [x_c[i] for i in remaining_indices]
        remaining_y = [y_c[i] for i in remaining_indices]
        
        # 使用K-means聚类将y坐标分为2行
        y_coords = np.array(remaining_y).reshape(-1, 1)
        unique_y = np.unique(y_coords)
        
        if len(unique_y) == 1:
            # 如果所有y坐标相同，按x坐标排序后分为两行
            sorted_indices = sorted(remaining_indices, key=lambda i: x_c[i])
            mid = len(sorted_indices) // 2
            row1_indices = sorted_indices[:mid]
            row2_indices = sorted_indices[mid:]
        else:
            # 按y坐标进行简单的分组（上下两行）
            y_threshold = np.median(remaining_y)
            row1_indices = [i for i in remaining_indices if y_c[i] <= y_threshold]
            row2_indices = [i for i in remaining_indices if y_c[i] > y_threshold]
        
        # 对每行按x坐标排序
        row1_indices.sort(key=lambda i: x_c[i])
        row2_indices.sort(key=lambda i: x_c[i])
        
        # 创建2x3的网格分组
        grid = []
        
        # 第一行，最多3个
        row1_groups = []
        for i, idx in enumerate(row1_indices[:3]):
            row1_groups.append({
                'index': idx,
                'det_id': det_id[idx],
                'position': (0, i),  # (row, col)
                'x_c': x_c[idx],
                'y_c': y_c[idx]
            })
        grid.append(row1_groups)
        
        # 第二行，最多3个
        row2_groups = []
        for i, idx in enumerate(row2_indices[:3]):
            row2_groups.append({
                'index': idx,
                'det_id': det_id[idx],
                'position': (1, i),  # (row, col)
                'x_c': x_c[idx],
                'y_c': y_c[idx]
            })
        grid.append(row2_groups)
        
        return grid
    
    def id2pos(self, id, grid):
        for row_idx, row in enumerate(grid):
            for item in row:
                if item['det_id'] == id:
                    return item['position']
        
        # 如果没有找到对应的id，返回None
        return None