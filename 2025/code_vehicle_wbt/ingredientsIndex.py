import yaml
import os

class IngIndex:
    def __init__(self, yaml_file_path='ingredients.yml'):
        self.yaml_file_path = yaml_file_path
        self.targets = self._load_targets()
        
    def _load_targets(self):
        try:
            with open(self.yaml_file_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                return data.get('targets', [])
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.yaml_file_path}")
            return []
        except yaml.YAMLError as e:
            print(f"错误: YAML 文件解析失败 - {e}")
            return []
    
    def find_target(self, identifier):
        if not self.targets:
            return None
            
        for target in self.targets:
            # target 格式: [id, value1, name, value2, ...]
            if len(target) >= 3:
                target_id = target[0]
                target_name = target[2]
                
                # 如果传入的是整数，按 ID 查找
                if isinstance(identifier, int):
                    if target_id == identifier:
                        return target
                # 如果传入的是字符串，按名称查找
                elif isinstance(identifier, str):
                    if target_name == identifier:
                        return target
        
        return None
    
if __name__ == "__main__":
    ing_index = IngIndex()
    
    result_id = ing_index.find_target(1)
    print(f"按 ID 1 查找: {result_id}")

    result_label = ing_index.find_target('tofu')
    print(f"按名称 'tofu' 查找: {result_label}")

    result_e = ing_index.find_target(999)
    print(f"按 ID 999 查找: {result_e}")