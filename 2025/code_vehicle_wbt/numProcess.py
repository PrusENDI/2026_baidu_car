class NumProcess:
    def __init__(self):
        pass

    def extract_first_integer(self, response_dict):
        import re
        
        if not response_dict or 'result' not in response_dict:
            return None
        
        result_text = str(response_dict['result'])
        
        match = re.search(r'\b([1-9]|1[0-2])\b', result_text)
        if match:
            return int(match.group(1))
        
        match = re.search(r'\d+', result_text)
        if match:
            num = int(match.group())
            if 1 <= num <= 12:
                return num
        
        return None
    
    def extract_first_number(self, response_dict, min_val=0.0, max_val=100.0):
        import re
        
        if not response_dict or 'result' not in response_dict:
            return None
        
        result_text = str(response_dict['result'])
        
        # 使用更精确的正则表达式匹配数字
        # (?<!\d) 负向前瞻，确保前面不是数字
        # (?!\d) 负向后瞻，确保后面不是数字（避免匹配数字的一部分）
        match = re.search(r'(?<!\d)(\d+(?:\.\d+)?)(?!\d)', result_text)
        if match:
            try:
                num = float(match.group(1))
                if min_val <= num <= max_val:
                    return num
            except ValueError:
                pass
        
        return None
    
if __name__ == '__main__':
    num_process = NumProcess()
    sample_response = {'result': '我进行了大量思考，我认为最符合的是食材2西红柿。'}
    extracted_num_int = num_process.extract_first_integer(sample_response)

    sample_response = {'result': '这个人的BMI是22.5，接下来我将解释为什么。'}
    extracted_num_float = num_process.extract_first_number(sample_response)

    print(extracted_num_int)
    print(extracted_num_float)