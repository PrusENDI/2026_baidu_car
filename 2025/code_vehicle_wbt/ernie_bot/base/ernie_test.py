
import erniebot
erniebot.api_type = 'aistudio'
erniebot.access_token = 'e8bbac20279d0f40945607bf857bd89236771bc4'
messages=[{'role': 'user', 'content': '''“这个水果是苹果,id记录为123456, 价格是5人民币每500克”, 请根据我的描述生成一段json结果,json数据参考如下的jsonschema的描述。
      ```{ "title": "Product",
      "description": "一个商品的目录",
      "type": "object",
      "properties": {"productId": { "description": "商品唯一识别码", "type": "integer"},
        "productName": {"description": "商品名称","type": "string"},
        "price": {"description": "商品的价格","type": "number","exclusiveMinimum": 0}
      },
      "required": [ "productId", "productName", "price" ]
      }```'''}]
response= erniebot.ChatCompletion.create(model='ernie-3.5', messages=messages)
first_response = response.get_result()
print(first_response)
