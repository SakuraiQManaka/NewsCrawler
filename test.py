import os
from openai import OpenAI
import json

input_texts = [
    '''
    值占规上工业12.3%，战略性新兴产业产值占规上工业16.9%。

　　“吉林资源禀赋独特，制造业基础较好，我们要牢记习近平总书记殷殷嘱托，咬定‘制造’不放松，抢抓机遇为制造强国建设源源不断注入吉林动力。”吉林省委书记黄强表示。
    '''
    ]

client = OpenAI(
    api_key="sk-846085efd2134c25afca40c2c55f7618",  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

completion = client.embeddings.create(
    model="text-embedding-v4",
    input=input_texts,
    dimensions=1024,
)

print(type(completion.data[0].embedding))