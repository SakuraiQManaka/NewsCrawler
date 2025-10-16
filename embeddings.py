import os
from openai import OpenAI
import json


class EmbeddingText:
    def __init__(self, text):
        self.text = text
        self.model = "text-embedding-v4"
        self.dimensions = 1024
        
        self.client = OpenAI(
            api_key="sk-846085efd2134c25afca40c2c55f7618",  
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def generate_embedding(self):
        completion = self.client.embeddings.create(
            model=self.model,
            input=self.text,
            dimensions=self.dimensions,
        )
        return completion.data[0].embedding
