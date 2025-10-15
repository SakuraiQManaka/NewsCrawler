import requests
from bs4 import BeautifulSoup
import json
import re
import time
from datetime import datetime
from collections import defaultdict
import hashlib
import openai
from typing import List, Dict, Any
import sqlite3
import os

class NewsProcessor:
    """新闻处理器"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
    
    def extract_keywords(self, text, max_keywords=10):
        """使用AI提取关键词"""
        try:
            prompt = f"""
            请从以下新闻内容中提取不超过{max_keywords}个最重要的关键词。
            要求：关键词要具有代表性，能够概括文章主要内容，用逗号分隔。
            
            新闻内容：
            {text[:2000]}  # 限制文本长度
            
            关键词：
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的新闻分析师。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            return keywords[:max_keywords]
            
        except Exception as e:
            print(f"关键词提取失败: {e}")
            # 备用方案：简单的文本处理提取关键词
            return self._fallback_keyword_extraction(text, max_keywords)
    
    def _fallback_keyword_extraction(self, text, max_keywords):
        """备用关键词提取方案"""
        words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        word_freq = defaultdict(int)
        for word in words:
            if len(word) >= 2:  # 至少两个字符
                word_freq[word] += 1
        
        # 返回频率最高的关键词
        return [word for word, freq in sorted(word_freq.items(), 
                                            key=lambda x: x[1], reverse=True)[:max_keywords]]
    
    def generate_summary(self, articles):
        """为同一事件的多个文章生成总结"""
        if not articles:
            return "", ""
        
        combined_content = "\n\n".join([
            f"来源: {article['source']}\n标题: {article['title']}\n内容: {article['content'][:500]}"
            for article in articles
        ])
        
        try:
            # 生成一句话新闻
            one_sentence_prompt = f"""
            基于以下多个来源的新闻报道，生成一句简洁的一句话新闻（不超过50字）：
            
            {combined_content[:3000]}
            
            一句话新闻：
            """
            
            one_sentence_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的新闻编辑。"},
                    {"role": "user", "content": one_sentence_prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            one_sentence = one_sentence_response.choices[0].message.content.strip()
            
            # 生成详细总结
            detailed_prompt = f"""
            基于以下多个来源的新闻报道，生成一个100字以内的总结：
            
            {combined_content[:3000]}
            
            总结：
            """
            
            detailed_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的新闻编辑。"},
                    {"role": "user", "content": detailed_prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            detailed_summary = detailed_response.choices[0].message.content.strip()
            
            return one_sentence, detailed_summary
            
        except Exception as e:
            print(f"总结生成失败: {e}")
            # 备用总结方案
            titles = [article['title'] for article in articles]
            fallback_sentence = titles[0] if titles else "暂无信息"
            fallback_summary = f"该事件被{len(articles)}个来源报道，主要内容涉及{articles[0]['title']}"
            return fallback_sentence, fallback_summary
