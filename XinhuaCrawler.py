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

class XinhuaCrawler(NewsCrawler):
    """新华网爬虫"""
    
    def crawl_news(self, url):
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 新华网特定的解析逻辑
            title = soup.find('h1').get_text().strip() if soup.find('h1') else ''
            
            # 获取发布时间
            time_element = soup.find('span', class_=re.compile(r'time|date'))
            publish_time = time_element.get_text().strip() if time_element else ''
            
            # 获取作者
            author_element = soup.find('span', class_=re.compile(r'author|source'))
            author = author_element.get_text().strip() if author_element else ''
            
            # 获取正文内容
            content_elements = soup.find_all('p', class_=re.compile(r'content|text'))
            content = '\n'.join([p.get_text().strip() for p in content_elements])
            
            return {
                'title': title,
                'content': content,
                'publish_time': publish_time,
                'author': author,
                'source': '新华网',
                'url': url
            }
        except Exception as e:
            print(f"爬取新华网新闻失败: {e}")
            return None
