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

class PeopleDailyCrawler(NewsCrawler):
    """人民日报爬虫"""
    
    def crawl_news(self, url):
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 人民日报特定的解析逻辑
            title = soup.find('h1').get_text().strip() if soup.find('h1') else ''
            
            time_element = soup.find('div', class_=re.compile(r'time'))
            publish_time = time_element.get_text().strip() if time_element else ''
            
            author_element = soup.find('span', class_=re.compile(r'author'))
            author = author_element.get_text().strip() if author_element else ''
            
            content_elements = soup.select('div.content p, div.text p')
            content = '\n'.join([p.get_text().strip() for p in content_elements])
            
            return {
                'title': title,
                'content': content,
                'publish_time': publish_time,
                'author': author,
                'source': '人民日报',
                'url': url
            }
        except Exception as e:
            print(f"爬取人民日报新闻失败: {e}")
            return None