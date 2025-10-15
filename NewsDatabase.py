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

class NewsDatabase:
    """新闻数据库"""
    
    def __init__(self, db_path='news.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建文章表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                source TEXT,
                publish_time TEXT,
                author TEXT,
                url TEXT UNIQUE,
                keywords TEXT,
                event_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建事件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                one_sentence TEXT,
                summary TEXT,
                weight REAL,
                article_count INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_article(self, article):
        """保存文章到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO articles 
                (title, content, source, publish_time, author, url, keywords, event_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article['title'],
                article['content'],
                article['source'],
                article['publish_time'],
                article['author'],
                article['url'],
                json.dumps(article.get('keywords', []), ensure_ascii=False),
                article.get('event_id')
            ))
            
            conn.commit()
        except Exception as e:
            print(f"保存文章失败: {e}")
        finally:
            conn.close()
    
    def save_event(self, event_id, one_sentence, summary, weight, article_count):
        """保存事件到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO events 
            (event_id, one_sentence, summary, weight, article_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (event_id, one_sentence, summary, weight, article_count))
        
        conn.commit()
        conn.close()