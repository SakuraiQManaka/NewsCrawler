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

class EventTracker:
    """事件追踪器"""
    
    def __init__(self):
        self.events = defaultdict(list)
        self.event_summaries = {}
    
    def calculate_similarity(self, keywords1, keywords2):
        """计算两个关键词列表的相似度"""
        set1 = set(keywords1)
        set2 = set(keywords2)
        
        if not set1 or not set2:
            return 0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0
    
    def find_similar_event(self, article_keywords, threshold=0.3):
        """寻找相似的事件"""
        for event_id, articles in self.events.items():
            # 计算与事件中所有文章的平均相似度
            total_similarity = 0
            for existing_article in articles:
                similarity = self.calculate_similarity(
                    article_keywords, existing_article['keywords']
                )
                total_similarity += similarity
            
            avg_similarity = total_similarity / len(articles) if articles else 0
            
            if avg_similarity >= threshold:
                return event_id
        
        return None
    
    def generate_event_id(self, keywords):
        """根据关键词生成事件ID"""
        key_string = ''.join(sorted(keywords))
        return hashlib.md5(key_string.encode()).hexdigest()[:8]
    
    def add_article(self, article):
        """添加文章到相应事件"""
        if not article.get('keywords'):
            return None
        
        # 寻找相似事件
        similar_event_id = self.find_similar_event(article['keywords'])
        
        if similar_event_id:
            event_id = similar_event_id
        else:
            # 创建新事件
            event_id = self.generate_event_id(article['keywords'])
        
        self.events[event_id].append(article)
        return event_id
    
    def calculate_event_weights(self):
        """计算事件权重（基于报道数量）"""
        weights = {}
        for event_id, articles in self.events.items():
            # 基础权重：报道数量
            base_weight = len(articles)
            
            # 考虑来源多样性
            sources = set(article['source'] for article in articles)
            diversity_bonus = len(sources) * 0.5
            
            weights[event_id] = base_weight + diversity_bonus
        
        return weights