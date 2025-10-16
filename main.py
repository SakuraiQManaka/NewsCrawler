# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 22:32:58 2025

@author: Troy
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
from datetime import datetime
from collections import defaultdict
import hashlib
from openai import OpenAI
from typing import List, Dict, Any
import sqlite3
import os
import numpy as np
from sklearn.cluster import DBSCAN
import pickle

# 配置（请替换为你的API密钥）
API_KEY = "sk-83669bdbb51f40b49a18e3f8fb5231d8"
BASE_URL = "https://api.deepseek.com"
# 或者使用其他AI服务，如Azure OpenAI、文心一言等

#默认新闻网站列表
sites = [
    "xinhua",
    ]

catalogue = {
    "xinhua": {
        "politics" : "https://www.xinhuanet.com/politics/szlb/",
        "fortune" : "https://www.xinhuanet.com/fortune/yx/",
        "world": "https://www.xinhuanet.com/worldpro/gjxw/",
        },
    "people_daily": {
        "layout": lambda month, day: f"https://paper.people.com.cn/rmrb/pc/layout/{month}/{day}/node_01.html",
        "content": lambda month, day, page: f"https://paper.people.com.cn/rmrb/pc/content/{month}/{day}/content_{page}.html",
        },
    }

def rqds(prompt, content, temperature):
    client = OpenAI(
        api_key = os.environ.get('DEEPSEEK_API_KEY'),,
        base_url = "https://api.deepseek.com/v1")
    
    response = client.chat.completions.create(
        model = "deepseek-chat",
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt}
        ],
        stream = False,
        temperature = temperature
    )
    return response

class NewsCrawler:
    """新闻爬虫基类"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36'
        })
    
    def generateurl(self, catalogue):
        """生成urls - 需要子类实现"""
        raise NotImplementedError
    
    def crawl_news(self, url):
        """爬取新闻文章 - 需要子类实现"""
        raise NotImplementedError

class XinhuaCrawler(NewsCrawler):
    """新华网爬虫"""
    
    def generateurl(self, catalogue):
        urls = []
        try:
            for key,value in catalogue.items():
                _ord = 0
                response = self.session.get(value, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")
                content_list = soup.find('div', id='content-list')
                if content_list:
                    tit_divs = content_list.find_all('div', class_='tit')
                    for tit_div in tit_divs:
                        a_tag = tit_div.find('a')
                        if a_tag and a_tag.has_attr('href'):
                            href = a_tag['href']
                            url = "https://www.xinhuanet.com"+href if "www" not in href else href
                            urls.append(url)
                            _ord += 1
                print(f"成功生成新华网 {key} 新闻列表，共{_ord}个新闻")      
            print(f"新华网全部新闻url生成完毕，总计 {len(urls)} 条")
            return urls
        except Exception as e:
            print(f"生成新华网新闻地址失败：{e}")
            return None
    
    def crawl_news(self, url):
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 新华网特定的解析逻辑
            title = soup.find("span", class_="title").get_text().strip() if soup.find("span", class_="title") else ''
            
            # 获取发布时间
            time_element = soup.find("div", class_="header-time")
            publish_time = "/".join(i.get_text() for i in time_element.find_all("em")) if time_element else ''
            
            # 获取正文内容
            content = soup.find("span", id="detailContent").get_text()
            cleaned_text = content.replace('\xa0', ' ').replace('\u3000', ' ').replace('\u2002', ' ').replace('\u2003', ' ')
            
            return {
                'title': title,
                'content': cleaned_text,
                'publish_time': publish_time,
                'source': 'xinhua',
                'url': url
            }
        except Exception as e:
            print(f"爬取新华网新闻失败: {e}")
            return None

class PeopleDailyCrawler(NewsCrawler):
    """
    人民日报爬虫 
    待完善……
    """
    
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
                'source': '人民日报',
                'url': url
            }
        except Exception as e:
            print(f"爬取人民日报新闻失败: {e}")
            return None

class NewsProcessor:
    """新闻处理器"""
    
    def __init__(self):
        pass
        
    def extract_keywords(self, text, max_keywords=10):
        """使用AI提取关键词"""
        try:
            prompt = f"""
            请从以下新闻内容中提取不超过{max_keywords}个最重要的关键词。
            要求：关键词要具有代表性，能够概括文章主要内容，用逗号分隔。
            
            新闻内容：
            {text}
            
            输出格式为一行新闻关键词，使用英文逗号分割，不要输出任何其他多余内容
            """
            content = "你是一个专业的新闻分析师。"
            
            response = rqds(prompt, content, 1.0)
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            print(f"成功提取关键词：{','.join(i for i in keywords)}")
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
        keywords = [word for word, freq in sorted(word_freq.items(), 
                                            key=lambda x: x[1], reverse=True)[:max_keywords]]
        print(f"使用备用提取方案，得到关键词：{','.join(i for i in keywords)}")
        return keywords
    
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
            
            {combined_content}
            
            一句话新闻：
            """
            one_sentence_content = "你是一个专业的新闻编辑。"
            one_sentence_response = rqds(one_sentence_prompt, one_sentence_content, 0.3)
            
            one_sentence = one_sentence_response.choices[0].message.content.strip()
            
            # 生成详细总结
            detailed_prompt = f"""
            基于以下多个来源的新闻报道，生成一个100字以内的总结：
            
            {combined_content}
            
            总结：
            """
            detailed_content = "你是一个专业的新闻编辑。"
            
            detailed_response = rqds(detailed_prompt, detailed_content, 0.3)
            detailed_summary = detailed_response.choices[0].message.content.strip()
            
            return one_sentence, detailed_summary
            
        except Exception as e:
            print(f"总结生成失败: {e}")
            # 备用总结方案
            titles = [article['title'] for article in articles]
            fallback_sentence = titles[0] if titles else "暂无信息"
            fallback_summary = f"该事件被{len(articles)}个来源报道，主要内容涉及{articles[0]['title']}"
            return fallback_sentence, fallback_summary

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
                (title, content, source, publish_time, url, keywords, event_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                article['title'],
                article['content'],
                article['source'],
                article['publish_time'],
                article['url'],
                json.dumps(article.get('keywords', []), ensure_ascii=False),
                article.get('event_id')
            ))
            
            conn.commit()
        except Exception as e:
            print(e)
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

class NewsAITool:
    """新闻AI工具主类"""
    
    def __init__(self):
        self.crawlers = {
            'xinhua': XinhuaCrawler(),
            'people_daily': PeopleDailyCrawler()
        }
        self.catalogue = catalogue
        self.processor = NewsProcessor()
        self.event_tracker = EventTracker()
        self.database = NewsDatabase()
        
        
        self.sites = sites
        self.sitetrans = {
            'xinhua':'新华网',
            'people_daily': '人民日报',
            }
    
    def crawl_all_news(self):
        """爬取所有新闻源的文章"""
        all_articles = []
        
        for site in self.sites:
            crawler = self.crawlers[site]
            if not crawler: 
                continue
            print(f"正在爬取 {self.sitetrans[site]} 新闻")
            print("\n")
            urls = crawler.generateurl(self.catalogue[site])
            #使用上边的先生成一个url列表，再用下边的依次爬取
            if not urls:
                continue
            #应该再加入一个检查url是否重复的功能
            for url in urls:
                print(f"爬取: {url}")
                article = crawler.crawl_news(url)
                if article:
                    # 提取关键词
                    article['keywords'] = self.processor.extract_keywords(
                        f"{article['title']} {article['content']}"
                    )
                    all_articles.append(article)
                    
                    # 添加到事件追踪
                    event_id = self.event_tracker.add_article(article)
                    article['event_id'] = event_id
                    
                    # 保存到数据库
                    self.database.save_article(article)
                
                time.sleep(1)  # 礼貌爬取
        
        return all_articles
    
    def process_events(self):
        """处理所有事件并生成总结"""
        # 计算事件权重
        weights = self.event_tracker.calculate_event_weights()
        
        # 为每个事件生成总结
        for event_id, articles in self.event_tracker.events.items():
            one_sentence, summary = self.processor.generate_summary(articles)
            
            self.event_tracker.event_summaries[event_id] = {
                'one_sentence': one_sentence,
                'summary': summary,
                'weight': weights[event_id],
                'article_count': len(articles)
            }
            
            # 保存到数据库
            self.database.save_event(
                event_id, one_sentence, summary, 
                weights[event_id], len(articles)
            )
    
    def get_important_events(self, top_n=5):
        """获取最重要的事件"""
        events_with_weights = [
            (event_id, info['weight'], info) 
            for event_id, info in self.event_tracker.event_summaries.items()
        ]
        
        # 按权重排序
        sorted_events = sorted(events_with_weights, key=lambda x: x[1], reverse=True)
        
        return sorted_events[:top_n]
    
    def run(self):
        """运行完整的新闻处理流程"""
        print("开始爬取新闻...")
        articles = self.crawl_all_news()
        print(f"成功爬取 {len(articles)} 篇文章")
        return True
    
        print("处理事件和生成总结...")
        self.process_events()
        
        print("获取重要事件...")
        important_events = self.get_important_events()
        
        # 输出结果
        print("\n=== 重要事件分析 ===")
        for i, (event_id, weight, info) in enumerate(important_events, 1):
            print(f"\n{i}. 事件 {event_id} (权重: {weight:.2f})")
            print(f"   一句话新闻: {info['one_sentence']}")
            print(f"   详细总结: {info['summary']}")
            print(f"   相关文章: {info['article_count']} 篇")

# 使用示例
if __name__ == "__main__":
    # 初始化工具（请替换为你的API密钥）
    ai_tool = NewsAITool()
    
    # 运行完整流程
    ai_tool.run()