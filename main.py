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

# 配置（请替换为你的API密钥）
API_KEY = "sk-83669bdbb51f40b49a18e3f8fb5231d8"
BASE_URL = "https://api.deepseek.com"
# 或者使用其他AI服务，如Azure OpenAI、文心一言等

class NewsAITool:
    """新闻AI工具主类"""
    
    def __init__(self, api_key):
        self.crawlers = {
            'xinhua': XinhuaCrawler(),
            'people_daily': PeopleDailyCrawler()
        }
        self.processor = NewsProcessor(api_key)
        self.event_tracker = EventTracker()
        self.database = NewsDatabase()
        
        # 新闻源URL列表（示例）
        self.news_sources = {
            'xinhua': [
                'http://www.xinhuanet.com/politics/2024-01/01/c_112....htm',
                # 更多新华网URL...
            ],
            'people_daily': [
                'http://paper.people.com.cn/rmrb/html/2024-01/01/nw.D110000renmrb_20240101_1-01.htm',
                # 更多人民日报URL...
            ]
        }
    
    def crawl_all_news(self):
        """爬取所有新闻源的文章"""
        all_articles = []
        
        for source_type, urls in self.news_sources.items():
            crawler = self.crawlers.get(source_type)
            if not crawler:
                continue
            
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
    ai_tool = NewsAITool(API_KEY)
    
    # 运行完整流程
    ai_tool.run()