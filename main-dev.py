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
import newscrawler as nc

# 配置（请替换为你的API密钥）
API_KEY = "sk-83669bdbb51f40b49a18e3f8fb5231d8"
BASE_URL = "https://api.deepseek.com"
# 或者使用其他AI服务，如Azure OpenAI、文心一言等

#默认新闻网站列表
sites = [
    "xinhua",
    "people"
    ]

catalogue = {
    "xinhua": {
        "politics" : "https://www.xinhuanet.com/politics/szlb/",
        "fortune" : "https://www.xinhuanet.com/fortune/yx/",
        "world": "https://www.xinhuanet.com/worldpro/gjxw/",
        },
    "people": {
        "layout": lambda month, day: f"https://paper.people.com.cn/rmrb/pc/layout/{month}/{day}/node_01.html",
        "content": lambda month, day, page: f"https://paper.people.com.cn/rmrb/pc/content/{month}/{day}/content_{page}.html",
        },
    }

def rqds(prompt, content, temperature):
    client = OpenAI(
        api_key = os.environ.get('DEEPSEEK_API_KEY'),
        base_url = "https://api.deepseek.com/v1")
    
    response = client.chat.completions.create(
        model = "deepseek-chat",
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt}
        ],
        stream = False,
        temperature = temperature)
    return response
  
def gtembd(text):
    client = OpenAI(
        api_key = os.environ.get('DASHSCOPE_API_KEY'),
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client.embeddings.create(
        model = "text-embedding-v4",
        input = text,
        dimensions = 1024)
    return completion.data[0].embedding

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
                embedding BLOB,  -- 存储1024维向量
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
                embedding BLOB,  -- 存储事件平均向量
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def gettodayurl(self):
        todayurls = []
        '''待完善'''
        return todayurls

    def save_article(self, article):
        """保存文章到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 处理embedding字段
            embedding_blob = None
            if 'embedding' in article:
                embedding_blob = pickle.dumps(article['embedding'])
            
            cursor.execute('''
                INSERT OR REPLACE INTO articles 
                (title, content, source, publish_time, url, keywords, event_id, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article['title'],
                article['content'],
                article['source'],
                article['publish_time'],
                article['url'],
                json.dumps(article.get('keywords', []), ensure_ascii=False),
                article.get('event_id'),
                embedding_blob
            ))
            
            conn.commit()
        except Exception as e:
            print(e)
            print(f"保存文章失败: {e}")
        finally:
            conn.close()
    
    def save_event(self, event_id, one_sentence, summary, weight, article_count, embedding=None):
        """保存事件到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = None
        if embedding is not None:
            embedding_blob = pickle.dumps(embedding)
        
        try:
            cursor.execute(''' 
                INSERT OR REPLACE INTO events 
                (event_id, one_sentence, summary, weight, article_count, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (event_id, one_sentence, summary, weight, article_count, embedding_blob))

            conn.commit()
            print(f"事件 {event_id} 成功保存到数据库")  # 调试信息
        except Exception as e:
            print(f"保存事件失败: {e}")
        finally:
            conn.close()

    
    def get_all_articles_basic(self):
        """获取所有文章的基本信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, title, url, publish_time FROM articles')
        articles = cursor.fetchall()
        
        conn.close()
        
        # 转换为字典列表
        result = []
        for article in articles:
            result.append({
                'id': article[0],
                'title': article[1],
                'url': article[2],
                'publish_time': article[3]
            })
        
        return result
    
    def clear_database(self):
        """清空数据库中的所有数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM articles')
            cursor.execute('DELETE FROM events')
            conn.commit()
            print("数据库已清空")
        except Exception as e:
            print(f"清空数据库失败: {e}")
        finally:
            conn.close()
    
    def get_articles_with_embeddings(self):
        """获取所有带有嵌入向量的文章"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, title, content, source, publish_time, url, keywords, embedding FROM articles WHERE embedding IS NOT NULL')
        articles = cursor.fetchall()
        
        conn.close()
        
        result = []
        for article in articles:
            embedding = None
            if article[7]:  # embedding字段
                embedding = pickle.loads(article[7])
            
            result.append({
                'id': article[0],
                'title': article[1],
                'content': article[2],
                'source': article[3],
                'publish_time': article[4],
                'url': article[5],
                'keywords': json.loads(article[6]) if article[6] else [],
                'embedding': embedding
            })
        
        return result
    
    def update_article_embedding(self, article_id, embedding):
        """更新文章的嵌入向量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = pickle.dumps(embedding)
        
        cursor.execute('UPDATE articles SET embedding = ? WHERE id = ?', (embedding_blob, article_id))
        
        conn.commit()
        conn.close()

class VectorEventTracker:
    """基于向量的事件追踪器"""
    
    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.events = {}  # event_id -> {articles, centroid}
        self.event_summaries = {}  # 添加这个属性来存储事件总结
    
    def cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        if vec1 is None or vec2 is None:
            return 0
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_event_by_vector(self, embedding):
        """通过向量寻找相似的事件"""
        if embedding is None:
            return None
        
        best_event_id = None
        best_similarity = 0
        
        for event_id, event_data in self.events.items():
            similarity = self.cosine_similarity(embedding, event_data['centroid'])
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_event_id = event_id
        
        return best_event_id
    
    def calculate_event_centroid(self, embeddings):
        """计算事件中所有向量的质心"""
        if not embeddings:
            return None
        
        embeddings_array = np.array(embeddings)
        return np.mean(embeddings_array, axis=0).tolist()
    
    def add_article(self, article):
        """添加文章到相应事件（基于向量）"""
        if 'embedding' not in article or article['embedding'] is None:
            print(f"文章 {article['title']} 没有嵌入向量，跳过")  # 调试信息
            return None

        # 寻找相似事件
        event_id = self.find_similar_event_by_vector(article['embedding'])
        
        if event_id:
            # 添加到现有事件
            self.events[event_id]['articles'].append(article)
            # 更新质心
            embeddings = [a['embedding'] for a in self.events[event_id]['articles']]
            self.events[event_id]['centroid'] = self.calculate_event_centroid(embeddings)
            print(f"文章 {article['title']} 被添加到事件 {event_id}")  # 调试信息
        else:
            # 创建新事件
            event_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            self.events[event_id] = {
                'articles': [article],
                'centroid': article['embedding']
            }
            print(f"文章 {article['title']} 创建了新事件 {event_id}")  # 调试信息
        
        return event_id

    
    def cluster_articles(self, articles_with_embeddings, eps=0.5, min_samples=2):
        """使用DBSCAN对文章进行聚类"""
        if not articles_with_embeddings:
            return {}
        
        # 提取嵌入向量
        embeddings = [article['embedding'] for article in articles_with_embeddings]
        
        # 使用DBSCAN进行聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
        
        # 将文章分组到事件中
        events = {}
        for i, label in enumerate(clustering.labels_):
            if label == -1:  # 噪声点，不属于任何事件
                continue
            
            event_id = f"cluster_{label}"
            if event_id not in events:
                events[event_id] = []
            
            events[event_id].append(articles_with_embeddings[i])
        
        # 计算每个事件的质心
        for event_id, articles in events.items():
            embeddings = [article['embedding'] for article in articles]
            centroid = self.calculate_event_centroid(embeddings)
            
            self.events[event_id] = {
                'articles': articles,
                'centroid': centroid
            }
        
        return self.events

    def calculate_event_weights(self):
        """计算事件权重（基于报道数量）"""
        weights = {}
        for event_id, event_data in self.events.items():
            articles = event_data['articles']
            
            # 确保articles是字典列表
            if articles and isinstance(articles[0], dict):
                # 基础权重：报道数量
                base_weight = len(articles)
                
                # 考虑来源多样性
                sources = set(article['source'] for article in articles if isinstance(article, dict) and 'source' in article)
                diversity_bonus = len(sources) * 0.5
                
                weights[event_id] = base_weight + diversity_bonus
            else:
                # 如果articles不是字典列表，使用默认权重
                weights[event_id] = len(articles) if articles else 0
                print(f"警告：事件 {event_id} 的文章格式异常")
        
        return weights

class NewsAITool:
    """新闻AI工具主类"""
    
    def __init__(self, sites, catalogue):
        self.catalogue = catalogue
        self.processor = NewsProcessor()
        self.vector_event_tracker = VectorEventTracker()
        self.database = NewsDatabase()
        self.todayurl = self.database.gettodayurl()    
        self.sites = sites
        self.crawlers = {
            'xinhua': nc.XinhuaCrawler() if 'xinhua' in self.sites else '',
            'people': nc.PeopleDailyCrawler() if 'people' in self.sites else '',
        }
    
        self.sitetrans = {
            'xinhua':'新华网',
            'people': '人民日报',
            }
    
    def get_all_articles_basic(self):
        """获取所有文章的基本信息"""
        return self.database.get_all_articles_basic()
    
    def clear_database(self):
        """清空数据库"""
        self.database.clear_database()
    
    def add_embedding_to_article(self, article_id, embedding):
        """为文章添加嵌入向量"""
        self.database.update_article_embedding(article_id, embedding)
    
    def cluster_articles_by_vectors(self, eps=0.5, min_samples=2):
        """使用向量对文章进行聚类"""
        # 获取所有带有嵌入向量的文章
        articles_with_embeddings = self.database.get_articles_with_embeddings()

        # 使用向量事件追踪器进行聚类
        events = self.vector_event_tracker.cluster_articles(articles_with_embeddings, eps, min_samples)

        # 调试：确认有多少事件被生成
        print(f"聚类后生成的事件数: {len(events)}")  # 调试信息

        # 为每个事件生成总结并保存到数据库
        for event_id, event_data in events.items():
            one_sentence, summary = self.processor.generate_summary(event_data['articles'])

            # 调试：检查生成的事件数据
            print(f"生成的事件 {event_id}：一句话新闻: {one_sentence}, 详细总结: {summary}")

            # 保存事件到数据库
            self.database.save_event(
                event_id, one_sentence, summary, 
                len(event_data['articles']), len(event_data['articles']),
                event_data['centroid']
            )

            # 更新文章中事件ID
            for article in event_data['articles']:
                article['event_id'] = event_id
                self.database.save_article(article)

        return events
    
    def crawl_all_news(self):
        """爬取所有新闻源的文章"""

        all_articles = []

        for site in self.sites:
            _ord = 0
            crawler = self.crawlers[site]
            if not crawler: 
                continue

            print(f"正在爬取 {self.sitetrans[site]} 新闻")
            urls = crawler.generateurl()
            '''
            urls = [
                'http://www.news.cn/world/20251020/27e6e9a077d0489c8fd1ace6ef15b23b/c.html',
                'http://www.news.cn/politics/20251020/4893586fb647414d8769f2fdb73352e8/c.html',
            ]
            '''
            todayurls = self.database.gettodayurl()
            urls = set(urls) - set(todayurls)
            _tot = len(urls)
            if not urls:
                continue

            for url in urls:
                _ord += 1
                print(f"正在爬取第{_ord}/{_tot} 条")
                article = crawler.crawl_news(url)
                if article:
                    # 提取关键词
                    article['keywords'] = self.processor.extract_keywords(
                        f"{article['title']} {article['content']}"
                    )

                    # 生成文章的嵌入向量
                    embedding = gtembd(article['title'] + article['content'])  # Adjust embedding generation logic if needed
                    article['embedding'] = embedding

                    print(f"文章 {article['title']} 的嵌入向量：{embedding[:10]}...")  # 只显示前10个数字

                    # 使用向量追踪事件
                    event_id = self.vector_event_tracker.add_article(article)
                    article['event_id'] = event_id

                    all_articles.append(article)
                    
                    # 保存到数据库
                    self.database.save_article(article)
                
                time.sleep(1)  # 礼貌爬取
        
        return all_articles
    
    
    def process_events(self):
        """处理所有事件并生成总结"""
        # 计算事件权重
        weights = self.vector_event_tracker.calculate_event_weights()
        print(f"事件权重: {weights}")  # 调试信息

        # 为每个事件生成总结
        for event_id, event_data in self.vector_event_tracker.events.items():
            articles = event_data['articles']
            one_sentence, summary = self.processor.generate_summary(articles)
            
            # 确保生成了总结
            if one_sentence and summary:
                print(f"事件 {event_id} 总结生成成功：{one_sentence}, {summary}")
            else:
                print(f"事件 {event_id} 总结生成失败")

            self.vector_event_tracker.event_summaries[event_id] = {
                'one_sentence': one_sentence,
                'summary': summary,
                'weight': weights.get(event_id, 0),
                'article_count': len(articles)
            }

            # 保存到数据库
            self.database.save_event(
                event_id, one_sentence, summary, 
                weights.get(event_id, 0), len(articles)
            )

    
    def get_important_events(self, top_n=5):
        """获取最重要的事件"""
        events_with_weights = [
            (event_id, info['weight'], info) 
            for event_id, info in self.vector_event_tracker.event_summaries.items()
        ]
        print(f"事件总数: {len(events_with_weights)}")

        # 按权重排序
        sorted_events = sorted(events_with_weights, key=lambda x: x[1], reverse=True)
        
        # 输出调试信息
        for event_id, weight, info in sorted_events:
            print(f"事件 {event_id} 权重: {weight:.2f}")

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
        
        return True

# 使用示例
if __name__ == "__main__":
    # 初始化工具（请替换为你的API密钥）
    ai_tool = NewsAITool(sites, catalogue)
    
    ai_tool.database.clear_database()

    # 运行完整流程
    ai_tool.run()
    