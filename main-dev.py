# -*- coding: utf-8 -*-

"""
新闻聚合与事件追踪系统
Copyright (C) 2025 Troy Wang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Created on Tue Oct 21 17:25 2025

@author: Troy
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
from openai import OpenAI
from typing import List, Dict, Any
import peewee
import os
import numpy as np
from sklearn.cluster import DBSCAN
import pickle
import newscrawler as nc

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

# Peewee 数据库定义
db = peewee.SqliteDatabase('news.db')

class BaseModel(peewee.Model):
    class Meta:
        database = db

class Article(BaseModel):
    title = peewee.TextField()
    content = peewee.TextField(null=True)
    source = peewee.TextField()
    publish_time = peewee.TextField(null=True)
    url = peewee.TextField(unique=True)
    keywords = peewee.TextField(null=True)  # 存储为JSON字符串
    event_id = peewee.TextField(null=True)
    embedding = peewee.BlobField(null=True)  # 存储1024维向量
    created_at = peewee.DateTimeField(constraints=[peewee.SQL('DEFAULT CURRENT_TIMESTAMP')])

class Event(BaseModel):
    event_id = peewee.TextField(primary_key=True)
    one_sentence = peewee.TextField(null=True)
    summary = peewee.TextField(null=True)
    weight = peewee.FloatField(null=True)
    article_count = peewee.IntegerField(null=True)
    embedding = peewee.BlobField(null=True)  # 存储事件平均向量
    last_updated = peewee.DateTimeField(constraints=[peewee.SQL('DEFAULT CURRENT_TIMESTAMP')])
    # 新增时间标签字段
    time_label = peewee.TextField(null=True)  # 存储第一篇相关文章的发布时间

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
        if len(articles)<=10:
            combined_content = "\n\n".join([
                f"来源: {article['source']}\n标题: {article['title']}\n内容: {article['content']}"
                for article in articles
            ])
        elif len(articles)>10:
            combined_content = "\n\n".join([
                f"来源: {article['source']}\n标题: {article['title']}\n内容: {article['content'][:800]}"
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

    def analyze_recent_events(self, events_data):
        """分析七日内所有事件的综合趋势"""
        if not events_data:
            return "暂无近期事件可供分析"
        
        try:
            # 构建事件分析内容
            events_content = ""
            for i, (event_id, weight, info) in enumerate(events_data, 1):
                events_content += f"\n{i}. 事件权重: {weight:.2f}\n"
                events_content += f"   一句话新闻: {info['one_sentence']}\n"
                events_content += f"   详细总结: {info['summary']}\n"
                events_content += f"   相关文章数: {info['article_count']}\n"
                events_content += f"   时间标签: {info['time_label']}\n"
            
            prompt = f"""
            请分析以下七日内发生的新闻事件，总结整体趋势和热点话题：
            
            {events_content}
            
            请从以下几个方面进行分析：
            1. 整体趋势和热点话题
            2. 各事件的关联性和可能的发展方向
            3. 对社会、经济或政治的可能影响
            4. 建议关注的重点领域
            
            请用清晰的结构输出分析结果，使用中文回答。
            """
            content = "你是一个专业的新闻分析师，擅长从多个事件中识别趋势和模式。"
            
            response = rqds(prompt, content, 0.5)
            analysis = response.choices[0].message.content.strip()
            
            return analysis
            
        except Exception as e:
            print(f"事件分析失败: {e}")
            return "事件分析失败，请检查网络连接或API配置"

class NewsDatabase:
    """新闻数据库"""
    
    def __init__(self, db_path='news.db'):
        self.db_path = db_path
        # 更新数据库路径
        db.init(self.db_path)
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        db.connect()
        db.create_tables([Article, Event], safe=True)
        # 确保索引存在
        try:
            # 对于SQLite，使用原始SQL创建索引
            db.execute_sql('CREATE INDEX IF NOT EXISTS idx_article_publish_time ON article(publish_time);')
        except Exception as e:
            print(f"创建索引时出错: {e}")
        db.close()
    
    def gettodayurl(self):
        """获取今天发布的所有文章的URL"""
        today = datetime.now().strftime("%Y/%m/%d")
        todayurls = []

        try:
            # 使用数据库索引，只查询今天的数据
            articles = Article.select().where(Article.publish_time == today)
            
            for article in articles:
                todayurls.append(article.url)
            
            print(f"找到今天发布的文章URL数量: {len(todayurls)}")
            return todayurls
            
        except Exception as e:
            print(f"获取今天URL失败: {e}")
            return []

    def save_article(self, article):
        """保存文章到数据库"""
        try:
            # 处理embedding字段
            embedding_blob = None
            if 'embedding' in article:
                embedding_blob = pickle.dumps(article['embedding'])
            
            # 处理keywords字段
            keywords_str = None
            if 'keywords' in article:
                keywords_str = json.dumps(article['keywords'], ensure_ascii=False)
            
            # 使用Peewee的insert或replace操作
            Article.replace(
                title=article['title'],
                content=article['content'],
                source=article['source'],
                publish_time=article['publish_time'],
                url=article['url'],
                keywords=keywords_str,
                event_id=article.get('event_id'),
                embedding=embedding_blob
            ).execute()
            
        except Exception as e:
            print(f"保存文章失败: {e}")
    
    def save_event(self, event_id, one_sentence, summary, weight, article_count, embedding=None, time_label=None):
        """保存事件到数据库"""
        embedding_blob = None
        if embedding is not None:
            embedding_blob = pickle.dumps(embedding)
        
        try:
            Event.replace(
                event_id=event_id,
                one_sentence=one_sentence,
                summary=summary,
                weight=weight,
                article_count=article_count,
                embedding=embedding_blob,
                time_label=time_label
            ).execute()
            print(f"事件 {event_id} 成功保存到数据库")
        except Exception as e:
            print(f"保存事件失败: {e}")

    def get_recent_events(self, days=7):
        """获取最近指定天数内的事件"""
        try:
            # 计算起始日期
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y/%m/%d")
            
            # 查询时间标签在指定日期之后的事件
            events = Event.select().where(Event.time_label >= start_date)
            
            result = []
            for event in events:
                result.append({
                    'event_id': event.event_id,
                    'one_sentence': event.one_sentence,
                    'summary': event.summary,
                    'weight': event.weight,
                    'article_count': event.article_count,
                    'time_label': event.time_label
                })
            
            print(f"找到最近{days}天内的事件数量: {len(result)}")
            return result
            
        except Exception as e:
            print(f"获取近期事件失败: {e}")
            return []
    
    def get_all_articles_basic(self):
        """获取所有文章的基本信息"""
        articles = Article.select(Article.id, Article.title, Article.url, Article.publish_time)
        
        # 转换为字典列表
        result = []
        for article in articles:
            result.append({
                'id': article.id,
                'title': article.title,
                'url': article.url,
                'publish_time': article.publish_time
            })
        
        return result
    
    def clear_database(self):
        """清空数据库中的所有数据"""
        try:
            Article.delete().execute()
            Event.delete().execute()
            print("数据库已清空")
        except Exception as e:
            print(f"清空数据库失败: {e}")
    
    def get_articles_with_embeddings(self):
        """获取所有带有嵌入向量的文章"""
        articles = Article.select().where(Article.embedding.is_null(False))
        
        result = []
        for article in articles:
            embedding = None
            if article.embedding:
                embedding = pickle.loads(article.embedding)
            
            keywords = []
            if article.keywords:
                keywords = json.loads(article.keywords)
            
            result.append({
                'id': article.id,
                'title': article.title,
                'content': article.content,
                'source': article.source,
                'publish_time': article.publish_time,
                'url': article.url,
                'keywords': keywords,
                'embedding': embedding
            })
        
        return result
    
    def update_article_embedding(self, article_id, embedding):
        """更新文章的嵌入向量"""
        embedding_blob = pickle.dumps(embedding)
        
        Article.update(embedding=embedding_blob).where(Article.id == article_id).execute()

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
            
            # 更新事件时间标签（取最早的文章时间）
            if 'time_label' not in self.events[event_id] or \
               (article['publish_time'] and article['publish_time'] < self.events[event_id]['time_label']):
                self.events[event_id]['time_label'] = article['publish_time']
                
            print(f"文章 {article['title']} 被添加到事件 {event_id}")  # 调试信息
        else:
            # 创建新事件
            event_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            self.events[event_id] = {
                'articles': [article],
                'centroid': article['embedding'],
                'time_label': article['publish_time']  # 设置时间标签为第一篇文章的时间
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
        
        # 计算每个事件的质心和时间标签
        for event_id, articles in events.items():
            embeddings = [article['embedding'] for article in articles]
            centroid = self.calculate_event_centroid(embeddings)
            
            # 计算时间标签（取最早的文章时间）
            time_label = None
            for article in articles:
                if article['publish_time']:
                    if time_label is None or article['publish_time'] < time_label:
                        time_label = article['publish_time']
            
            self.events[event_id] = {
                'articles': articles,
                'centroid': centroid,
                'time_label': time_label
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
                event_data['centroid'],
                event_data.get('time_label')  # 保存时间标签
            )

            # 更新文章中事件ID
            for article in event_data['articles']:
                article['event_id'] = event_id
                self.database.save_article(article)

        return events
    
    def crawl_all_news(self):
        """爬取所有新闻源的文章"""

        all_articles = []
        todayurls = self.database.gettodayurl()

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
        _tot = len(self.vector_event_tracker.events.items())
        _ord = 0

        # 为每个事件生成总结
        for event_id, event_data in self.vector_event_tracker.events.items():
            _ord += 1
            articles = event_data['articles']
            one_sentence, summary = self.processor.generate_summary(articles)
            
            # 确保生成了总结
            if one_sentence and summary:
                print(f"(进度{_ord}/{_tot})事件 {event_id} 总结生成成功：{one_sentence}, {summary}")
            else:
                print(f"(进度{_ord}/{_tot})事件 {event_id} 总结生成失败")

            self.vector_event_tracker.event_summaries[event_id] = {
                'one_sentence': one_sentence,
                'summary': summary,
                'weight': weights.get(event_id, 0),
                'article_count': len(articles),
                'time_label': event_data.get('time_label')  # 保存时间标签
            }

            # 保存到数据库
            self.database.save_event(
                event_id, one_sentence, summary, 
                weights.get(event_id, 0), len(articles),
                embedding=None,  # 如果需要保存embedding，可以添加
                time_label=event_data.get('time_label')  # 保存时间标签
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

    def analyze_recent_events(self, days=7):
        """分析最近指定天数内的事件"""
        # 从数据库获取近期事件
        recent_events = self.database.get_recent_events(days)
        
        if not recent_events:
            print(f"最近{days}天内没有事件可供分析")
            return None
        
        # 转换为与get_important_events相同的格式
        events_data = []
        for event in recent_events:
            events_data.append((
                event['event_id'],
                event['weight'] or 0,
                {
                    'one_sentence': event['one_sentence'],
                    'summary': event['summary'],
                    'article_count': event['article_count'],
                    'time_label': event['time_label']
                }
            ))
        
        # 按权重排序
        sorted_events = sorted(events_data, key=lambda x: x[1], reverse=True)
        
        # 使用AI分析近期事件
        analysis = self.processor.analyze_recent_events(sorted_events)
        
        return analysis
    
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
            print(f"   时间标签: {info.get('time_label', '未知')}")
        
        # 分析近期事件
        print("\n=== 七日事件综合分析 ===")
        recent_analysis = self.analyze_recent_events(7)
        print(recent_analysis)
        
        return True

# 使用示例
if __name__ == "__main__":
    # 初始化工具（请替换为你的API密钥）
    ai_tool = NewsAITool(sites, catalogue)
    
    # ai_tool.database.clear_database()

    # 运行完整流程
    ai_tool.run()