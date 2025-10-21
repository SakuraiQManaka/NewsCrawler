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
from logs import info, log, warning, error, debug, clean

class NewsCrawler:
    """新闻爬虫基类"""
    
    def __init__(self):
        #clean()
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
    
    def generateurl(self):
        urls = []
        catalogue = {
        "politics" : "https://www.xinhuanet.com/politics/szlb/index.html",
        "fortune" : "https://www.xinhuanet.com/fortune/gundong/index.html",
        "world": "https://www.news.cn/world/jsxw/index.html",
        }
        try:
            for key,value in catalogue.items():
                _ord = 0
                response = self.session.get(value, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")
                '''
                content_list = soup.find('div', id='content-list')
                tit = content_list.find('div', class_='tit')
                if content_list:
                    a_tags = content_list.find_all('a')
                    for a in a_tags:
                        if a.has_attr('href'):
                            href = a['href']
                            url = "https://www.xinhuanet.com" + href if "www" not in href else href
                            urls.append(url)
                            log(url)
                            _ord += 1
                '''
                a_list = soup.find_all('a')
                for a in a_list:
                    href = a['href']
                    if "c.html" in href and "share" not in href:
                        url = "https://www.xinhuanet.com" + href if "www" not in href else href
                    elif  "c.html" in href and "share" in href :
                        pos1 = href.index("=")
                        pos2 = href.index("&")
                        url = "https://www.xinhuanet.com" + href[pos1+1:pos2]
                        urls.append(url)
                        log(url)
                        _ord += 1
                log(f"成功生成新华网 {key} 新闻列表，共{_ord}个新闻")
                print(f"成功生成新华网 {key} 新闻列表，共{_ord}个新闻")      
            log(f"新华网全部新闻url生成完毕，总计 {len(urls)} 条")
            print(f"新华网全部新闻url生成完毕，总计 {len(urls)} 条")
            return urls
        except Exception as e:
            error(f"生成新华网新闻地址失败：{e}")
            print(f"生成新华网新闻地址失败：{e}")
            return None
    
    def crawl_news(self, url):
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 获取标题
            title = soup.find("span", class_="title").get_text().strip() if soup.find("span", class_="title") else ''
            print(f"标题： {title}")

            # 获取发布时间
            time_element = soup.find("div", class_="header-time")
            publish_time = "/".join(i.get_text() for i in time_element.find_all("em")) if time_element else ''
            
            # 获取正文内容
            content = soup.find("span", id="detailContent").get_text()
            cleaned_text = content.replace('\xa0', ' ').replace('\u3000', ' ').replace('\u2002', ' ').replace('\u2003', ' ')
            
            log(url, opr = "crawl")

            return {
                'title': title,
                'content': cleaned_text,
                'publish_time': publish_time,
                'source': 'xinhua',
                'url': url
            }
        except Exception as e:
            error(f"爬取新华网新闻失败: {e}, "+url)
            print(f"爬取新华网新闻失败: {e}")
            return None

class PeopleDailyCrawler(NewsCrawler):
    """
    人民日报爬虫 
    待完善……
    """
    def generateurl(self):
        urls = []
        catalogue = {
        "finance" : "http://finance.people.com.cn/GB/70846/index.html",
        "society" : "http://society.people.com.cn/GB/136657/index.html",
        "ent": "http://ent.people.com.cn/GB/436801/index.html",
        "health" : "http://health.people.com.cn/GB/415859/index.html",
        "world" : "https://world.people.com.cn/GB/157278/index.html",
        "military": "http://military.people.com.cn/GB/172467/index.html",
        }
        try:
            for key,value in catalogue.items():
                _ord = 0
                response = self.session.get(value, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")
                content_list = soup.find('div', class_='ej_list_box')
                if content_list:
                    lis = content_list.find_all('li')
                    for tit_div in lis:
                        a_tag = tit_div.find('a')
                        if a_tag and a_tag.has_attr('href'):
                            href = a_tag['href']
                            url = f"http://{key}.people.com.cn" + href if "www" not in href else href
                            urls.append(url)
                            _ord += 1
                            log(url)
                log(f"成功生成人民网 {key} 新闻列表，共{_ord}个新闻")
                print(f"成功生成人民网 {key} 新闻列表，共{_ord}个新闻")      
            log(f"人民网全部新闻url生成完毕，总计 {len(urls)} 条")
            print(f"人民网全部新闻url生成完毕，总计 {len(urls)} 条")
            return urls
        except Exception as e:
            error(f"生成人民网新闻地址失败：{e}")
            print(f"生成人民网新闻地址失败：{e}")
            return None


    def crawl_news(self, url):
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            # 获取标题
            title_h1 = soup.find('h3', class_='pre').find_next_sibling()
            title = title_h1.get_text().strip() if soup.find('h1') else ''

            # 获取发布时间
            time_text = soup.find('div', class_='channel').contents[1].get_text().strip()
            timelist = [time_text[0:4],time_text[5:7],time_text[8:10]]
            publish_time = "/".join(i for i in timelist) if timelist else ''
            
            # 获取正文内容
            content_elements = soup.find('div', class_="bza").parent.select('p[style = "text-indent: 2em;"]')
            content = '\n'.join([p.get_text().strip() for p in content_elements])
            
            return {
                'title': title,
                'content': content,
                'publish_time': publish_time,
                'source': 'people',
                'url': url
            }
        except Exception as e:
            error(f"爬取人民网新闻失败: {e}, "+url)
            print(f"爬取人民网新闻失败: {e}")
            return None