from openai import OpenAI
import os

"""
DeepSeek 和 DashScope API 的代理类

这个类封装了与 DeepSeek 聊天补全和 DashScope 文本嵌入相关的 API 调用，
提供统一的接口来处理对话生成和文本向量化任务。

Attributes:
    deepseek_api_key (str): DeepSeek API 的访问密钥
    dashscope_api_key (str): DashScope API 的访问密钥

Raises:
    ValueError: 当环境变量中缺少必要的 API 密钥时
"""
class Agent:
    def __init__(self):
        """
        初始化 Agent 实例
        
        检查并加载 DeepSeek 和 DashScope 的 API 密钥。
        两个密钥都必须存在于环境变量中。
        
        Raises:
            ValueError: 如果 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY 环境变量未设置
        """
        if not os.environ.get('DEEPSEEK_API_KEY') or not os.environ.get('DASHSCOPE_API_KEY'):
            raise ValueError("DeepSeek API key or DashScope API key not found in environment variables")
        else:
            self.deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY')
            self.dashscope_api_key = os.environ.get('DASHSCOPE_API_KEY')

    def rqds(self, sysprompt, userprompt, temperature):
        """
        DeepSeek 聊天补全请求
        
        向 DeepSeek API 发送聊天补全请求，生成对话回复。
        
        Args:
            sysprompt (str): 系统提示词，用于设定助手的行为和角色
            userprompt (str): 用户提示词，即用户的输入内容
            temperature (float): 温度参数，控制回复的随机性 (0-2)
                            较低值产生更确定的输出，较高值产生更多样化的输出
        
        Returns:
            ChatCompletion: DeepSeek API 的响应对象，包含生成的回复内容
        
        Note:
            使用 deepseek-chat 模型，关闭流式输出
        """
        client = OpenAI(
            api_key = self.deepseek_api_key,
            base_url = "https://api.deepseek.com/v1")
        
        response = client.chat.completions.create(
            model = "deepseek-chat",
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": userprompt}
            ],
            stream = False,
            temperature = temperature)
        return response
    
    def gtembd(self, text):
        """
        DashScope 文本嵌入生成
        
        使用 DashScope 的 text-embedding-v4 模型将文本转换为 1024 维向量。
        
        Args:
            text (str): 需要转换为嵌入向量的输入文本
        
        Returns:
            Embedding: DashScope API 的响应对象，包含文本的嵌入向量
        
        Note:
            嵌入维度固定为 1024,适用于语义搜索、文本相似度等任务
        """
        client = OpenAI(
            api_key = self.dashscope_api_key,
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        completion = client.embeddings.create(
            model = "text-embedding-v4",
            input = text,
            dimensions = 1024)
        return completion


"""
Agent 类的单例实例

在整个模块中共享同一个 Agent 实例，避免重复初始化开销，
确保 API 客户端和配置的一致性。
"""
_agent = Agent()

def ds(sysprompt, userprompt, temperature) -> str:
    """
    DeepSeek 聊天补全接口
    
    高级封装函数，提供简化的 DeepSeek 聊天补全调用接口，
    包含参数验证和错误处理。
    
    Args:
        sysprompt (str): 系统提示词，定义助手的行为和角色
        userprompt (str): 用户输入内容
        temperature (float): 温度参数，范围 0-2,控制回复的随机性
    
    Returns:
        str: 生成的回复内容，去除首尾空白字符
        
    Raises:
        ValueError: 当输入参数不符合要求时
    """
    if not isinstance(sysprompt, str) or not sysprompt.strip():
        raise ValueError("sysprompt must be a non-empty string")
    
    if not isinstance(userprompt, str) or not userprompt.strip():
        raise ValueError("userprompt must be a non-empty string")
        
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")

    try:
        response = _agent.rqds(sysprompt, userprompt, temperature)
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        else:
            return ""
    except Exception as e:
        print(f"Error in ds: {e}")
        return ""

def bd(text) -> list[float]:
    """
    DashScope 文本嵌入接口
    
    高级封装函数，提供简化的 DashScope 文本嵌入调用接口，
    包含参数验证和错误处理。
    
    Args:
        text (str): 需要嵌入的文本内容
    
    Returns:
        list[float]: 1024 维的浮点数向量，表示文本的语义嵌入
        
    Raises:
        ValueError: 当输入文本为空或不是字符串时
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    
    try:
        completion = _agent.gtembd(text)
        if completion.data and completion.data[0].embedding:
            return completion.data[0].embedding
        else:
            return []
    except Exception as e:
        print(f"Error in bd: {e}")
        return []