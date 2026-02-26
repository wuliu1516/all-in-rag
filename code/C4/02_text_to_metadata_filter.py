import logging
import os

import requests
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import BiliBiliLoader
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU",
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]


def manual_load_bili(bv_list, sessdata):
    docs = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Cookie": f"SESSDATA={sessdata}",
        "Referer": "https://www.bilibili.com",
    }

    for bv in bv_list:
        # 获取视频基本信息 API
        api_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bv}"
        resp = requests.get(api_url, headers=headers)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            print("*" * 100)
            # 模拟 LangChain Document
            from langchain_core.documents import Document

            content = data.get("desc", "") + " " + data.get("title", "")
            # 保持原始API返回的数据结构，供后续处理
            metadata = data
            print(f"标题: {data.get('title')}")
            print(f"作者: {data.get('owner', {}).get('name')}")
            print(f"观看次数: {data.get('stat', {}).get('view')}")
            print(f"时长: {data.get('duration')}秒")
            docs.append(Document(page_content=content, metadata=metadata))
            print(f"成功加载: {data.get('title')}")
    return docs


bili = []
try:
    # loader = BiliBiliLoader(
    #     video_urls=video_urls,
    #     sessdata="ab986b01%2C1784623446%2Cf841b%2A11",
    #     bili_jct="b8dcac081d4b9d6612b781a9a7e97d89",
    #     buvid3="AA824CEA-38E8-D073-B67C-6217CB17138A83543infoc",
    # )
    # docs = loader.load()
    video_bvids = ["BV1Bo4y1A7FU", "BV1ug4y157xA", "BV1yh411V7ge"]
    docs = manual_load_bili(video_bvids, "ab986b01%2C1784623446%2Cf841b%2A11")

    for doc in docs:
        original = doc.metadata

        # 提取基本元数据字段
        metadata = {
            "title": original.get("title", "未知标题"),
            "author": original.get("owner", {}).get("name", "未知作者"),
            "source": original.get("bvid", "未知ID"),
            "view_count": original.get("stat", {}).get("view", 0),
            "length": original.get("duration", 0),
        }

        doc.metadata = metadata
        bili.append(doc)

except Exception as e:
    print(f"加载BiliBili视频失败: \n{str(e)}")

if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

# 2. 创建向量存储
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(bili, embed_model)

# 3. 配置元数据字段信息
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(name="length", description="视频长度（整数）", type="integer"),
]

# 4. 创建自查询检索器
llm = ChatDeepSeek(model="deepseek-chat", temperature=0, api_key=os.getenv("DEEPSEEK_API_KEY"))

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True,
)

# 5. 执行查询示例
queries = ["时长最小的一个视频", "时长大于600秒的视频"]

for query in queries:
    print(f"\n--- 查询: '{query}' ---")
    results = retriever.invoke(query)
    if results:
        for doc in results:
            title = doc.metadata.get("title", "未知标题")
            author = doc.metadata.get("author", "未知作者")
            view_count = doc.metadata.get("view_count", "未知")
            length = doc.metadata.get("length", "未知")
            print(f"标题: {title}")
            print(f"作者: {author}")
            print(f"观看次数: {view_count}")
            print(f"时长: {length}秒")
            print("=" * 50)
    else:
        print("未找到匹配的视频")
