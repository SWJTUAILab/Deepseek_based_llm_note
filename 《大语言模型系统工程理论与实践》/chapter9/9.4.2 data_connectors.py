from llama_index.core import SimpleDirectoryReader

def load_documents(directory_path="data"):
    """
    从指定目录加载文档数据
    """
    documents = SimpleDirectoryReader(directory_path).load_data()
    return documents
