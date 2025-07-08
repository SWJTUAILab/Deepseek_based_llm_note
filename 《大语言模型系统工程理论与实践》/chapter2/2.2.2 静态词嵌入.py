from gensim.models import Word2Vec, FastText, KeyedVectors #使用gensim训练Word2Vec、FastText和KeyedVector属于本地训练，
                                                            #直接在命令行中输入pip install gensim即可，不是外部加载模型

# 假设我们有一个句子列表 (分好词的)
sentences = [["this", "is", "the", "first", "sentence"],
            ["this", "is", "the", "second", "sentence"]]

# 训练Word2Vec模型
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
word_vector_w2v = model_w2v.wv["sentence"]
print("Word2Vec vector for 'sentence':", word_vector_w2v)

# 训练FastText模型
model_ft = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4)
word_vector_ft = model_ft.wv["sentence"]
print("FastText vector for 'sentence':", word_vector_ft)

# 加载预训练的GloVe模型 (通常以文本格式提供)
# glove_file = 'path/to/glove.6B.100d.txt' 
# model_glove = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
# word_vector_glove = model_glove["sentence"]
# print("GloVe vector for 'sentence':", word_vector_glove)

# 实际应用中，通常会使用更大的语料库进行训练或加载大规模预训练模型
