以下前一步输出皆为后一步输入

###0 数据整理
先用extract中的extract_train_data.ipynb处理，得到一系列mongodb collections
###1 预处理:
在项目目录下执行python preprocess.py -a oa帐号，可得到：
####1.1 生成原始数据
原始的“标签-正文”文件，放在data/raw目录中
####1.2 分词和词性标注
正文分词后的“标签-正文”文件，放在data/corpus目录中
####1.3 划分测试集和训练集
划分为训练集和测试集，放在data/corpus目录中
###2 训练并生成词向量
在项目目录下执行python train_w2v.py -a oa帐号，使用word2vec模型对语料库进行训练
###3 将语料库转换为词向量
在项目目录下执行python prepare_data.py -a oa帐号，