import numpy as np


class BatchGenerator(object):
    """
    batch生成器
    1.加工原始数据得到句子向量:get_sentence_matrix
    2.从原始文件中得到所有句子的向量:read_raw_input
    """

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.max_seq_length = config.max_seq_length

        self.raw_input_file = config.raw_input_file
        self.label_file = config.label_file

        self.label_dict = dict()
        self.pointer = 0

        self.create_batches()

    def reset_batch_pointer(self):
        self.pointer = 0

    def get_num_classes(self):
        return len(self.label_dict)

    def create_batches(self):
        sentence_matrices = self.read_raw_input()
        labels, self.label_dict = self.read_labels()  # label_dict作为成员变量

        # 通过数量粗略检查输入集和标签集是否匹配
        assert len(sentence_matrices) == len(labels), '输入集和标签集大小不一致，恐有错误'

        # 先划分训练集和测试集，80%的数据为训练集，20%为测试集
        total_data_len = len(sentence_matrices)
        num_train = total_data_len // 5 * 4
        num_test = total_data_len - num_train
        # 输入训练集
        input_train_set = sentence_matrices[:num_train]
        input_test_set = sentence_matrices[num_train:]
        # 标签训练集
        label_train_set = labels[:num_train]
        label_test_set = labels[num_train:]

        # 再划分batch
        self.num_train_batches = num_train // self.batch_size  # 训练集batch数量
        self.num_test_batches = num_test // self.batch_size  # 测试集batch数量
        # 输入训练集batch
        input_train_set = input_train_set[
                          :self.num_train_batches * self.batch_size]  # 截取掉尾巴上不满batch_size个的数据，下面好进行split
        input_test_set = input_test_set[:self.num_test_batches * self.batch_size]  # 截取掉尾巴上不满batch_size个的数据，下面好进行split
        self.train_x = np.split(np.array(input_train_set), self.num_train_batches)  # 必须转换为np.array才能被split
        self.test_x = np.split(np.array(input_test_set), self.num_test_batches)
        # 标签训练集batch
        label_train_set = label_train_set[
                          :self.num_train_batches * self.batch_size]  # 截取掉尾巴上不满batch_size个的数据，下面好进行split
        label_test_set = label_test_set[:self.num_test_batches * self.batch_size]  # 截取掉尾巴上不满batch_size个的数据，下面好进行split
        self.train_y = np.split(np.array(label_train_set), self.num_train_batches)  # 必须转换为np.array才能被split
        self.test_y = np.split(np.array(label_test_set), self.num_test_batches)

        # 至此，训练集和测试集已划分完毕，
        # 训练集的输入和标签的shape为：
        # 输入：[self.num_train_batches,self.batch_size,max_seq_length]
        # 标签：[self.num_train_batches,self.batch_size]
        ##############################################################
        # 测试集的输入和标签的shape为：
        # 输入：[self.num_test_batches,self.batch_size,max_seq_length]
        # 标签：[self.num_test_batches,self.batch_size]
        print('训练集和测试集划分完毕！')

    def next_train_batch(self):
        current_batch_x, current_batch_y = self.train_x[self.pointer], self.train_y[self.pointer]
        self.pointer += 1

        return current_batch_x, current_batch_y

    def read_raw_input(self):
        with open(self.raw_input_file, 'r') as f:
            sentences = f.readlines()

        sentence_matrices = []
        for sentence in sentences:
            sentence_matrices.append(sentence)

        return sentence_matrices
