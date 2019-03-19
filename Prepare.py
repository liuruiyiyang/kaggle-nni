import numpy as np


class Data(object):

    def __init__(self, vocabulary_path, train_data_path, stop_word_path):

        self.doc_num = 0
        self.voc_size = 0
        self.stop_word_size = 0
        self.T = []
        self.stop_word_id = []
        self.vocabulary_dict = {}
        self.stop_word_dict = {}
        self.vocabulary_path = vocabulary_path
        self.train_data_path = train_data_path
        self.stop_word_path = stop_word_path
        self.create_vocabulary_dict(vocabulary_path)
        self.create_stop_word_dict(stop_word_path)
        self.create_stop_word_id()
        self.create_training_set(train_data_path)

    def create_vocabulary_dict(self, vocabulary_path):

        voc_file = open(vocabulary_path)
        i = 0
        for line in voc_file.readlines():
            i = i + 1
            self.vocabulary_dict[i] = line.strip()
            self.voc_size += 1
        voc_file.close()
        # print("vocabulary_dict:")
        # print(self.vocabulary_dict)
        print("vocabulary_dict size:" + str(self.voc_size))

    def create_training_set(self, train_data_path):

        train_data_file = open(train_data_path)
        current_doc_id = "1"
        doc = np.zeros(self.voc_size, dtype='float')
        for line in train_data_file.readlines():
            doc_id = line.split(' ')[0]
            word_id = line.split(' ')[1]
            word_count = line.split(' ')[2]
            if doc_id != current_doc_id:
                self.T.append(doc)
                self.doc_num += 1
                doc = np.zeros(self.voc_size, dtype='float')
                current_doc_id = doc_id
            if int(word_id) in self.stop_word_id:
                doc[int(word_id)] = 0.0
            else:
                doc[int(word_id)] = float(word_count)
        self.T.append(doc)
        self.doc_num += 1
        train_data_file.close()
        print("training set T size:" + str(self.doc_num))

    def create_stop_word_dict(self, stop_word_path):

        stop_word_file = open(stop_word_path)
        i = 0
        for line in stop_word_file.readlines():
            i = i + 1
            self.stop_word_dict[i] = line.strip()
            self.stop_word_size += 1
        stop_word_file.close()
        print("stop word size:" + str(self.stop_word_size))

    def create_stop_word_id(self):

        for i in range(1, self.voc_size + 1):
            for j in range(1, self.stop_word_size + 1):
                if self.vocabulary_dict[i] == self.stop_word_dict[j]:
                    self.stop_word_id.append(i)
        # print(self.stop_word_id)
