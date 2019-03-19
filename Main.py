import numpy as np
import Prepare
import MultinomialMixtureModel

mn = 1.0
K = [5, 10 , 20, 30]
index = 0
top_word_num = 5
max_epoch = 30

vocabulary_path = "./data/vocabulary.txt"
train_data_path = "./data/train.data"
stop_word_path = "./data/stopwords.txt"

print("Preparing data set......")
Data = Prepare.Data(vocabulary_path=vocabulary_path, train_data_path=train_data_path, stop_word_path=stop_word_path)
print("Preparing data set done.")

for i in range(len(K)):
    cluster_num = K[i]
    MMM = MultinomialMixtureModel.MultinomialMixtureModel(Data.doc_num, cluster_num, Data.voc_size)

    print("Training model......")
    MMM.train(T=np.array(Data.T, dtype='float'), max_epoch=max_epoch)
    print("Training model done.")

    MMM.output(top_word_num, data=Data)

    MMM.predict(doc_size=Data.doc_num, T=np.array(Data.T, dtype='float'))
    # MMM.test(T=np.array(Data.T, dtype='float'), doc_size=Data.doc_num)


