import numpy as np
import time


class MultinomialMixtureModel(object):

    def __init__(self, N, K, W):

        self.N = N
        self.K = K
        self.W = W
        self.q = 0
        np.random.seed(36)
        self.PI = np.random.rand(self.K)
        self.PI = self.PI / np.sum(self.PI)
        self.MU = np.random.rand(self.K, self.W)
        self.MU = self.MU / np.sum(self.MU, axis=1).reshape(-1, 1)

    def E_step(self, T, mu, pi):

        log = np.dot(T, np.log(mu.T)) + np.log(pi)
        m = np.max(log, axis=1).reshape(-1, 1)
        d = m + np.log(np.sum(np.exp(log - m), axis=1).reshape(-1, 1))
        q = log - d
        return np.exp(q)

    def M_step(self, T, q, n):

        pi = np.sum(q, axis=0) / n
        tmp = np.dot(q.T, T) + 1e-30
        m = np.sum(tmp, axis=1).reshape(-1, 1)
        mu = tmp / m
        return pi, mu

    def train(self, T, max_epoch):

        q_old = self.q
        mu_old = self.MU
        pi_old = self.PI
        for i in range(max_epoch):
            start_time = time.time()
            # E step
            self.q = self.E_step(T=T, mu=self.MU, pi=self.PI)
            # M step
            self.PI, self.MU = self.M_step(T=T, q=self.q, n=self.N)
            q_delta = q_old - self.q
            mu_delta = mu_old - self.MU
            pi_delta = pi_old - self.PI
            q_old = self.q
            mu_old = self.MU
            pi_old = self.PI
            end_time = time.time()
            iter_duration = end_time - start_time
            print("Iteration %d cost %f seconds." % (i, iter_duration))
            # print("Delta q:")
            # print(q_delta)
            # print("Delta mu:")
            # print(mu_delta)
            # print("Delta pi:")
            # print(pi_delta)
        print("Training over.")

    def output(self, most_frequent_num, data):

        word = [[] for i in range(self.K)]
        f = open('./data/result.txt', 'a')
        print("For K = %d" % self.K)
        for i in range(self.K):
            print("The cluster %d\'s ratio is %f, most-frequent %d words are: " % (i + 1, self.PI[i], most_frequent_num))
            f.write("The cluster %d\'s ratio is %f, most-frequent %d words are: \n" % (i + 1, self.PI[i], most_frequent_num))
            list = []
            for j in range(self.W):
                list.append((self.MU[i, j], j))
            list.sort(key=lambda x: x[0], reverse=True)
            for j in range(most_frequent_num):
                frequent_word = data.vocabulary_dict[list[j][1]]
                word[i].append(frequent_word)
                print(frequent_word)
                f.write(frequent_word + " ")
            f.write("\n")
            print("")

        for i in range(self.K):
            for j in range(most_frequent_num):
                print(word[i][j], end=' ')
            print("")

    def predict(self, doc_size, T):
        label_pred = []
        for doc_id in range(doc_size):
            current_k = 0
            max_log_pk = -9999999999.0
            for i in range(self.K):
                log_pk = np.log(self.PI[i]) + np.dot(T[doc_id], np.log(self.MU[i]))
                if log_pk >= max_log_pk:
                    current_k = i
                    max_log_pk = log_pk
            label_pred.append(current_k)

        print("label_pred:")
        print(label_pred)
        print("label_pred size:")
        print(len(label_pred))

