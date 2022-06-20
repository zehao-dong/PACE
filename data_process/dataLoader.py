import random
import torch
import torch.utils.data as Data
import numpy as np

class DataLoader(Data.Dataset):
    def __init__(self, part, config, LOG, prepare_func=None):
        self.part = part
        self.data_file = eval("config."+part+"_data")
        self.pair_file = eval("config."+part+"_pair")
        self.batch_size = config.batch_size
        self.config = config
        self.log = LOG
        self.prepare_func = prepare_func

        self.n_data = 0
        self.Data = []
        self.n_pair = []
        self.Pair = []
        self.n_batch = 0
        self.Batch = []
        self.Batch_idx = []

    def shuffle(self):
        self.log.log('Start Shuffling')

        data = self.Pair
        number = self.n_pair

        shuffle_Index = list(range(number))
        random.shuffle(shuffle_Index)

        data_new = [data[shuffle_Index[Index]] for Index in range(number)]

        self.Pair = data_new
        self.log.log('Finish Shuffling')

    def genBatches(self):
        batch_size = self.batch_size
        data = self.Pair
        number = self.n_pair
        n_dim = len(data[0])

        number_batch = number // batch_size
        batches = []

        for bid in range(number_batch):
            batch_i = []
            for j in range(n_dim):
                data_j = [item[j] for item in data[bid * batch_size: (bid + 1) * batch_size]]
                batch_i.append(data_j)
            batches.append(batch_i)

        if (number_batch * batch_size < number):
            if (number - number_batch * batch_size >= torch.cuda.device_count()):
                batch_i = []
                for j in range(n_dim):
                    data_j = [item[j] for item in data[number_batch * batch_size:]]
                    batch_i.append(data_j)
                batches.append(batch_i)
                number_batch += 1

        self.n_batch = number_batch
        self.Batch = batches
        self.Batch_idx = list(range(self.n_batch))

    def load(self):
        pass

    def afterLoad(self):
        self.log.log("Data Shuffle")
        self.shuffle()
        self.log.log("Generate Batch")
        self.genBatches()

    def batchShuffle(self):
        random.shuffle(self.Batch_idx)

    def __len__(self):
        return self.n_batch

    def __getitem__(self, index):
        batch_of_index = self.Batch[self.Batch_idx[index]][0]
        batch_of_data = []

        for i, pair in enumerate(batch_of_index):
            index_x, index_y = pair
            dataX = self.Data[index_x]
            dataY = self.Data[index_y]
            batch_of_data.append([dataX, dataY])

        if self.prepare_func is None:
            return batch_of_data

        return self.prepare_func(batch_of_data, self.config)

class Dataset(DataLoader):
    def __init__(self, part, config, LOG, prepare_func=None):
        super(Dataset, self).__init__(part, config, LOG, prepare_func)
        self.log.log('Building dataset %s from orignial text documents' % self.part)
        self.n_data, self.Data, self.n_pair, self.Pair = self.load()
        self.log.log('Finish Loading dataset %s' % self.part)
        self.afterLoad()

    def load(self):
        data_ = torch.load(self.data_file)
        pair_ = torch.load(self.pair_file)

        data = []
        for i in range(len(data_)):
            R = data_[i]['adj']
            X = np.argmax(np.asarray(data_[i]['ops']), axis=-1)
            data.append([X, R])

        pair = []
        for i in range(len(pair_)):
            if isinstance(pair_[i], set):
                t = list(pair_[i])[0]
                pair.append([(t[0], t[1])])
            else:
                pair.append([(pair_[i][0], pair_[i][1])])

        return len(data), data, len(pair), pair