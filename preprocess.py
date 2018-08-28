'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: preprocess.py
@time: 2018/8/28 下午12:14
@desc: shanghaijiaotong university
'''
from collections import defaultdict
import sys
import os
import random

# There are 13 integer features and 26 categorical features
continous_features = range(1, 14)
categorial_features = range(14, 40)

# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class FeatureGenerator:
    def __init__(self, num_category_feature, num_continuous_feature):

        self.dicts = []
        self.num_category_feature = num_category_feature
        self.dicts_size = []
        for i in range(0, num_category_feature):
            self.dicts.append(defaultdict(int))

        self.num_continuous_feature = num_continuous_feature
        self.min = [sys.maxsize] * num_continuous_feature
        self.max = [-sys.maxsize] * num_continuous_feature

    def build(self, datafile, categorial_features, continous_features, cutoff=0):
        with open(datafile, 'r') as f:
            print('build for {}'.format(datafile))
            cnt = 1
            for line in f:
                if cnt % 100 == 0:
                    print('cnt: {}'.format(cnt))
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_category_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1

                for i in range(0, self.num_continuous_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val != '':
                            val = int(val)
                            if val > continous_clip[i]:
                                val = continous_clip[i]
                            self.min[i] = min(self.min[i], val)
                            self.max[i] = max(self.max[i], val)
                cnt += 1

            for i in range(0, self.num_category_feature):
                self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items())
                self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
                vocabs, _ = list(zip(*self.dicts[i]))
                self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
                self.dicts[i]['<unk>'] = 0
                self.dicts_size.append(len(vocabs))
                print('categorial_feature {}: vocabs size {}'.format(i, len(vocabs)))

    def gen(self, datadir, outdir):
        random.seed(123)
        with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
            with open(os.path.join(outdir, 'valid.txt'), 'w') as out_valid:
                with open(os.path.join(datadir, 'train_sub.txt'), 'r') as f:
                    print('gen for train and valid')
                    cnt = 1
                    for line in f:
                        if cnt % 100000 == 0:
                            print('cnt: {}'.format(cnt))
                        features = line.rstrip('\n').split('\t')
                        continous_vals = []
                        for i in range(0, self.num_continuous_feature):
                            val = features[continous_features[i]]
                            if val == '':
                                val = 0.0
                            val = float(val)
                            val = (val - self.min[i]) / (self.max[i] - self.min[i])
                            continous_vals.append("{0:.6f}".format(val).rstrip('0')
                                                  .rstrip('.'))

                        categorial_vals = []
                        for i in range(0, self.num_category_feature):
                            key = features[categorial_features[i]]
                            if key not in self.dicts[i]:
                                val = self.dicts[i]['<unk>']
                            else:
                                val = self.dicts[i][key]
                            categorial_vals.append(str(val))

                        label = features[0]
                        if random.randint(0, 9999) % 10 != 0:
                            out_train.write('\t'.join(
                                [label] + continous_vals + categorial_vals) + '\n')
                        else:
                            out_valid.write('\t'.join(
                                [label] + continous_vals + categorial_vals) + '\n')
                        cnt += 1


def preprocess(datadir, outdir):
    dists = FeatureGenerator(len(categorial_features), len(continous_features))
    dists.build(os.path.join(datadir, 'train_sub.txt'), categorial_features, continous_features, cutoff=200)
    dists.gen(datadir, outdir)


if __name__ == "__main__":
    preprocess('/Users/xiongfei/Downloads/dac', './data')
