import gzip
import csv
import collections
import random
import numpy as np

Dataset = collections.namedtuple('Dataset', ['data', 'target'])

"""Based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py#L59"""
"""Returns (2d array of data, 1d array of targets)"""
def load_gzipped_csv_without_header(filename,
                                    target_dtype,
                                    features_dtype,
                                    target_column=-1,
                                    max_rows=-1):
    with gzip.open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        data, target = [], []
        for row in data_file:
            if len(row) > 0:
                target.append(row.pop(target_column))
                data.append(np.asarray(row, dtype=features_dtype))
                if len(target) % 1000 == 0:
                    print 'loaded %d rows' % len(target)
            if max_rows != -1 and len(target) == max_rows:
                break

    target = np.array(target, dtype=target_dtype)
    data = np.array(data)
    return (data, target)

"""Streams tuples of (1d array data row, scalar target)"""
def stream_gzipped_csv_without_header(filename,
                                      target_dtype,
                                      features_dtype,
                                      target_column=-1):
    with gzip.open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        for row in data_file:
            if len(row) > 0:
                target = target_dtype(row.pop(target_column))
                data = np.asarray(row, dtype=features_dtype)
                yield (data, target)

def infinitely_stream_gzipped_csv_without_header(filename,
                                      target_dtype,
                                      features_dtype,
                                      target_column=-1):
    while True:
        for x in stream_gzipped_csv_without_header(filename, target_dtype, features_dtype, target_column):
            yield x

def next_training_batch(training_iter, batch_size):
    features = []
    targets = []
    for _ in range(batch_size):
        fs, target = training_iter.next()
        features.append(fs)
        targets.append(target)
    return (features, targets)

def split_test_train_gzipped(input_filename,
                             output_test_filename,
                             output_train_filename,
                             portion=0.1, seed=9001):
    random.seed(seed)
    with gzip.open(input_filename, 'r') as input:
        with gzip.open(output_test_filename, 'w') as out_test:
            with gzip.open(output_train_filename, 'w') as out_train:
                for line in input:
                    if random.random() < portion:
                        out_test.write(line)
                    else:
                        out_train.write(line)
