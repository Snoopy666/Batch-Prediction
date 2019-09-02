#coding:utf-8
#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import os
import sys
import collections
import numpy as np
import pickle
import pandas as pd
import math
import tensorflow as tf
# from hanziconv import HanziConv
import json
import gensim
from gensim.models import KeyedVectors
from tensorflow import keras
from tensorflow.python import debug as tfdbg
import datetime
from datetime import timedelta
# import codecs
import time
from sklearn import metrics
import numpy
# import argparse
import logging

logging.basicConfig(level = logging.INFO, format = '%(asctime)s [INFO] %(message)s')

# parser = argparse.ArgumentParser()

# parser.add_argument('--sen_len', type=int, default=100)
# parser.add_argument('--doc_len', type=int, default=100)
# parser.add_argument('--train_file', type=str, default='./data/split_90/')
# parser.add_argument('--validation_file', type=str, default='./data/split_90/valid')
# # parser.add_argument('--model_dir', type=str, default='./runs/1532436443/checkpoints/')
# parser.add_argument('--model_dir', type=str, default='./checkpoints/')
# parser.add_argument('--epochs', type=int, default=15)
# parser.add_argument('--hidden', type=int, default=110)
# parser.add_argument('--lr', type=float, default=1e-4)

FLAGS=tf.app.flags.FLAGS
# 批量预测
tf.app.flags.DEFINE_string("delimiter","\001","delimiter of columns")
# tf.app.flags.DEFINE_integer("line_per_file",500000,"lines per result file")
tf.app.flags.DEFINE_float("valid_portion",0.125,"valid_portion")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("workers", 1, "work node num")
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_string("log_dir", "log/recreason_lm_log", "log directory")
tf.app.flags.DEFINE_string("output_dir", "./output", "predict data directory")

tf.app.flags.DEFINE_integer("batch_size", 10000, "Batch size for training/evaluating.") #训练用10, 1000
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("total_workers", 10, "work node num")
tf.app.flags.DEFINE_integer("line_per_file",50000,"lines per result file") # 500000
tf.app.flags.DEFINE_boolean("predict_incrementally",True,"if need to predict only the latest part") #是否需要增量预测
tf.app.flags.DEFINE_string("predict_target_file","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/dp_struccontent_summary_model_new/test","predict result")
tf.app.flags.DEFINE_string("training_data_path","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/summary_model_incre","path of traning data.")  # recreason_before_lm_model_incre
#tf.app.flags.DEFINE_string("predict_target_file","viewfs://hadoop-meituan/user/hive/warehouse/mart_dpsr.db/bert_comment_sample_info_seg_lm/test","predict result")
#tf.app.flags.DEFINE_string("training_data_path","viewfs://hadoop-meituan/user/hive/warehouse/mart_dpsr.db/bert_comment_sample_info_seg_bak","path of traning data.")  # bert数据过滤
tf.app.flags.DEFINE_string("ckpt_dir","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/summary/","checkpoint location for the model")
# tf.app.flags.DEFINE_string("vocabulary_word2index","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/recreason_lm/word2index1203_2.pkl","vocabulary_word2index")
# tf.app.flags.DEFINE_string("vocabulary_label2index","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/recreason_lm/label2index1203_2.pkl","vocabulary_label2index")
#tf.app.flags.DEFINE_string("emb_path","viewfs://hadoop-meituan/zw01nn11/warehouse/mart_dpsr.db/lantian/summary/model2.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("emb_path","model2","word2vec's vocabulary and vectors")


config_proto = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
config_proto.gpu_options.allow_growth = True
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


class Vocab:

    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        # model = KeyedVectors.load("./checkpoint/model2", mmap='r')
        if token not in self._token2index:
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    def feed_word(self, token):
        if token not in self._token2index:
            index = 0
            self._token2index[token] = index
            self._index2token.append(token)
        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    @property
    def token2index(self):
        return self._token2index

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        # print(token)
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)

def load_from_fileList(fileList, write_index, max_doc_length, max_sent_length):
    word_vocab = Vocab()
    word_vocab.feed('unk')
    word_vocab.feed('{')
    word_vocab.feed('}')
    word_vocab.feed(' ')
    # df = pd.read_excel('./data/split_90/5.xlsx', sheet_name='Sheet0')
    # headlineids = df['temp_summary_high_ctr_seg.headlineid'].tolist()
    # contents = df['temp_summary_high_ctr_seg.sentencesegs'].tolist()
    # contents_raw = df['temp_summary_high_ctr_seg.content'].tolist()
    # texts = df['temp_summary_high_ctr_seg.feed_text'].tolist()
    # df = pd.read_excel('./data/split_90/14.xlsx', sheet_name='Sheet1')
    # headlineids = df['headlineid'].tolist()
    # contents = df['sentencesegs'].tolist()
    # contents_raw = df['content'].tolist()
    # texts = df['feed_text'].tolist()
    print("start loading embeddings : " + str(datetime.datetime.now()))
    model = KeyedVectors.load(FLAGS.emb_path, mmap='r')
    print("loading done : " + str(datetime.datetime.now()))
    for word in model.vocab:
        word_vocab.feed(word)
    idset = {}
    lines_list = []
    index_of_line = 0
    shouldEnd = False
    for train_file in fileList:
        lines = tf.gfile.GFile(train_file).readlines()
        for i, line in enumerate(lines):
            index_of_line = index_of_line + 1
            if index_of_line > write_index and index_of_line <= write_index + FLAGS.line_per_file:
                line = line.replace("\n", "")
                lines_list.append(line)
                lines_splits = line.split(FLAGS.delimiter)
                headlineid = lines_splits[0]
                content = lines_splits[9]
                # content = content.decode('utf-8')
                idset[headlineid] = content
    if index_of_line < write_index+FLAGS.line_per_file:
        shouldEnd = True
    actual_max_doc_length = 0
    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    ids = collections.defaultdict(list)
    index = 0
    for headlineid in idset:
        fname = "test"
        word_doc = []
        label_doc = []
        content = idset[headlineid]
        sentences = content.split('䧪')
        index += 1
        for sentence in sentences:
            label = sentence.replace('@','')
            label_doc.append(label)
            sent_list = sentence.split("@")
            if len(sent_list) > max_sent_length - 2:  # space for 'start' and 'end' words
                sent_list = sent_list[:max_sent_length - 2]
            word_array = [word_vocab.feed_word(c) for c in ['{'] + sent_list + ['}']]
            word_doc.append(word_array)
        if len(word_doc) > max_doc_length:
            word_doc = word_doc[:max_doc_length]
            label_doc = label_doc[:max_doc_length]
            # print(actual_max_doc_length)
        actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

        word_tokens[fname].append(word_doc)
        labels[fname].append(label_doc)
        ids[fname].append(headlineid)

    actual_max_doc_length = max_doc_length
    print('total index', index)
    assert actual_max_doc_length <= max_doc_length

    print("max_doc_len:", actual_max_doc_length)
    print('actual longest document length is:', actual_max_doc_length)
    print('size of word vocabulary:', word_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    label_tensors = {}
    id_tensors = {}

    for fname in ('train', 'valid', 'test'):
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length],
                                       dtype=np.int64)
        # print(word_tensors)
        # label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int64)
	label_tensors[fname] = []
        id_tensors[fname] = np.zeros([len(ids[fname])], dtype=np.int64)

        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                # print(fname, i, j, len(word_array))
                word_tensors[fname][i][j][0:len(word_array)] = word_array
	
	for i, label_doc in enumerate(labels[fname]):
            tmp = []
            for j in range(max_doc_length):
                if j < len(label_doc):
                    tmp.append(label_doc[j])
                else:
                    tmp.append('')
            label_tensors[fname].append(tmp)
        label_tensors[fname] = np.asarray(label_tensors[fname])
        #for i, label_doc in enumerate(labels[fname]):
            # label_tensors[fname][i][0:len(label_doc)] = label_doc

        for i, id_doc in enumerate(ids[fname]):
            id_tensors[fname][i] = id_doc

    return word_vocab, word_tensors, actual_max_doc_length, label_tensors, id_tensors, shouldEnd

def load_embed(word_vocab):
    # model = gensim.models.Word2Vec.load("./checkpoint/model1")
    model = KeyedVectors.load(FLAGS.emb_path, mmap='r')
    # f = codecs.open("./oov_list_mm", "w", "utf-8")
    # print("space: ", model[" "])
    embed_list = []
    # uni_list = [0 for i in range(150)]
    uni_list = [0 for i in range(100)]
    embed_list.append(uni_list)
    # print (embed_list)
    vocab_dict = word_vocab._token2index
    vocab_sorted = sorted(vocab_dict.items(), key=lambda asd: asd[1], reverse=False)
    i = 0
    n_sum = 0
    w_dict = {}
    for item, index in vocab_sorted:
        # print("space:", item,index, ".")
        n_sum += 1
        if item not in model:
            # f.write(item)
            # f.write("\n")
            i += 1
            if item not in w_dict.keys():
                w_dict[item] = 1
            else:
                w_dict[item] += 1
            # embed_list.append(np.random.uniform(-0.25, 0.25, 150).round(6).tolist())
            embed_list.append(np.random.uniform(-0.25, 0.25, 100).round(6).tolist())

            # embed_list.append(uni_list)
        else:
            # print(list(model[item]))
            embed_list.append(list(model[item]))

    print("no in:", i)
    print("all:", n_sum)
    # f.close()
    # print(embed_list[0])
    embedding_array = np.array(embed_list, np.float32)
    print("no in:", i)
    print("all:", n_sum)
    # f = codecs.open("./fre_sum", "w", "utf-8")
    # for key, value in w_dict.items():
    #     f.write(key)
    #     f.write("\t")
    #     f.write(str(value))
    #     f.write("\n")
    # f.close()
    return embedding_array


# This function is used to get the word embedding of current words from
def get_embed(word_vocab):
    #model = gensim.models.Word2Vec.load("./checkpoint/model1")
    model = KeyedVectors.load("./checkpoint/model2", mmap='r')
    f = codecs.open("./oov_list_mm", "w", "utf-8")
    #print("space: ", model[" "])
    embed_list = []
    #uni_list = [0 for i in range(150)]
    uni_list = [0 for i in range(100)]
    embed_list.append(uni_list)
    #print (embed_list)
    vocab_dict = word_vocab._token2index
    vocab_sorted = sorted(vocab_dict.items(),key=lambda asd:asd[1], reverse=False)
    i = 0
    n_sum = 0
    w_dict = {}
    for item, index in vocab_sorted:
        #print("space:", item,index, ".")
        n_sum += 1
        if item not in model:
            f.write(item)
            f.write("\n")
            i += 1
            if item not in w_dict.keys():
                w_dict[item] = 1
            else:
                w_dict[item] += 1
            #embed_list.append(np.random.uniform(-0.25, 0.25, 150).round(6).tolist())
            embed_list.append(np.random.uniform(-0.25, 0.25, 100).round(6).tolist())
           
            #embed_list.append(uni_list)
        else:
            #print(list(model[item]))
            embed_list.append(list(model[item]))
    print("no in:", i)
    print("all:", n_sum)
    f.close()
    #print(embed_list[0])
    embedding_array = np.array(embed_list, np.float32)
    print("no in:", i)
    print("all:", n_sum)
    f = codecs.open("./fre_sum", "w", "utf-8")
    for key, value in w_dict.items():
        f.write(key)
        f.write("\t")
        f.write(str(value))
        f.write("\n")
    f.close()
    return embedding_array


class DataReader_v2:

    def __init__(self, word_tensor, label_tensor, id_tensor, batch_size):

        length = word_tensor.shape[0]
        #print (length)
        doc_length = word_tensor.shape[1]
        #print (doc_length)
        sent_length = word_tensor.shape[2]
        #print (sent_length)

        # round down length to whole number of slices

        clipped_length = int(length / batch_size) * batch_size
        #print (clipped_length)
        word_tensor = word_tensor[:clipped_length]
        label_tensor = label_tensor[:clipped_length]
        id_tensor = id_tensor[:clipped_length]
        print(word_tensor.shape)

        x_batches = word_tensor.reshape([batch_size, -1, doc_length, sent_length])
        #print(x_batches.shape)
        y_batches = label_tensor.reshape([batch_size, -1, doc_length])
        z_batches = id_tensor.reshape([batch_size, -1])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        #print(x_batches.shape)
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))
        z_batches = np.transpose(z_batches, axes=(1, 0))

        self._x_batches = list(x_batches)
        #print(len(self._x_batches))
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        #print (self.length)
        self.batch_size = batch_size
        self.max_sent_length = sent_length
        self._z_batches = list(z_batches)
        assert len(self._x_batches) == len(self._z_batches)
    def iter(self):

        for x, y, z in zip(self._x_batches, self._y_batches, self._z_batches):
            yield x, y, z

    def __len__(self):

        return len(self._x_batches)


# args = parser.parse_args()
# max_sen_length = args.sen_len
# max_doc_length = args.doc_len
max_sen_length = 20
max_doc_length = 60
tart_time = time.time()
logging.info('generate config')

start = datetime.datetime.now()
print("starting time : " + str(start))
workers = FLAGS.workers
list_name = tf.gfile.ListDirectory(FLAGS.training_data_path)
total_file_num = len(list_name)
print("list_name : " + str(list_name))
print("taskindex : " + str(FLAGS.task_index))
print("total file num : " + str(total_file_num))
cur_file_names = list_name[FLAGS.task_index:total_file_num:FLAGS.total_workers]
print("cur_file_names : " + str(cur_file_names))
fileList = [os.path.join(FLAGS.training_data_path, a) for a in cur_file_names]
print("fileList : " + str(fileList))
print("finish getting filelists : " + str(datetime.datetime.now()))

batch_size = 1
time1 = time.time()
write_index = 0
shouldEnd = False
sub_task_id = 0

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # saver = tf.compat.v1.train.import_meta_graph('./runs/1564993889/checkpoints/model-512.meta')
        # module_file = tf.train.latest_checkpoint("./runs/1564993889/" + 'checkpoints/')
        saver = tf.train.import_meta_graph(FLAGS.ckpt_dir + 'model_v2.meta')
        module_file = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
        saver.restore(sess, module_file)
        # f0 = codecs.open("summaries", "w", "utf-8")
        while not shouldEnd:
            input_x = graph.get_operation_by_name("inputs/x_input").outputs[0]
            predict = graph.get_operation_by_name("score_layer/prediction").outputs[0]
            resultlines = []
            loss_sum = 0
            count = 0
            # word_vocab, word_tensors, max_doc_length, label_tensors = \
            #     dr.load_test_v2(args.train_file, max_doc_length, max_sen_length)
            # word_vocab, word_tensors, max_doc_length, label_tensors, id_tensors = \
            #     dr.load_test_v3(args.train_file, max_doc_length, max_sen_length)
            print("current subtask id: " + str(sub_task_id))
            print("start loading files : " + str(datetime.datetime.now()))
            word_vocab, word_tensors, max_doc_length, label_tensors, id_tensors, shouldEnd = \
                load_from_fileList(fileList, write_index, max_doc_length, max_sen_length)
            # test_reader = dr.DataReader(word_tensors['test'], label_tensors['test'],
            #                             batch_size)
            test_reader = DataReader_v2(word_tensors['test'], label_tensors['test'],
                                       id_tensors['test'], batch_size)
            number_of_training_data = len(word_tensors)
            print("predict data : " + str(number_of_training_data))
            if number_of_training_data != 0:    
                for x, y, z in test_reader.iter():
                    count += 1
                    x = x[0]
                    y = y[0]
                    # print (x)
                    y_ = sess.run(predict, feed_dict = {input_x : x})
                    # ys_ = sess.run(predict, feed_dict = {input_x, xs})
                    # for x, y, y_ in zip(xs[0], ys[0], ys_[0]):
                    max_len = 0
                    for i, item in enumerate(x):
                        #print item
                        temp = 0
                        for sub_item in item:
                            #print(type(int(sub_item)))
                            if sub_item > 0:
                                temp += 1
                                #print temp
                        if temp == 0:
                            x = x[:i, :max_len]
                            y_ = y_[:i]
                            y = y[:i]
                            break
                        if temp > max_len:
                            max_len = temp
                    x = x[:, :max_len]

                    tmp_str = ''
                    actual_length = 0
                    index = 0
                    out_flag = 0
                    res = {}
                    top_sentence_index = numpy.argmax(y_)
                    left = top_sentence_index - 1
                    right = top_sentence_index + 1
                    # print(top_sentence_index)
                    # print(y_)
                    while len(tmp_str) < 30 and out_flag == 0:
                        y_[top_sentence_index] = 2
                        # cur_sentence = ''
                        # for word in x[top_sentence_index]:
                            #if word == 1:
                                # tmp_str += str(y_[top_sentence_index]) + '\t' + str(y[top_sentence_index]) + '\t'
                                #continue
                            #elif word == 2:
                                # tmp_str += '\n'
                                #break
                            #else:
                                #tmp_str += str(word_vocab.token(word))
                                #cur_sentence += str(word_vocab.token(word))
                        res[top_sentence_index] = y[top_sentence_index]
			tmp_str += y[top_sentence_index]
                        if len(tmp_str) < 30:
                            if left < 0 and right > len(x) - 1:
                                out_flag = 1
                            elif left < 0:
                                top_sentence_index = right
                                right += 1
                            elif right > len(x) - 1:
                                top_sentence_index = left
                                left -= 1
                            else:
                                if y_[right] >= y_[left]:
                                    top_sentence_index = right
                                    right += 1
                                else:
                                    top_sentence_index = left
                                    left -= 1
                            # if top_sentence_index == len(x) - 1:
                            #     tmp = ''
                            #     top_sentence_index -= 1
                            # else:
                            #     top_sentence_index += 1

                        else:
                            out_flag = 1
                    tmp_str = ''
                    for i, score in enumerate(y_):
                        if score == 2:
                            tmp_str += res[i]
                    if str(z)[1:-1] == '207410403':
                        print(tmp_str)
                        print(len(tmp_str))
                        print(y_)
                    resultlines.append(str(z)[1:-1] + FLAGS.delimiter + tmp_str + "\n")
                    # f0.write(str(z)[1:-1] + '\t' + tmp_str + '\n')
                # print(count)
                # print(sub_task_id)
                result_filename = FLAGS.predict_target_file + "_" + str(FLAGS.task_index) + "_" + str(sub_task_id)
                if FLAGS.predict_incrementally == True:
                    result_filename = result_filename + "_" + str(datetime.date.today())
                predict_target_file_f = tf.gfile.GFile(result_filename, 'w')
                for result in resultlines:
                    predict_target_file_f.write(result)
                predict_target_file_f.close()
                write_index = write_index + FLAGS.line_per_file
                sub_task_id = sub_task_id + 1
            else:
                break
        time_dif = timedelta(seconds=int(round(time.time() - time1)))
        print("Time usage:", time_dif)
