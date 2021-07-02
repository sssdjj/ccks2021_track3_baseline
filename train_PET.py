#! -*- coding:utf-8 -*-
# attern-Exploiting Training(PET) score:0.7812

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import json

num_classes = 3
maxlen = 130
batch_size = 32

path = "data/"
config_path = path + 'nezha_base/bert_config.json'
checkpoint_path = path + 'nezha_base/model.ckpt-900000'
dict_path = path + 'nezha_base/vocab.txt'

labels = {f'不匹配': 0, f'完全匹配': 1, f'部分匹配': 2}


def load_data(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    E = []
    with open(filename, encoding='utf-8') as f:
        for k, i in enumerate(f):
            i = json.loads(i)
            for j in i["candidate"]:
                if k % 10 == 0:
                    E.append((i["query"], j["text"], labels.get(j["label"], 3)))
                else:
                    D.append((i["query"], j["text"], labels.get(j["label"], 3)))
    return D, E


def load_data1(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
    E = []
    with open(filename, encoding='utf-8') as f:
        for k, i in enumerate(f):
            i = json.loads(i)
            D.append(i)

    return D


# 加载数据集
train_data, valid_data = load_data('data/Xeon3NLP_round1_train_20210524.txt')

test_data = load_data1('data/Xeon3NLP_round1_test_20210524.txt')



# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
prefix = u'很匹配。'
mask_idx = 1  # 包含 cls 【很或不】的位置是 1
pos_id = tokenizer.token_to_id(u'很')
neg_id = tokenizer.token_to_id(u'不')
zhong_id = tokenizer.token_to_id(u'半')
id2labels = {0: f'不匹配', 1: f'完全匹配', 2: f'部分匹配'}

token2id = {neg_id: 0, pos_id: 1, zhong_id: 2}


def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:  # 12%
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:  # 1.5%
            source.append(t)
            target.append(t)
        elif r < 0.15:  # 1.5%
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:  # 85%
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):  # 调用 forfit()时 默认设置 random为True
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            if label != 3:
                text1 = prefix + text1
            token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if label == 0:  # 如果样本中 有 标注数据 ，被预测的 部分一定 要mask 就是 【很或不】 字处
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neg_id
            elif label == 1:  # 如果样本中 有 标注数据 ，被预测的 部分一定 要mask 就是 【很或不】 字处
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = pos_id
            elif label == 2:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = zhong_id
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [
                          batch_token_ids, batch_segment_ids, batch_output_ids
                      ], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True, model="nezha"
)

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
outputs = CrossEntropy(1)([y_in, model.output])

train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(optimizer=Adam(1e-5))
train_model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_f1 = 0.
        self.flag = 0

    def on_epoch_end(self, epoch, logs=None):
        f1 = evaluate(valid_generator)
        if f1 > self.best_f1:
            self.best_f1 = f1
            model.save_weights('train/best_model.weights')
        print(
            u' f1: %.5f, best_f1: %.5f\n' %
            (f1, self.best_f1)
        )


# TP: 将正类预测为正类数 40
# FN: 将正类预测为负类数 20
# FP: 将负类预测为正类数 10
# TN: 将负类预测为负类数 30
#
# def evaluate(data):
#     TP, FN, FP, TN = 0.0, 0.0, 0.0, 0.0
#     for x_true, _ in tqdm(data):
#         x_true, y_true = x_true[:2], x_true[2]
#         y_pred = model.predict(x_true)
#         # 获取 每一个样本的 第一个位置 （也就是 _满意，里边_的位置），提取 ”很“和”不“ 的概率
#         # 哪个大，就属于哪一类  很-> 正向情绪，不-> 负向情绪
#         y_pred = y_pred[:, mask_idx, [neg_id, pos_id, zhong_id]].argmax(axis=1)
#         y_true = (y_true[:, mask_idx] == pos_id).astype(int)  # list类型
#
#         TP += ((y_pred == y_true) == (y_true == 1)).sum()
#         FN += (y_true == (y_pred == 0)).sum()
#         FP += (y_pred == (y_true == 0)).sum()
#         TN += ((y_pred == y_true) == (y_true == 0)).sum()
#     # acc ,recall, f1
#     acc = (TP + TN) / (TP + FN + FP + TN)
#     recall = TP / (TP + FN)
#     f1 = 2 * (acc * recall) / (acc + recall)
#     return f1, acc, recall


def evaluate(data):
    y_trues = []
    y_preds = []
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        y_pred = y_pred[:, mask_idx, [neg_id, pos_id, zhong_id]].argmax(axis=1)
        y_true = y_true[:, mask_idx]
        y_true = list(map(lambda i: token2id[i], y_true))
        y_trues.extend(y_true)
        y_preds.extend(y_pred)
    y0, p0, y1, p1, y2, p2 = [], [], [], [], [], []
    for i, j in zip(y_trues, y_preds):
        if i == 0:
            y0.append(i)
            p0.append(j)
        elif i == 1:
            y1.append(i)
            p1.append(j)
        else:
            y2.append(i)
            p2.append(j)
    f1 = (f1_score(y0, p0, average='micro') + f1_score(y1, p1, average='micro') + f1_score(y2, p2,
                                                                                           average='micro')) / 3.0
    return f1


def submit(data):
    path = open("train/移花接木_addr_match_runid.txt", "w+", encoding='utf-8')
    for i in tqdm(data):
        for m, n in tqdm(enumerate(i["candidate"])):
            token_ids, segment_ids = tokenizer.encode(
                prefix + i["query"], n["text"], maxlen=maxlen
            )
            token_ids[mask_idx] = tokenizer._token_mask_id
            token_ids = sequence_padding([token_ids])
            segment_ids = sequence_padding([segment_ids])
            y_pred = model.predict([token_ids, segment_ids])
            y_pred = y_pred[:, mask_idx, [neg_id, pos_id, zhong_id]].argmax(axis=1)

            i["candidate"][m]["label"] = id2labels[y_pred[0]]
        path.write(json.dumps(i, ensure_ascii=False) + "\n")
    path.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    # train_model.fit_generator(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=10,
    #     callbacks=[evaluator]
    # )
    model.load_weights('data/best_model.weights')
    submit(test_data)
