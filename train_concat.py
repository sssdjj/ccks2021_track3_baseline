#! -*- coding:utf-8 -*-
# score: 0.7786
"""bert每层捕获的信息不同，代表的语义粒度也不同，将不同粒度的信息拼接起来，然后送进CNN后做分类。
ret:
  https://arxiv.org/pdf/2008.06460.pdf
  """
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import Concatenate
from keras.layers import Dropout, Dense, Lambda
import json
from tqdm import tqdm
from sklearn.metrics import f1_score
from dgcnn import AttentionPooling1D, DGCNN

set_gelu('tanh')  # 切换gelu版本

maxlen = 130
batch_size = 64

# path = "/Users/sssdjj/bert_source/"
path = "data/"
config_path = path + 'nezha_base/bert_config.json'
checkpoint_path = path + 'nezha_base/model.ckpt-900000'
dict_path = path + 'nezha_base/vocab.txt'

labels = {f'不匹配': 0, f'完全匹配': 1, f'部分匹配': 2}
id2labels = {0: f'不匹配', 1: f'完全匹配', 2: f'部分匹配'}


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
                    E.append((i["query"], j["text"], labels[j["label"]]))
                else:
                    D.append((i["query"], j["text"], labels[j["label"]]))
    return D, E


def load_data1(filename):
    """加载数据
    单条格式：(文本1, 文本2, 标签id)
    """
    D = []
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


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    model="nezha",
    return_keras_model=False,
)

inputs = bert.inputs

outputs = []
x = bert.apply_embeddings(inputs)

for idx in range(bert.num_hidden_layers):
    x = bert.apply_main_layers(x, idx)
    output = Lambda(lambda x: x[:, 0:1])(x)
    outputs.append(output)

output = Concatenate(1)(outputs)

output = DGCNN(dilation_rate=1, dropout_rate=0.1)(output)
output = DGCNN(dilation_rate=2, dropout_rate=0.1)(output)
output = DGCNN(dilation_rate=2, dropout_rate=0.1)(output)
output = DGCNN(dilation_rate=1, dropout_rate=0.1)(output)

output = AttentionPooling1D()(output)
output = Dropout(0.5)(output)
output = Dense(
    units=len(labels), activation='softmax', kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def evaluate(data):
    f1 = 0.0
    y_trues = []
    y_preds = []
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
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
        for m, n in enumerate(i["candidate"]):
            token_ids, segment_ids = tokenizer.encode(
                i["query"], n["text"], maxlen=maxlen
            )
            y_pred = model.predict([[token_ids], [segment_ids]]).argmax(axis=1)
            i["candidate"][m]["label"] = id2labels[y_pred[0]]
        path.write(json.dumps(i, ensure_ascii=False) + "\n")
    path.close()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('train/best_model.weights')
        print(
            u'val_f1: %.5f, best_val_f1: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    # model.fit(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=10,
    #     callbacks=[evaluator]
    # )

    model.load_weights('data/best_model.weights')
    submit(test_data)

else:

    model.load_weights('best_model.weights')
