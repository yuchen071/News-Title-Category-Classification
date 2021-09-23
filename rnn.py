import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence

import spacy
from collections import Counter
from torchtext.vocab import Vocab # torchtext 0.9.0

import csv
import math
import pandas as pd
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% manual seed
# import numpy as np
# import random
# seed = 86
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)

#%% Params
PATH_TRAIN = "news_data/train.csv"
PATH_TEST = "news_data/test.csv"

MAX_SEQ = 15    # set to 0 for auto max text len
HID_DIM = 100    # hidden dimension of the rnn
RNN_LAYERS = 2
DROP = 0.1

EPOCHS = 300
LR = 10e-5
BATCH_SIZE = 900
CLIP_GRAD = 1

SPLIT_PERCENT = 0.9	 # split percentage between training and validation dataset

#%% Dataset
class trainDataset(Dataset):
    def __init__(self, categories, label_list, titles):
        self.labels = [label_list.index(cate) for cate in categories]
        self.titles = titles

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, i):
        text, text_len =  self.titles[i]
        return (self.labels[i], text, text_len)

class testDataset(Dataset):
    def __init__(self, titles):
        self.titles = titles

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, i):
        return self.titles[i]

def collate_train(batch):
    label_list, text_list, len_list = [], [], []
    for (label, text, text_len) in batch:
        len_list.append(text_len)
        label_list.append(label)
        text = torch.tensor(text, dtype=torch.int64)
        text_list.append(text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.stack(text_list)
    return label_list.to(device), text_list.to(device), len_list

def collate_test(batch):
    text_list, len_list = [], []
    for (text, text_len) in batch:
        len_list.append(text_len)
        text = torch.tensor(text, dtype=torch.int64)
        text_list.append(text)
    text_list = torch.stack(text_list)
    return text_list.to(device), len_list


#%% model
class TextRNN(nn.Module):
    def __init__(self, embed_weights, hidden_dim, num_class, num_layers=2, dropout=0.6):
        super(TextRNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embed_weights).requires_grad_(True)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=embed_weights.shape[1],
                            hidden_size=hidden_dim,
                            bidirectional = True,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout=dropout
                            )

        self.fc_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Tanh(),
            # nn.BatchNorm1d(hidden_dim*2),
            nn.Linear(hidden_dim*2, num_class),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(hidden_dim, num_class),
            )

    def forward(self, text, seq_len):
        x = self.embed(text)                                # (batch_size, max_seq_len, input_size)

        x = pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        x_o, x_h = self.rnn(x)  # gru
        x = torch.cat((x_h[-2,:,:], x_h[-1,:,:]), dim = 1)  # (batch_size, hidden_size)
        # x = x[-1,:,:] # if not bidirectional
        x = self.fc_layer(x)
        return x

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, bsz, self.hidden_dim)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

    elif type(m) in [nn.GRU, nn.LSTM]:
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.uniform_(param.data, -0.1, 0.1)

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

#%% train
def train(df_train, df_test, label_list):

    # text preprocess
    print("Text Preprocessing...", end="")
    spacy_en = spacy.load('en_core_web_sm', disable=['parser'])

    def tokenizer(title, filter_ent):
        title = title.strip()
        title_doc = spacy_en(title)

        with title_doc.retokenize() as retokenizer:
            for ent in title_doc.ents:
                if ent.label_ in filter_ent:
                    # retokenizer.merge(title_doc[ent.start:ent.end], attrs={"LOWER": ent.label_})
                    retokenizer.merge(title_doc[ent.start:ent.end], attrs={"LEMMA": ent.label_})

        # title_tok = [word.lower_ for word in title_doc if not word.is_punct]
        title_tok = [word.lemma_ for word in title_doc if not word.is_punct and not word.is_stop]
        title_tok = [word.lower() if word not in filter_entity else word for word in title_tok]
        return title_tok

    filter_entity = ["MONEY", "TIME", "PERCENT", "DATE"]
    train_tok = [tokenizer(title, filter_entity) for title in df_train["Title"]]
    test_tok = [tokenizer(title, filter_entity) for title in df_test["Title"]]
    # specials = ["<pad>", "<unk>", "<sos>"]
    specials = ["<pad>", "<unk>"]
    specials.extend(filter_entity)

    counter = Counter()
    max_seq = MAX_SEQ
    for text_tok in train_tok:
        counter.update(text_tok)
        if MAX_SEQ == 0:
            if len(text_tok) > max_seq:
                max_seq = len(text_tok)

    for text_tok in test_tok:
        counter.update(text_tok)

    vocab = Vocab(counter, min_freq=1, vectors='glove.6B.300d', specials=specials)
    embedding = vocab.vectors

    def text_pipeline(text_tok, max_seq):
        text_len = len(text_tok)
        if max_seq > text_len:
            # pad text seq with <pad>
            # text2 = ['<sos>'] + text_tok + ['<pad>'] * (max_seq - text_len - 1)
            text2 = text_tok + ['<pad>'] * (max_seq - text_len)
        else:
            text2 = text_tok[:max_seq]
            text_len = len(text2)
        return [vocab[token] for token in text2], text_len

    train_list = [text_pipeline(text_tok, max_seq) for text_tok in train_tok]

    print("Done!")

    # make dataset and split to train and validation
    data_train = trainDataset(df_train['Category'], label_list, train_list)
    num_train = int(len(data_train) * SPLIT_PERCENT)
    data_train, data_vali = random_split(data_train, [num_train, len(data_train) - num_train])

    print("Train data: %d, Validation data: %d, Train batches: %.1f\n" %  \
          (len(data_train), len(data_vali), len(data_train)/BATCH_SIZE))

    trainloader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_train)
    validloader = DataLoader(data_vali, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_train)

    # init model
    num_class = len(label_list)
    model = TextRNN(embed_weights=embedding,
                     hidden_dim=HID_DIM,
                     num_class=num_class,
                     num_layers=RNN_LAYERS,
                     dropout=DROP
                     )
    model.apply(init_weights)
    model.init_hidden(BATCH_SIZE)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineWarmupScheduler(optimizer=optimizer,
                                      warmup=5,
                                      max_iters=EPOCHS)

    # train
    print("Training...")
    sleep(0.3)
    train_loss_hist, train_acc_hist = [], []
    valid_loss_hist, valid_acc_hist = [], []

    t = tqdm(range(EPOCHS), ncols=200, bar_format='{l_bar}{bar:15}{r_bar}{bar:-10b}', unit=' epoch')
    for epoch in t:
        model.train()

        train_loss, train_acc, train_count = 0, 0, 0
        batch_acc, batch_count = 0, 0
        for batch_id, (label, text, seq_len) in enumerate(trainloader):
            optimizer.zero_grad()

            out = model(text, seq_len)
            loss = criterion(out, label)
            loss.backward()
            clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

            batch_acc = (out.argmax(1) == label).sum().item()
            batch_count = label.size(0)

            train_loss += loss.item()
            train_acc += batch_acc
            train_count += batch_count

        scheduler.step()

        model.eval()
        valid_loss, valid_acc, valid_count = 0, 0, 0
        with torch.no_grad():
            for batch_id, (label, text, seq_len) in enumerate(validloader):
                out = model(text, seq_len)
                loss2 = criterion(out, label)

                valid_loss += loss2.item()
                valid_acc += (out.argmax(1) == label).sum().item()
                valid_count += label.size(0)

        train_loss = train_loss/train_count
        train_acc = train_acc/train_count*100
        valid_loss = valid_loss/valid_count
        valid_acc = valid_acc/valid_count*100

        tl_post = "%2.5f" % (train_loss)
        ta_post = "%3.3f" % (train_acc)
        vl_post = "%2.5f" % (valid_loss)
        va_post = "%3.3f" % (valid_acc)
        t.set_postfix({"T_Loss": tl_post, "T_Acc": ta_post, "V_Loss": vl_post, "V_Acc": va_post})
        t.update(0)

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        valid_loss_hist.append(valid_loss)
        valid_acc_hist.append(valid_acc)

    # plot
    plt.figure()
    plt.plot(train_acc_hist, label="Train")
    plt.plot(valid_acc_hist, label="Valid")
    plt.title("Average Accuracy History")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()

    plt.figure()
    plt.plot(train_loss_hist, label="Train")
    plt.plot(valid_loss_hist, label="Valid")
    plt.title("Average Loss History")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()

    # eval
    print("Eval...", end="")
    # test_tok = [tokenizer(title, filter_entity) for title in df_test["Title"]]
    test_list = [text_pipeline(text_tok, max_seq) for text_tok in test_tok]
    data_test = testDataset(test_list)
    testloader = DataLoader(data_test, batch_size=10, shuffle=False, collate_fn=collate_test)

    sleep(0.5)
    model.eval()
    ans_list = []
    with torch.no_grad():
        for batch_id, (text, seq_len) in enumerate(testloader):
            out = model(text, seq_len)
            ans_list.extend(out.argmax(1).tolist())

    # print(len(ans_list))
    ans_labeled = [label_list[idx] for idx in ans_list]
    id_list = list(range(len(ans_list)))

    with open("output_RNN.csv", "w", newline="") as fp:
        fp.write("Id,Category\n")
        c_writer = csv.writer(fp)
        c_writer.writerows(zip(id_list, ans_labeled))

    print("Done!")


#%% main
if __name__ == "__main__":
    df_train = pd.read_csv(PATH_TRAIN)
    df_test = pd.read_csv(PATH_TEST)
    label_list = sorted(list(set(df_train['Category'])))

    train(df_train, df_test, label_list)