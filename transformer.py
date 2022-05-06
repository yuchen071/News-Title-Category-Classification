#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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

#%% Params
PATH_TRAIN = "news_data/train 2.csv"
PATH_TEST = "news_data/test 2.csv"
# DATA = "Title"
DATA = "Description"

# max_seq clips text tok, set max_seq to 0 for auto max text len
# num_head has to be dividable to embed_dim (300)
# without scheduler, lr = 1e-4 optimal, 1e-3 and higher will not train well
MAX_SEQ = 0     # Just needs to be longer than sequence
NUM_HID = 600   # 
NUM_HEAD = 10   #!
NUM_LAYERS = 2  #! 2~3, over 4 will crash
DROPOUT = 0.5   #! 0.1~0.3

EPOCHS = 300
LR = 1e-4
BATCH_SIZE = 500
CLIP_GRAD = 1

#%% Dataset
class trainDataset(Dataset):
    def __init__(self, categories, label_list, titles):
        self.labels = [label_list.index(cat) for cat in categories]
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
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerNet(nn.Module):
    def __init__(self, embed_pretrain, padding_idx, max_sequence, n_hid, n_class, n_head=6, n_layers=2, dropout=0.5):
        """
        n_tokens: vocab size
        embed_dim: size of vector for each token
        encoder: embedding matrix with size (n_tokens x embed_dim), can be imported from vocab

        n_class: number of classes to output
        n_head: number of attention heads for trans_encode
        n_hid: number of hidden nodes in NN part of trans_encode
        n_layers: number of trans_encoderlayer in trans_encode
        """
        super(TransformerNet, self).__init__()
        self.encoder = nn.Embedding.from_pretrained(embed_pretrain).requires_grad_(True)
        self.embed_dim = embed_pretrain.shape[1]
        self.n_tokens = embed_pretrain.shape[0]
        self.pad_idx = padding_idx

        self.pos_enc = PositionalEncoding(self.embed_dim, dropout)

        encoder_layers = TransformerEncoderLayer(self.embed_dim, n_head, n_hid, dropout)
        self.trans_enc = TransformerEncoder(encoder_layers, n_layers)

        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(self.embed_dim, self.embed_dim//4),
            # nn.BatchNorm1d(self.embed_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim//4, n_class),

            # nn.Linear(self.embed_dim, n_class),
            )

    def forward(self, x):                                   # input: (batch, seq)
        # sm = self.generate_square_subsequent_mask(x.size(1)).to(device)
        km = self.get_padding_mask(x)

        x = torch.transpose(x, 0, 1)                        # (seq, batch)
        x = self.encoder(x) * math.sqrt(self.embed_dim)     # (seq, batch, emb_dim)
        x = self.pos_enc(x)                                 # (seq, batch, emb_dim)

        # x = self.trans_enc(x, mask=sm)                      # (seq, batch, emb_dim)
        x = self.trans_enc(x, src_key_padding_mask=km)
        # x = self.trans_enc(x)

        # pure fc
        # kmt = torch.transpose(km,0,1).unsqueeze(-1)
        # x = x*kmt
        # x = x[0]
        x = x.mean(dim=0)                                   # (batch, emb_dim)
        x = self.fc1(x)                                     # (batch, n_class)

        return x

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_padding_mask(self, text):
        mask = (text == self.pad_idx).to(device)  #! (batch_size, word_pad_len)
        return mask

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

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

# =============================================================================
#     text preprocess
# =============================================================================
    print("Text Preprocessing...", end="")
    spacy_en = spacy.load('en_core_web_sm', disable=['parser'])

    def tokenizer(title, filter_ent):
        title = title.strip()
        title_doc = spacy_en(title)

        # method 1
        with title_doc.retokenize() as retokenizer:
            for ent in title_doc.ents:
                if ent.label_ in filter_ent:
                    retokenizer.merge(title_doc[ent.start:ent.end], attrs={"LOWER": ent.label_})
        title_tok = [word.lower_ for word in title_doc if not word.is_punct]

        # # method 2
        # with title_doc.retokenize() as retokenizer:
        #     for ent in title_doc.ents:
        #         if ent.label_ in filter_ent:
        #             retokenizer.merge(title_doc[ent.start:ent.end], attrs={"LEMMA": ent.label_})
        # title_tok = [word.lemma_ for word in title_doc if not word.is_punct and not word.is_stop]
        # title_tok = [word.lower() if word not in filter_entity else word for word in title_tok]
        
        return title_tok


    filter_entity = ["MONEY", "TIME", "PERCENT", "DATE"]
    train_tok = [tokenizer(title, filter_entity) for title in df_train[DATA]]
    test_tok = [tokenizer(title, filter_entity) for title in df_test[DATA]]
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
    if MAX_SEQ == 0:
        max_seq += 10

    for text_tok in test_tok:
        counter.update(text_tok)

    vocab = Vocab(counter, min_freq=1, vectors='glove.6B.300d', specials=specials)
    pad_idx = vocab["<pad>"]
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

# =============================================================================
#     make dataset and split to train and validation
# =============================================================================
    data_train = trainDataset(df_train['Category'], label_list, train_list)

    print(f"Train on: {DATA}")
    print("Train data: %d, Train batches: %.1f, max seq len: %d\n" %  \
          (len(data_train), len(data_train)/BATCH_SIZE, max_seq))

    trainloader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_train)

    # init model
    num_class = len(label_list)
    model = TransformerNet(
        embedding,
        padding_idx = pad_idx,
        max_sequence = max_seq,
        n_hid = NUM_HID,
        n_class = num_class,
        n_head = NUM_HEAD,
        n_layers = NUM_LAYERS,
        dropout = DROPOUT
        )
    model.apply(init_weights)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # ref: attention is all u need
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=1e-6,
                            betas=(0.9, 0.98))
    scheduler = CosineWarmupScheduler(optimizer=optimizer,
                                      warmup=10,
                                      max_iters=EPOCHS)

# =============================================================================
#     train
# =============================================================================
    print("Training...")
    sleep(0.3)
    train_loss_hist, train_acc_hist = [], []

    t = tqdm(range(EPOCHS), ncols=200, bar_format='{l_bar}{bar:15}{r_bar}{bar:-10b}', unit='epoch')
    model.train()
    for epoch in t:
        train_loss, train_acc, train_count = 0, 0, 0
        batch_acc, batch_count = 0, 0
        for batch_id, (label, text, seq_len) in enumerate(trainloader):
            optimizer.zero_grad()

            out = model(text)
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

        train_loss = train_loss/train_count
        train_acc = train_acc/train_count*100

        tl_post = "%2.5f" % (train_loss)
        ta_post = "%3.3f" % (train_acc)
        t.set_postfix({"T_Loss": tl_post, "T_Acc": ta_post})
        t.update(0)

        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

    # plot
    plt.figure()
    plt.plot(train_acc_hist, label="Train")
    plt.title("Average Accuracy History")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()

    plt.figure()
    plt.plot(train_loss_hist, label="Train")
    plt.title("Average Loss History")
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()

# =============================================================================
#     eval
# =============================================================================
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
            out = model(text)
            ans_list.extend(out.argmax(1).tolist())

    # print(len(ans_list))
    ans_labeled = [label_list[idx] for idx in ans_list]
    id_list = list(range(len(ans_list)))

    with open("output_transformer.csv", "w", newline="") as fp:
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

    if str(device) == 'cuda':
        torch.cuda.empty_cache()

