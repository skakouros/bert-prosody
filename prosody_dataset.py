import os
import sys
import random
import math
from torch.utils import data
from transformers import AutoTokenizer
import torch
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path


class Dataset(data.Dataset):
    def __init__(self, tagged_sents, tag2index, pos2index, config, word_to_embid=None):
        sents, tags_li, values_li, pos_li = [], [], [], [] # list of lists
        self.config = config

        for sent in tqdm(tagged_sents):
            words = [word_tag[0] for word_tag in sent]
            tags = [word_tag[1] for word_tag in sent]
            values = [word_tag[3] for word_tag in sent] #+++HANDE
            pos = [word_tag[5] for word_tag in sent] if config.use_pos else []

            if self.config.model != 'LSTM' and self.config.model != 'BiLSTM':
                sents.append(["[CLS]"] + words + ["[SEP]"])
                tags_li.append(["<pad>"] + tags + ["<pad>"])
                values_li.append(["<pad>"] + values + ["<pad>"])
                pos_li.append(["<pad>"] + pos + ["<pad>"])
            else:
                sents.append(words)
                tags_li.append(tags)
                values_li.append(values)
                pos_li.append(pos)

        if config.use_pos:
            assert False not in [len(s)==len(p) for s,p in zip(sents,pos_li)], 'Mismatch POS<->Token length!'

        self.sents, self.tags_li, self.values_li, self.pos_li = sents, tags_li, values_li, pos_li
        if self.config.model == 'BertUncased':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            # self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        # create mappings
        self.tag2index = tag2index
        self.word_to_embid = word_to_embid
        self.pos2index = pos2index

    def __len__(self):
        return len(self.sents)

    def convert_tokens_to_emb_ids(self, tokens):
        UNK_id = self.word_to_embid.get('UNK')
        return [self.word_to_embid.get(token, UNK_id) for token in tokens]

    def __getitem__(self, id):
        words, tags, values_li, pos_li = self.sents[id], self.tags_li[id], self.values_li[id], self.pos_li[id] # words, tags, values, pos: string list

        x, y, values, k = [], [], [], [] # list of ids
        is_main_piece = [] # only score the main piece of each word
        for w, t, v, p in zip(words, tags, values_li, pos_li):
            if self.config.model in ['LSTM', 'BiLSTM', 'LSTMRegression']:
                tokens = [w]
                xx = self.convert_tokens_to_emb_ids(tokens)
            else:
                # tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                xx = self.tokenizer.convert_tokens_to_ids(tokens)

            # CHECK THIS AND CORRECT IF NECESSARY
            #t = [t] + ["1"] * (len(tokens) - 1)  # <PAD>: no decision
            t = [t] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [self.tag2index[each] for each in t]  # (T,)
            p = [p] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            kk = [self.pos2index[each] for each in p] if self.config.use_pos else [] # (T,)

            head = [1] + [0]*(len(tokens) - 1) # identify the main piece of each word

            x.extend(xx)
            is_main_piece.extend(head)
            y.extend(yy)
            k.extend(kk)

        assert len(x) == len(y) == len(is_main_piece), "len(x)={}, len(y)={}, len(is_main_piece)={}".format(len(x), len(y), len(is_main_piece))
        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        pos = " ".join(pos_li) if self.config.use_pos else []

        if self.config.log_values:
            # Use log-values to remove affects of 0-skewed value distribution
            values = [np.log(np.log(float(v) + 1)+1) if v not in ['<pad>','NA'] else self.config.invalid_set_to for v in values_li]
        else:
            values = [float(v) if v not in ['<pad>', 'NA'] else self.config.invalid_set_to for v in values_li]

        return words, x, is_main_piece, tags, y, seqlen, values, k, pos, self.config.invalid_set_to


def add_pos_to_dataset(config) -> None:
    # dependency import
    spacy = __import__("spacy")
    # init vars
    splits = dict()
    words = []
    all_sents = []
    new_file_suffix = '_pos'
    # init taggers (note: tagger separate calls make runtime slower)
    # (separate taggers were created with the assumption that one or two will be used for each
    # processing round)
    nlp = spacy.load("en_core_web_sm")
    pos_coarse_grained_tagger = lambda x: [nlp(token.strip("'#"))[0].pos_ if len(token)>1 else nlp(token)[0].pos_ for token in x]
    pos_fine_grained_tagger = lambda x: [nlp(token.strip("'#"))[0].tag_ if len(token)>1 else nlp(token)[0].tag_ for token in x]
    lemma_tagger = lambda x: [nlp(token.strip("'#"))[0].lemma_ if len(token)>1 else nlp(token)[0].lemma_ for token in x]
    stop_tagger = lambda x: [nlp(token.strip("'#"))[0].is_stop if len(token)>1 else nlp(token)[0].is_stop for token in x]
    dep_tagger = lambda x: [nlp(token.strip("'#"))[0].dep_ if len(token)>1 else nlp(token)[0].dep_ for token in x]
    # tag lists
    pos_tag_list_coarse = ['ADJ','ADP','ADV','AUX','CONJ','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X','SPACE']
    pos_tag_list_fine = ['AFX', 'JJ', 'JJR', 'JJS', 'PDT', 'PRP$', 'WDT', 'WP$', 'IN', 'EX', 'RB', 'RBR', 'RBS', 'WRB', 'CC', 'DT', 'NN', 'UH', 'NNS', 'WP', 'CD', 'POS', 'RP', 'TO', 'PRP', 'NNP', 'NNPS', '-LRB-', '-RRB-', ',', ':', '.', '”', '“”', '“', 'HYPH', 'LS', 'NFP', '#', '$', 'SYM', 'BES', 'HVS', 'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'ADD', 'FW', 'GW', 'XX', '_SP', 'NIL']

    # iterate through dataset files and write to file
    for split in ['train', 'dev', 'test']:
        filename = config.train_set if split == 'train' else split
        print(f'\nUpdating file {config.datadir}/{config.split_dir}/{filename}.txt with POS tags...\n')
        with open(Path(config.datadir).joinpath(config.split_dir,filename).with_suffix('.txt'),'r') as f, \
          open(Path(config.datadir).joinpath(config.split_dir,filename+new_file_suffix).with_suffix('.txt'),'w') as fn:
            lines = f.readlines()
            for i, line in tqdm(enumerate(lines)):
               	split_line = line.split('\t')
               	if i != 0 and split_line[0] != "<file>":
                    word = split_line[0].strip()
                    tag_prominence = split_line[1].strip()
                    tag_boundary = split_line[2].strip()
                    value_prominance = split_line[3].strip()
                    value_boundary = split_line[4].strip()
                    pos_coarse = pos_coarse_grained_tagger([word])[0]
                    pos_fine = pos_fine_grained_tagger([word])[0]
                    lemma = lemma_tagger([word])[0]
                    stop_tag = stop_tagger([word])[0]
                    fn.write(f'{word}\t{tag_prominence}\t{tag_boundary}\t{value_prominance}\t{value_boundary}\t{pos_coarse}\t{pos_fine}\t{lemma}\t{stop_tag}')
                    fn.write('\n')
                elif (split_line[0] == "<file>") or i+1 == len(lines):
                    fn.write(line)


def load_dataset(config):
    # init vars
    splits = dict()
    words = []
    all_sents = []
    # init tag lists
    pos_tag_list_coarse = ['ADJ','ADP','ADV','AUX','CONJ','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X','SPACE']
    pos_tag_list_fine = ['AFX', 'JJ', 'JJR', 'JJS', 'PDT', 'PRP$', 'WDT', 'WP$', 'IN', 'EX', 'RB', 'RBR', 'RBS', 'WRB', 'CC', 'DT', 'NN', 'UH', 'NNS', 'WP', 'CD', 'POS', 'RP', 'TO', 'PRP', 'NNP', 'NNPS', '-LRB-', '-RRB-', ',', ':', '.', '”', '“”', '“', 'HYPH', 'LS', 'NFP', '#', '$', 'SYM', 'BES', 'HVS', 'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'ADD', 'FW', 'GW', 'XX', '_SP', 'NIL']

    # iterate across sets
    for split in ['train', 'dev', 'test']:
        tagged_sents = []
        filename = config.train_set if split == 'train' else split
        with open(Path(config.datadir).joinpath(config.split_dir,filename).with_suffix('.txt'),'r') as f:
            lines = f.readlines()
            if config.fraction_of_train_data < 1 and split == 'train':
                slice = len(lines) * config.fraction_of_train_data
                lines = lines[0:int(round(slice))]
            sent = []
            for i, line in enumerate(lines):
                split_line = line.split('\t')
                if i != 0 and split_line[0] != "<file>":
                    word = split_line[0]
                    tag_prominence = split_line[1]
                    tag_boundary = split_line[2]
                    value_prominance = split_line[3]
                    value_boundary = split_line[4]
                    tag_pos = split_line[5] if config.use_pos else []

                    # Modify tag value if we specified a different config.nclasses
                    # than default value of 3
                    if config.nclasses == 2:
                        if tag_prominence == '2': tag_prominence = '1' # Collapse the non-0 classes
                    elif config.nclasses > 3:
                        tag_prominence = rediscretize_tag(value_prominance, config.nclasses)

                    sent.append((word, tag_prominence, tag_boundary, value_prominance, value_boundary, tag_pos))
                    words.append(word)
                elif (i != 0 and split_line[0] == "<file>") or i+1 == len(lines):
                    tagged_sents.append(sent)
                    sent = []

        # shuffle
        if config.shuffle_sentences:
            random.shuffle(tagged_sents)

        splits[split] = tagged_sents
        all_sents = all_sents + tagged_sents

    # create vocabulary
    vocab = []
    for token in words:
        if token not in vocab:
            vocab.append(token)
    vocab = set(vocab)

    # update tag lists
    tags = list(set(word_tag[1] for sent in all_sents for word_tag in sent))
    tags = ["<pad>"] + tags
    pos_tag_list_coarse = list(set(word_tag[5] for sent in all_sents for word_tag in sent))
    pos_tag_list_coarse = ["<pad>"] + sorted(pos_tag_list_coarse)
    pos_tag_list_fine = ["<pad>"] + sorted(pos_tag_list_fine)

    # assign tag dictionaries
    tag2index = {tag: index for index, tag in enumerate(tags)}
    index2tag = {index: tag for index, tag in enumerate(tags)}
    pos2index = {tag:i for i,tag in enumerate(pos_tag_list_coarse)}
    index2pos = {i:tag for i,tag in enumerate(pos_tag_list_coarse)}

    # print status
    print('Training sentences: {}'.format(len(splits["train"])))
    print('Dev sentences: {}'.format(len(splits["dev"])))
    print('Test sentences: {}'.format(len(splits["test"])))

    # sort
    if config.sorted_batches:
        random.shuffle(splits["train"])
        splits["train"].sort(key=len)

    return splits, tag2index, index2tag, vocab, pos2index, index2pos


def pad(batch):
    # Pad sentences to the longest sample
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_main_piece = f(2)
    tags = f(3)
    pos = f(8)
    seqlens = f(5)
    maxlen = np.array(seqlens).max()
    invalid_set_to = f(9)[0]
    import pdb;pdb.set_trace()
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(4, maxlen)
    k = f(7, maxlen)

    f = lambda x, seqlen: [sample[x] + [invalid_set_to] * (seqlen - len(sample[x])) for sample in batch] #invalid values are NA and <pad>
    values = f(6, maxlen)

    f = torch.LongTensor
    return words, f(x), is_main_piece, tags, f(y), seqlens, torch.FloatTensor(values), f(k), pos, invalid_set_to


def load_embeddings(config, vocab):
    vocab.add('UNK')
    word2id = {word: id for id, word in enumerate(vocab)}
    embed_size = 300
    vocab_size = len(vocab)
    sd = 1 / np.sqrt(embed_size)
    weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
    weights = weights.astype(np.float32)
    with open(config.embedding_file, encoding='utf8', mode="r") as textFile:
        for line in textFile:
            line = line.split()
            word = line[0]

            # If word is in our vocab, then update the corresponding weights
            id = word2id.get(word, None)
            if id is not None and len(line) == 301:
                weights[id] = np.array([float(val) for val in line[1:]])

    return weights, word2id

def rediscretize_tag(value_prominance, nclasses):
    if value_prominance == 'NA':
        return 'NA'

    # Simple dividing into bins:
    SOFT_MAX_BOUND = 6.0
    return str(int(min(float(value_prominance) * nclasses / SOFT_MAX_BOUND, nclasses)))

