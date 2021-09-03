from src.preprocess import load_sentence_data
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import random
import pickle

random.seed(1)

class Dataset(data.Dataset):
    def __init__(self, turns):
        self.turns = turns

    def __getitem__(self, item):
        return self.turns[item][0], self.turns[item][1], self.turns[item][2]

    def __len__(self):
        return len(self.turns)

def collate_fn(data):
    sentence, cs_sentence, label = zip(*data)
    label = torch.LongTensor(label)

    return sentence, cs_sentence, label

def load_data(params, mapping=None):

    pri_turns_train = load_sentence_data("data/amazon_review/en/"+params.domain+"/train","en",mapping=mapping, dynamic_mix=params.dynamic_mix,
                                              dynamic_ratio= params.dynamic_ratio)
    pri_turns_val = load_sentence_data("data/amazon_review/en/"+params.domain+"/test","en",mapping=mapping, dynamic_mix=params.dynamic_mix,
                                              dynamic_ratio= params.dynamic_ratio)
    pri_turns_total = pri_turns_train + pri_turns_val
    val_count = len(pri_turns_total)
    random.shuffle(pri_turns_total)
    pri_turns_train = pri_turns_total[0:int(params.dev_ratio * val_count)]
    pri_turns_val = pri_turns_total[int(params.dev_ratio * val_count):]
    pri_turns_test = load_sentence_data("data/amazon_review/"+params.trans_lang+'/'+params.domain+"/test", params.trans_lang ,mapping=mapping, dynamic_mix=params.dynamic_mix,
                                              dynamic_ratio= params.dynamic_ratio)

    return pri_turns_train, pri_turns_val, pri_turns_test

def get_sc_dataloader(params):
    if params.mix_train == True:
        with open(params.mapping_for_mix, "rb") as f:
            mapping_for_mix = pickle.load(f)
    else:
        mapping_for_mix = None

    train_turns, tgt_val_turns, tgt_test_turns = load_data(params, mapping=mapping_for_mix)
    dataset_tr = Dataset(train_turns)
    dataset_val = Dataset(tgt_val_turns)
    dataset_test = Dataset(tgt_test_turns)

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False,collate_fn=collate_fn)

    return dataloader_tr, dataloader_val, dataloader_test