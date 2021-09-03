from src.preprocess import load_woz_data, load_woz_train_data
from src.utils import binarize_dst_data

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import numpy as np
import pickle
import random
import pdb

random.seed(1)

class Dataset(data.Dataset):
    def __init__(self, turns):
        self.turns = turns

    def __getitem__(self, index):
        # dialog_idx, utterance, acts_request, acts_slot_type, acts_slot_value, turn_slot, turn_slot_label, turn_request_label, system_transcriptions
        return self.turns[index][0], self.turns[index][1], self.turns[index][2], self.turns[index][3], \
               self.turns[index][4], self.turns[index][5], self.turns[index][6], self.turns[index][7], \
               self.turns[index][8], self.turns[index][9]

    def __len__(self):
        return len(self.turns)


def collate_fn(data):
    dialogu_idx, original, utterances, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels, turn_request_labels, systems = zip(
        *data)

    turn_slot_labels = torch.LongTensor(turn_slot_labels)
    turn_request_labels = torch.FloatTensor(turn_request_labels)

    return dialogu_idx, original, utterances,  acts_request, acts_slot, acts_values, slot_names, turn_slot_labels, turn_request_labels, systems


def load_data(params, dialogue_ontology, mapping=None):

    if params.saliency == True:
        pri_turns_train = load_woz_train_data("data/dst/dst_data/tok_woz_train_en_"+ params.trans_lang +".json",  "en",
                                              dialogue_ontology,file_saliency = params.saliency_train_path, mapping=mapping, dynamic_mix=params.dynamic_mix,
                                              dynamic_ratio=params.dynamic_ratio)
        # pri_turns_val = load_woz_train_data("data/dst/dst_data/tok_woz_validate_en_"+ params.trans_lang +".json",  "en",
        #                                     dialogue_ontology, mapping=mapping,file_saliency = params.saliency_dev_path, dynamic_mix=params.dynamic_mix,
        #                                     dynamic_ratio=params.dynamic_ratio)
    else:
        pri_turns_train = load_woz_data("data/dst/dst_data/tok_woz_train_en_"+ params.trans_lang +".json", "en",
                                              dialogue_ontology, mapping=mapping, dynamic_mix=params.dynamic_mix,
                                              dynamic_ratio=params.dynamic_ratio)
        # pri_turns_val = load_woz_data("data/dst/dst_data/tok_woz_validate_en_"+ params.trans_lang +".json", "en",
        #                                     dialogue_ontology, mapping=mapping, dynamic_mix=params.dynamic_mix,
        #                                     dynamic_ratio=params.dynamic_ratio)

    # val_count = len(pri_turns_val)
    # random.shuffle(pri_turns_val)
    # pri_turns_train = pri_turns_train * params.train_resample_ratio + pri_turns_val[0:int(params.dev_ratio * val_count)]
    #
    # tgt_pri_turns_val = load_woz_data("data/dst/dst_data/tok_woz_validate_" + params.trans_lang + ".json",
    #                                   params.trans_lang, dialogue_ontology, mapping=mapping,
    #                                   dynamic_mix=params.dynamic_mix, dynamic_ratio=params.dynamic_ratio)
    pri_turns_val = load_woz_data("data/dst/dst_data/tok_woz_validate_en_" + params.trans_lang + ".json", "en",
                                        dialogue_ontology, mapping=mapping,
                                        dynamic_mix=params.dynamic_mix,
                                        dynamic_ratio=params.dynamic_ratio)
    tgt_pri_turns_test = load_woz_data("data/dst/dst_data/tok_woz_test_" + params.trans_lang + ".json",
                                       params.trans_lang, dialogue_ontology, mapping=mapping,
                                       dynamic_mix=params.dynamic_mix, dynamic_ratio=params.dynamic_ratio)

    return pri_turns_train, pri_turns_val, tgt_pri_turns_test


def get_dst_dataloader(params, dialogue_ontology, train_turns=None, tgt_val_turns=None, tgt_test_turns=None):
    if params.mix_train == True:
        with open(params.mapping_for_mix, "rb") as f:
            mapping_for_mix = pickle.load(f)
    else:
        mapping_for_mix = None

    if train_turns is None:
        train_turns, tgt_val_turns, tgt_test_turns = load_data(params, dialogue_ontology, mapping=mapping_for_mix)

    train_turns_bin = binarize_dst_data(params, train_turns, dialogue_ontology, lang="en", isTestset=False)
    tgt_turns_val_bin = binarize_dst_data(params, tgt_val_turns, dialogue_ontology, lang="en",
                                          isTestset=True)
    tgt_turns_test_bin = binarize_dst_data(params, tgt_test_turns, dialogue_ontology, lang=params.trans_lang,
                                           isTestset=True)

    dataset_tr = Dataset(train_turns_bin)
    dataset_val = Dataset(tgt_turns_val_bin)
    dataset_test = Dataset(tgt_turns_test_bin)

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False,
                                 collate_fn=collate_fn)

    return dataloader_tr, dataloader_val, dataloader_test, train_turns, tgt_val_turns, tgt_test_turns