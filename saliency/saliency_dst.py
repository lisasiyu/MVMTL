import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
import math
import torch.autograd as autograd
import pickle
import codecs
import json
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import logging
import numpy as np
from config import get_params
logger = logging.getLogger()

from src.preprocess import load_woz_data
from collections import defaultdict

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import numpy as np
import pickle
import random
import pdb


class Dataset(data.Dataset):
    def __init__(self, turns):
        self.turns = turns

    def __getitem__(self, index):
        # dialog_idx, utterance, acts_request, acts_slot_type, acts_slot_value, turn_slot, turn_slot_label, turn_request_label, system_transcriptions
        return self.turns[index][0], self.turns[index][1], self.turns[index][2], self.turns[index][3], \
               self.turns[index][4], self.turns[index][5], self.turns[index][6], self.turns[index][7], \
               self.turns[index][8]

    def __len__(self):
        return len(self.turns)


def collate_fn(data):
    dialogu_idx, utterances, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels, turn_request_labels, systems = zip(
        *data)

    turn_slot_labels = torch.LongTensor(turn_slot_labels)
    turn_request_labels = torch.FloatTensor(turn_request_labels)

    return dialogu_idx, utterances, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels, turn_request_labels, systems


def load_data(params, dialogue_ontology, mapping=None):
    pri_turns_train = load_woz_data("../data/dst/dst_data/tok_woz_train_en.json", "en", dialogue_ontology, mapping=mapping,
                                    dynamic_mix=params.dynamic_mix, dynamic_ratio=params.dynamic_ratio)
    return pri_turns_train


def binarize_dst_data(params, turns, dialogue_ontology, lang, isTestset=False):
    if lang == "en":
        if params.mix_train:
            if params.trans_lang == 'de':
                class_type_dict = {"essen": [], "preisklasse": [], "gegend": [], "request": []}
                de2en_mapping = {"essen": "food", "preisklasse": "price range", "gegend": "area", "request": "request"}
            else:
                class_type_dict = {"cibo": [], "prezzo": [], "area": [], "request": []}
                it2en_mapping = {"cibo": "food", "prezzo": "price range", "area": "area", "request": "request"}
        else:
            class_type_dict = {"food": [], "price range": [], "area": [], "request": []}
    elif lang == "de":
        class_type_dict = {"essen": [], "preisklasse": [], "gegend": [], "request": []}
        de2en_mapping = {"essen": "food", "preisklasse": "price range", "gegend": "area", "request": "request"}
    elif lang == "it":
        class_type_dict = {"cibo": [], "prezzo": [], "area": [], "request": []}
        it2en_mapping = {"cibo": "food", "prezzo": "price range", "area": "area", "request": "request"}

    for slot_type in class_type_dict.keys():
        if lang == "de":
            slot_type_ = de2en_mapping[slot_type]
        elif lang == "it":
            slot_type_ = it2en_mapping[slot_type]
        else:
            if params.mix_train:
                if params.trans_lang == 'de':
                    slot_type_ = de2en_mapping[slot_type]
                else:
                    slot_type_ = it2en_mapping[slot_type]
            else:
                slot_type_ = slot_type
        class_type_dict[slot_type] = dialogue_ontology[slot_type_][lang]

    with codecs.open('../data/dst/dst_data/ontology-mapping.json', 'r', 'utf8') as f:
        ontology_mapping = json.load(f)

    lang_dict = {"en": 0, "de": 1, "it": 2}
    ontology_vocab = []
    lang_id = lang_dict[lang]
    for item in ontology_mapping:
        ontology_vocab.append(item[lang_id])

    binarized_turns = []

    logger.info("Binarizing labels ...")
    for i, each_turn in enumerate(turns):
        binarized_slots, binarized_slot_values, binarized_request_values = [], [], []

        dialogue_idx, utterance, acts_inform_slots, acts_slot_type, acts_slot_value, test_label, turn_label, system = each_turn

        labels = test_label if isTestset == True else turn_label

        for label_slot, label_value in labels.items():

            binarized_slots.append(label_slot)
            if label_value == "none":
                binarized_slot_values.append(len(class_type_dict[label_slot]))  # label for none
            else:
                if type(label_value) is list:
                    r_label = [0] * params.request_class
                    r_indices = [class_type_dict[label_slot].index(item) for item in label_value]
                    for idx in r_indices:
                        r_label[idx] = 1
                    binarized_request_values = r_label
                else:
                    binarized_slot_values.append(class_type_dict[label_slot].index(label_value))

        current_turn = (dialogue_idx, utterance, acts_inform_slots, acts_slot_type, acts_slot_value, binarized_slots,
                        binarized_slot_values, binarized_request_values, system)

        binarized_turns.append(current_turn)

    return binarized_turns

def get_dst_dataloader(params, dialogue_ontology, train_turns=None):

    mapping_for_mix = None

    if train_turns is None:
        train_turns = load_data(params, dialogue_ontology, mapping=mapping_for_mix)

    train_turns_bin = binarize_dst_data(params, train_turns, dialogue_ontology, lang="en", isTestset=False)

    dataset_tr = Dataset(train_turns_bin)
    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_tr

# adapted from AllenNLP Interpret
def _register_embedding_list_hook(model, embeddings_list, model_type):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    if model_type == 'XLMR':
        embedding_layer = model.ptm_model.embeddings.word_embeddings
    else:
        raise ValueError("Current model type not supported.")
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def _register_embedding_gradient_hooks(model, embeddings_gradients, model_type):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0])
    if model_type == 'XLMR':
        embedding_layer = model.ptm_model.embeddings.word_embeddings
    else:
        raise ValueError("Current model type not supported.")
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook

def softmax(x):
    tem = 0.1
    x = torch.tensor(x)
    return np.exp(x/tem) / torch.sum(np.exp(x/tem))

def saliency_map(model, utters, acts_request, acts_slot, acts_value, slot_name, slot_labels, request_labels, systems,model_type='XLMR'):
    # tokenizer = AutoTokenizer.from_pretrained(params.ptm_folder)
    embeddings_list = []
    handle = _register_embedding_list_hook(model, embeddings_list, model_type)
    embeddings_gradients = []
    hook = _register_embedding_gradient_hooks(model, embeddings_gradients, model_type)
    loss_fn1 = nn.CrossEntropyLoss()
    loss_fn2 = nn.MSELoss()
    food_label = slot_labels[:, 0]  # (bsz, 1)
    price_range_label = slot_labels[:, 1]  # (bsz, 1)
    area_label = slot_labels[:, 2]  # (bsz, 1)
    model.zero_grad()
    pred_food, pred_price, pred_area, pred_request, input_ids = model(utters, acts_request, acts_slot, acts_value, slot_name, "en", systems)
    loss_food = loss_fn1(pred_food, food_label.cuda())
    loss_price = loss_fn1(pred_price, price_range_label.cuda())
    loss_area = loss_fn1(pred_area, area_label.cuda())
    loss_request = loss_fn2(pred_request, request_labels.cuda())
    total_loss = loss_food + loss_price + loss_area + loss_request
    total_loss.backward()
    handle.remove()
    hook.remove()

    saliency_grad = embeddings_gradients[0].detach().cpu().numpy()
    # saliency_grad = saliency_grad * embeddings_list[0]
    saliency_grad = np.sum(saliency_grad, axis=(0,2))
    norm = np.linalg.norm(saliency_grad, ord=1)
    saliency_grad = [np.maximum(e,-e) / norm for e in saliency_grad]
    # saliency_prob = softmax(saliency_grad)
    # saliency_grad = [(- e) / norm for e in saliency_grad] # negative gradient for loss means positive influence on decision
    return saliency_grad,input_ids


def dst(params):
    best_model_path = '../experiments/dst/ende/999/best_model.pth'
    with codecs.open('../data/dst/dst_data/ontology_classes.json', 'r', 'utf8') as f:
        dialogue_ontology = json.load(f)
    best_model = torch.load(best_model_path)
    model = best_model["dialog_state_tracker"]
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(params.ptm_folder)
    dataloader_tr= get_dst_dataloader(params,dialogue_ontology)
    pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
    saliency_dict = defaultdict(int)
    last_dia_id = 0
    dialogue = []
    saliency_dict = defaultdict(float)
    freq = defaultdict(float)
    dia_id = -1

    for i, (t_id, utters, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels, turn_request_labels, turn_systems) in pbar:
        if t_id[0] == 0:
            # if dia_id != -1:
                # SA.append({"dialogue_idx": dia_id, "dialogue": dialogue})
                # dialogue = []
            dia_id += 1
        model.eval()
        saliency_scores, input_ids = saliency_map(model, utters, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels, turn_request_labels, turn_systems, model_type='XLMR')
        sentence = tokenizer.convert_ids_to_tokens(input_ids[0])
        final_saliency = []
        sentence1 = utters[0].split(' ')
        flag = 0
        s = 1.
        saliency = []
        SA = []
        for j, sub_word in enumerate (sentence):
            if sub_word == '</s>':
                flag += 1
            if flag == 1:
                if '▁' in sub_word:
                    if s != 1:
                        saliency[-1] = saliency[-1]/s
                    saliency.append(saliency_scores[j])

                    s = 1
                elif '▁' not in sub_word and sub_word!= '<s>' and sub_word != '</s>' and sub_word != '<pad>' and sub_word != '<mask>':
                    saliency[-1] += saliency_scores[j]
                    s += 1.
            if flag == 2:
                break
        for z in range(len(saliency)):
            saliency_dict[sentence1[z]] += saliency_scores[z]
            freq[sentence1[z]] += 1
    vocab_size = len(freq)
    for key, value in freq.items():
        freq[key] = -math.log(value / vocab_size)
        saliency_dict[key] = freq[key] * saliency_dict[key]
    with open("saliency_dst.txt", "w") as f:
        d = sorted(saliency_dict.items(), key=lambda kv: kv[1], reverse=True)
        for (key, value) in d:
            f.writelines(key + ' ' + str(value) + '\n')
    f.close()

        for z in range(len(saliency)):
            SA.append((sentence1[z], saliency[z]))
        SA.sort(key=lambda x: x[1],reverse = True)
        # print(str(dia_id),utters[0])
        l = len(sentence1)
        # for z in range(l):
        #     saliency_dict[SA[z][0]] += 1
        if len(sentence1) >=5:
            for z in range(5):
                saliency_dict[SA[z][0]] += 1
        else:
            saliency_dict[SA[0][0]] += 1
        saliency = softmax(saliency).numpy().tolist()
        final_saliency.append(saliency)

        dialogue.append({'utters':utters[0],'saliency':final_saliency})

    if dia_id == 599:
        SA.append({"dialogue_idx": dia_id, "dialogue": dialogue})


    with open("saliency_train_result.json", "w") as f:
        f.write(json.dumps(SA, indent=4))
    with open("saliency_dst_top5.txt","w") as f:
        d = sorted(saliency_dict.items(), key = lambda kv:kv[1],reverse=True)
        for (key,value) in d:
            f.writelines(key +' ' + str(value) +'\n')


if __name__ == "__main__":
    params = get_params()
    dst(params)