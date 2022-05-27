import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import random
import pickle
import torch
import torch.nn as nn
import math
import torch.autograd as autograd
import pickle
import os
import codecs
import json
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
import logging
import numpy as np
from config import get_params

class Dataset(data.Dataset):
    def __init__(self, turns):
        self.turns = turns

    def __getitem__(self, item):
        return self.turns[item][0], self.turns[item][1],self.turns[item][2]

    def __len__(self):
        return len(self.turns)

def collate_fn(data):
    sentence,file, label = zip(*data)
    label = torch.LongTensor(label)

    return file, sentence,  label

def load_data(params, mapping=None):

    pri_turns_train = load_sentence_data("../data/amazon_review/en/music/train","en",mapping=mapping, dynamic_mix=params.dynamic_mix,
                                              dynamic_ratio= params.dynamic_ratio)
    pri_turns_val = load_sentence_data("../data/amazon_review/en/dvd/test", "en", mapping=mapping,
                                       dynamic_mix=params.dynamic_mix,
                                       dynamic_ratio=params.dynamic_ratio)
    return pri_turns_train,pri_turns_val

def get_sc_dataloader(params):
    if params.mix_train == True:
        with open(params.mapping_for_mix, "rb") as f:
            mapping_for_mix = pickle.load(f)
    else:
        mapping_for_mix = None

    train_turns, val_turns = load_data(params, mapping=mapping_for_mix)
    # pri_turns_total = train_turns + val_turns
    pri_turns_total = train_turns
    dataset_tr = Dataset(pri_turns_total)

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_tr

def load_sentence_data(file_path, language, mapping=None, dynamic_mix=False,dynamic_ratio=0.75):
    turns =[]
    pos_path = file_path + "/positive/"
    for file in os.listdir(pos_path):
        filepath = pos_path + file
        temp = []
        for line in open(filepath):
            temp.append(line)
        temp = ''.join(temp)
        turn = process_sc_dialogue(temp, file, 1, language, mapping=mapping, dynamic_mix=dynamic_mix, dynamic_ratio=dynamic_ratio)
        turns.extend(turn)

    neg_path = file_path + "/negative/"
    for file in os.listdir(neg_path):
        filepath = neg_path + file
        temp = []
        for line in open(filepath):
            temp.append(line)
        temp = ''.join(temp)
        turn = process_sc_dialogue(temp,file, 0, language, mapping=mapping, dynamic_mix=dynamic_mix, dynamic_ratio=dynamic_ratio)
        turns.extend(turn)
    return turns

def process_sc_dialogue(orig_sentence,file,label, language, mapping=None, dynamic_mix=False,dynamic_ratio=0.75):
    representation = []
    representation.append((orig_sentence,file,label))
    return representation


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

def saliency_map(model, file,sentence, label,model_type='XLMR'):
    # tokenizer = AutoTokenizer.from_pretrained(params.ptm_folder)
    embeddings_list = []
    handle = _register_embedding_list_hook(model, embeddings_list, model_type)
    embeddings_gradients = []
    hook = _register_embedding_gradient_hooks(model, embeddings_gradients, model_type)
    loss_fn1 = nn.CrossEntropyLoss()
    model.zero_grad()
    pred,input_ids = model(sentence, label)
    loss = loss_fn1(pred, label.cuda())
    loss.backward()
    handle.remove()
    hook.remove()

    saliency_grad = embeddings_gradients[0].detach().cpu().numpy()
    saliency_grad = np.sum(saliency_grad[0] , axis=1)
    # saliency_grad = np.sum(saliency_grad[0] * embeddings_list[0], axis=1)
    norm = np.linalg.norm(saliency_grad, ord=1)
    saliency_grad = [np.maximum(e,-e) / norm for e in saliency_grad]
    # saliency_prob = softmax(saliency_grad)
    # saliency_grad = [(- e) / norm for e in saliency_grad] # negative gradient for loss means positive influence on decision
    return saliency_grad, input_ids

def sc(params):
    best_model_path = '../experiments/sc/music/ende/999/best_model.pth'
    best_model = torch.load(best_model_path)
    model = best_model["sentiment_classification"]
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(params.ptm_folder)
    dataloader_tr= get_sc_dataloader(params)
    pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
    last_dia_id = 0
    dialogue = []
    dia_id = -1
    saliency_dict = defaultdict(float)
    saliency_dict1 = defaultdict(int)
    saliency_dict2 = defaultdict(int)
    saliency_dict3 = defaultdict(int)
    saliency_dict4 = defaultdict(int)
    freq = defaultdict(float)


    for i, (file,sentence, label) in pbar:
        model.eval()
        saliency_scores, input_ids = saliency_map(model,file,sentence, label, model_type='XLMR')
        tok_sentence = tokenizer.convert_ids_to_tokens(input_ids[0])
        final_saliency = []
        sentence = sentence[0].split(' ')

        flag = 0
        s = 1.
        saliency = []
        SA = []
        for j, sub_word in enumerate (tok_sentence):
            if '▁' in sub_word:
                if s != 1:
                    saliency[-1] = saliency[-1]/s
                saliency.append(saliency_scores[j])
                s = 1
            elif '▁' not in sub_word and sub_word!= '<s>' and sub_word != '</s>' and sub_word != '<pad>' and sub_word != '<mask>':
                saliency[-1] += saliency_scores[j]
                s += 1.
        # saliency = softmax(saliency).numpy().tolist()
        for z in range(len(saliency)):
            saliency_dict[sentence[z]] +=  saliency_scores[z]
            freq[sentence[z]] += 1
    vocab_size = len(freq)
    for key, value in freq.items():
        freq[key] = -math.log(value / vocab_size)
        saliency_dict[key] = freq[key] * saliency_dict[key]
    with open("saliency_sc_music.txt","w") as f:
        d = sorted(saliency_dict.items(), key = lambda kv:kv[1],reverse=True)
        for (key,value) in d:
            f.writelines(key +' ' + str(value) +'\n')
    f.close()


        for z in range(len(saliency)):
            SA.append((sentence[z], saliency[z]))
        SA.sort(key=lambda x: x[1],reverse = True)

        for z in range(1):
            saliency_dict1[SA[z][0]] += 1
        for z in range(2):
            saliency_dict2[SA[z][0]] += 1
        for z in range(3):
            saliency_dict3[SA[z][0]] += 1
        for z in range(4):
            saliency_dict4[SA[z][0]] += 1
        for z in range(len(saliency)):
            saliency_dict[SA[z][0]] += 1


        dialogue.append({'id':file[0],'utters': sentence[0],'saliency':np.array(saliency).tolist()})



    with open("saliency_sc_result.json", "w") as f:
        f.write(json.dumps(dialogue, indent=4))
    # with open("saliency_sc_music00.txt","w") as f:
    #     d = sorted(saliency_dict.items(), key = lambda kv:kv[1],reverse=True)
    #     for (key,value) in d:
    #         f.writelines(key +' ' + str(value) +'\n')
    # f.close()
    #
    # with open("saliency_sc_music_top11.txt","w") as f:
    #     d = sorted(saliency_dict1.items(), key = lambda kv:kv[1],reverse=True)
    #     for (key,value) in d:
    #         f.writelines(key +' ' + str(value) +'\n')
    # f.close()
    #
    # with open("saliency_sc_music_top22.txt","w") as f:
    #     d = sorted(saliency_dict2.items(), key = lambda kv:kv[1],reverse=True)
    #     for (key,value) in d:
    #         f.writelines(key +' ' + str(value) +'\n')
    # f.close()
    #
    # with open("saliency_sc_music_top33.txt","w") as f:
    #     d = sorted(saliency_dict3.items(), key = lambda kv:kv[1],reverse=True)
    #     for (key,value) in d:
    #         f.writelines(key +' ' + str(value) +'\n')
    # f.close()
    #
    # with open("saliency_sc_music_top44.txt","w") as f:
    #     d = sorted(saliency_dict4.items(), key = lambda kv:kv[1],reverse=True)
    #     for (key,value) in d:
    #         f.writelines(key +' ' + str(value) +'\n')
    # f.close()




if __name__ == "__main__":
    params = get_params()
    sc(params)