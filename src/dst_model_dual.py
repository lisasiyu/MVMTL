import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from transformers import AutoModel, AutoTokenizer
import pdb


class Predictor(nn.Module):
    def __init__(self, params):
        super(Predictor, self).__init__()
        self.hidden_size = params.hidden_size
        self.food_class = params.food_class
        self.price_range_class = params.price_range_class
        self.area_class = params.area_class
        self.request_class = params.request_class

        self.linear_food = nn.Linear(self.hidden_size, self.food_class)
        self.linear_price_range = nn.Linear(self.hidden_size, self.price_range_class)
        self.linear_area = nn.Linear(self.hidden_size, self.area_class)
        self.linear_request = nn.Linear(self.hidden_size, self.request_class)
        self.dropout = nn.Dropout(params.dropout)

        self.sigmoid = nn.Sigmoid()

    def forward(self, gates):
        # food slot
        feature_food = gates[:, 0, :]  # bsz, hidden_size
        feature_food = self.dropout(feature_food)
        food_value_pred = self.linear_food(feature_food)

        # price range slot
        feature_price_range = gates[:, 1, :]
        feature_price_range = self.dropout(feature_price_range)
        price_range_value_pred = self.linear_price_range(feature_price_range)

        # area slot
        feature_area = gates[:, 2, :]
        feature_area = self.dropout(feature_area)
        area_value_pred = self.linear_area(feature_area)

        # request
        feature_request = gates[:, 3, :]
        feature_request = self.dropout(feature_request)
        request_value_pred = self.sigmoid(self.linear_request(feature_request))

        return food_value_pred, price_range_value_pred, area_value_pred, request_value_pred


class DialogueStateTracker(nn.Module):
    def __init__(self, params):
        super(DialogueStateTracker, self).__init__()
        self.predictor = Predictor(params)

        self.ptm_model = AutoModel.from_pretrained(params.ptm_folder)
        self.tokenizer = AutoTokenizer.from_pretrained(params.ptm_folder)

        self.embed_size = params.embed_size
        self.exp_id = params.exp_id

    def forward(self, original, utters, acts_request, acts_slot, acts_value, slot_names, lang, systems):
        batch_size = len(acts_request)

        all_batch_seqs = []
        original_batch_seqs = []
        for batch_id in range(batch_size):

            req_list = acts_request[batch_id]
            slot_type_list = acts_slot[batch_id]
            slot_value_list = acts_value[batch_id]
            slot_name = slot_names[batch_id]
            # print(utters[batch_id], acts_request[batch_id], acts_slot[batch_id], acts_value[batch_id], slot_names[batch_id])

            if req_list == []:
                req_list = 'request nothing'
            else:
                req_list = 'request ' + ' & '.join(req_list)
            if slot_type_list == []:
                type_value_pairs = 'inform nothing'
            else:
                type_value_pairs = []
                for slot_type, slot_value in zip(slot_type_list, slot_value_list):
                    type_value_pairs.append(slot_type + ' - ' + slot_value)
                type_value_pairs = 'inform ' + ' & '.join(type_value_pairs)

            for id_, slot in enumerate(slot_name):
                if type(utters[batch_id]) == tuple:
                    context_sequence = [req_list + ' . ' + type_value_pairs + ' .'] + [utters[batch_id][id_]] + [slot]
                    context_sequence = ' </s> '.join(context_sequence)
                    all_batch_seqs.append(context_sequence)
                else:
                    context_sequence = [req_list + ' . ' + type_value_pairs + ' .'] + [utters[batch_id]] + [slot]
                    context_sequence = ' </s> '.join(context_sequence)

                    all_batch_seqs.append(context_sequence)

            for id_, slot in enumerate(slot_name):
                original_context_sequence = [req_list + ' . ' + type_value_pairs + ' .'] + [original[batch_id]] + [slot]
                original_context_sequence = ' </s> '.join(original_context_sequence)
                original_batch_seqs.append(original_context_sequence)

        # code switch
        all_batch_seqs = self.tokenizer(all_batch_seqs, return_tensors='pt', padding=True, return_length=True)
        input_ids = all_batch_seqs['input_ids'].cuda()
        embeddings = self.ptm_model(input_ids)
        embeddings = embeddings[1].reshape(batch_size, 4, self.embed_size)
        # original
        original_batch_seqs = self.tokenizer(original_batch_seqs, return_tensors='pt', padding=True, return_length=True)
        input_ids = original_batch_seqs['input_ids'].cuda()
        original_embeddings = self.ptm_model(input_ids)
        original_embeddings = original_embeddings[1].reshape(batch_size, 4, self.embed_size)

        # code switch predictions
        food_value_pred, price_range_value_pred, area_value_pred, request_value_pred = self.predictor(embeddings)

        # original predictions
        original_food_value_pred, original_price_range_value_pred, original_area_value_pred, original_request_value_pred = self.predictor(
            original_embeddings)

        return original_food_value_pred, original_price_range_value_pred, original_area_value_pred, original_request_value_pred, \
               food_value_pred, price_range_value_pred, area_value_pred, request_value_pred