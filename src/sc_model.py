from torch import nn
from transformers import AutoModel, AutoTokenizer

class Predictor(nn.Module):
    def __init__(self, params):
        super(Predictor, self).__init__()
        self.hidden_size = params.hidden_size
        self.class_num = 2
        self.linear = nn.Linear(self.hidden_size, self.class_num)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        feature = self.dropout(x)
        pred = self.linear(feature)
        return pred

class SentimentClassification(nn.Module):
    def __init__(self, params):
        super(SentimentClassification, self).__init__()
        self.predictor = Predictor(params)
        self.ptm_model = AutoModel.from_pretrained(params.ptm_folder)
        self.tokenizer = AutoTokenizer.from_pretrained(params.ptm_folder)
        self.embed_size = params.embed_size
        self.max_length = params.max_length
        self.exp_id = params.exp_id

    def forward(self, sentence, cs_sentence, label):
        batch_size = len(label)
        all_batch_seqs = []
        all_cs_batch_seqs = []
        for batch_id in range(batch_size):
            input_sentence = sentence[batch_id]
            input_cs_sentence = cs_sentence[batch_id]
            all_batch_seqs.append(input_sentence)
            all_cs_batch_seqs.append(input_cs_sentence)

        all_batch_seqs = self.tokenizer(all_batch_seqs, return_tensors='pt', truncation= True, padding=True, max_length= self.max_length)
        input_ids = all_batch_seqs['input_ids'].cuda()
        embeddings = self.ptm_model(input_ids)
        embeddings = embeddings[1]
        pred = self.predictor(embeddings)

        all_cs_batch_seqs = self.tokenizer(all_cs_batch_seqs, return_tensors='pt', truncation= True, padding=True, max_length= self.max_length)
        input_ids = all_cs_batch_seqs['input_ids'].cuda()
        embeddings_cs = self.ptm_model(input_ids)
        embeddings_cs = embeddings_cs[1]
        pred_cs = self.predictor(embeddings_cs)

        return pred, pred_cs, embeddings, embeddings_cs