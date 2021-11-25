# MVMLT
Code for EMNLP2021 findings "Saliency-based Multi-View Mixed Language Training for Zero-shot Cross-lingual Classification"
https://aclanthology.org/2021.findings-emnlp.55/

## Requirments
pip install requirements.txt
```
transformers==4.4.2
tqdm==4.62.0
numpy==1.21.1
torch==1.9.0
scikit_learn==0.24.2
```
Download the pretrained [xlm-roberta-base model](https://huggingface.co/xlm-roberta-base/tree/main) to ./xlm-roberta-base
## Training
The below mentioned commands trains a given model on a dataset and performs all the experiments mentioned in the paper.
### Sentiment Classification (SC)
```
export CUDA_VISIBLE_DEVICES=1
python -u main.py \
--exp_name ende \
--task sc \
--domain music \
--exp_id 1 \
--multi_view \
--loss_func kl \
--max_length 256 \
--trans_lang de \
--dynamic_mix \
--mix_train \
--batch_size 12 \
--lr 1e-6 \
--weight_decay 0.0001 \
--kl1 5 \
--dropout 0.5 \
--ptm_folder xlm-roberta-base \
--embed_size 768 \
--hidden_size 768 \
--dynamic_ratio 0.7 \
--dev_ratio 0.8 \
--mapping_for_mix dict/sc_dict/new_music/en2de_20000_onto_for_mix.dict
```
### Dialogue State Tracking (DST)
```
export CUDA_VISIBLE_DEVICES=1
python -u main.py \
--exp_name ende \
--task dst \
--exp_id 1 \
--trans_lang de \
--dynamic_mix \
--mix_train \
--batch_size 8 \
--lr 1e-5 \
--bidirection \
--ptm_folder xlm-roberta-base \
--embed_size 768 \
--hidden_size 768 \
--dynamic_ratio 0.5 \
--dev_ratio 0.75 \
--dropout 0.5 \
--kl1 1.0 \
--mapping_for_mix data/dst/dst_vocab/en2de_muse_onto_for_mix.dict
```
## Citation
Please cite the following paper if you use the code:
```
@inproceedings{lai-etal-2021-saliency-based,
    title = "Saliency-based Multi-View Mixed Language Training for Zero-shot Cross-lingual Classification",
    author = "Lai, Siyu  and
      Huang, Hui  and
      Jing, Dong  and
      Chen, Yufeng  and
      Xu, Jinan  and
      Liu, Jian",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.55",
    pages = "599--610",
}
```

