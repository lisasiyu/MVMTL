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
--kl1 5.0 \
--mapping_for_mix data/dst/dst_vocab/en2de_muse_onto_for_mix.dict