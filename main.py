from config import get_params
from src.utils import init_experiment
from src.dst_loader import get_dst_dataloader
from src.sc_loader import get_sc_dataloader
from src.dst_model_dual import DialogueStateTracker as DialogueStateTrackerDual
from src.trainer import DST_Trainer
from src.sc_model import SentimentClassification
from src.trainer import SC_Trainer

import torch
from tqdm import tqdm
import pickle
import json
import codecs
import numpy as np
import pdb

torch.autograd.set_detect_anomaly(True)


def train_dst(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)

    with codecs.open(params.ontology_class_path, 'r', 'utf8') as f:
        dialogue_ontology = json.load(f)

    # dataloader
    dst_model = DialogueStateTrackerDual(params)
    dst_model.cuda()

    # build trainer
    dst_trainer = DST_Trainer(params, dst_model)
    dataloader_tr, dataloader_val, dataloader_test, train_turns, tgt_val_turns, tgt_test_turns = get_dst_dataloader(
        params, dialogue_ontology)
    for e in range(params.epoch):
        logger.info("============== epoch {} ==============".format(e + 1))
        total_loss_list, kl_loss_list,food_loss_list, price_loss_list, area_loss_list, request_loss_list = [], [], [], [], [], []

        if params.dynamic_mix:
            dataloader_tr, dataloader_val, dataloader_test, _, _, _ = get_dst_dataloader(params, dialogue_ontology,
                                                                                         train_turns, tgt_val_turns,
                                                                                         tgt_test_turns)
        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        for i, (_, original, utters, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels,
                turn_request_labels, turn_systems) in pbar:

            turn_slot_labels, turn_request_labels = turn_slot_labels.cuda(), turn_request_labels.cuda()
            food_loss, price_loss, area_loss, request_loss, total_loss,kl_loss = dst_trainer.train_step(
                original, utters,  acts_request, acts_slot, acts_values, slot_names, turn_slot_labels,
                turn_request_labels, turn_systems)

            total_loss_list.append(total_loss)
            kl_loss_list.append(kl_loss)
            food_loss_list.append(food_loss)
            price_loss_list.append(price_loss)
            area_loss_list.append(area_loss)
            request_loss_list.append(request_loss)

            pbar.set_description(
                "(Epoch {}) FOOD:{:.4f} PRICE:{:.4f} AREA:{:.4f} REQUEST:{:.4f}".format(e + 1,   np.mean(food_loss),
                                                                                        np.mean(price_loss),
                                                                                        np.mean(area_loss),
                                                                                        np.mean(request_loss)))

        logger.info("Finish training epoch {}. total_loss {:.4f}, kl_loss:{:.4f},FOOD:{:.4f} PRICE:{:.4f} AREA:{:.4f} REQUEST:{:.4f}".format(e + 1, np.mean(total_loss),
                                                                                                                         np.mean(kl_loss),
                                                                                                           np.mean(
                                                                                                               food_loss),
                                                                                                           np.mean(
                                                                                                               price_loss),
                                                                                                           np.mean(
                                                                                                               area_loss),
                                                                                                           np.mean(
                                                                                                               request_loss)))

        logger.info("============== Evaluate {} ==============".format(e + 1))
        goal_acc, request_acc, joint_goal_acc, avg_acc, stop_training_flag = dst_trainer.evaluate(dataloader_val,
                                                                                                  isTestset=False)
        logger.info(
            "({}) Goal ACC: {:.4f}. Joint ACC: {:.4f}. Request ACC: {:.4f}. Avg ACC: {:.4f} (Best Goal Acc: {:.4f})".format(
                'en', goal_acc, joint_goal_acc, request_acc, avg_acc, dst_trainer.best_goal_acc))

        goal_acc, request_acc, joint_goal_acc, avg_acc, _ = dst_trainer.evaluate(dataloader_test, isTestset=True)
        logger.info(
            "({}) Goal ACC: {:.4f}. Joint ACC: {:.4f}. Request ACC: {:.4f}. Avg ACC: {:.4f}".format(params.trans_lang,
                                                                                                    goal_acc,
                                                                                                    joint_goal_acc,
                                                                                                    request_acc,
                                                                                                    avg_acc))

        if stop_training_flag == True:
            break

    logger.info("============== Final Test ==============")
    goal_acc, request_acc, joint_goal_acc, avg_acc, _ = dst_trainer.evaluate(dataloader_test, isTestset=True,
                                                                             load_best_model=True)
    logger.info(
        "Goal ACC: {:.4f}. Joint ACC: {:.4f}. Request ACC: {:.4f}. Avg ACC: {:.4f})".format(goal_acc, joint_goal_acc,
                                                                                            request_acc, avg_acc))


def train_sc(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    sc_model = SentimentClassification(params)
    sc_model.cuda()

    sc_trainer = SC_Trainer(params, sc_model)
    dataloader_tr, dataloader_val, dataloader_test = get_sc_dataloader(params)
    for e in range(params.epoch):
        logger.info("============== epoch {} ==============".format(e + 1))
        loss_list = []
        mse_loss_list = []

        if params.dynamic_mix:
            dataloader_tr, dataloader_val, dataloader_test =get_sc_dataloader(params)
        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))

        for i, (sentence, cs_sentence, label) in pbar:
            label = label.cuda()
            loss, mse_loss = sc_trainer.train_step(sentence, cs_sentence, label)
            loss_list.append(loss)
            mse_loss_list.append(mse_loss)

            pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e+1, np.mean(loss)))

        mean_loss = np.mean(loss_list)
        mean_mse_loss = np.mean(mse_loss_list)
        logger.info("Finish traing epoch{}. TOTAL LOSS:{:.4f} MSE LOSS:{:.4f}".format(e+1, mean_loss, mean_mse_loss))

        logger.info("============== Evaluate {} ==============".format(e + 1))
        acc, stop_training_flag = sc_trainer.evaluate(dataloader_val, isTestset=False)
        logger.info("(en) ACC:{:.4f}".format( acc))

        acc, stop_training_flag = sc_trainer.evaluate(dataloader_test, isTestset=True)
        logger.info("({}) ACC:{:.4f}".format(params.trans_lang, acc))

        if stop_training_flag == True:
            break


    logger.info("============== Final Test ==============")
    acc, _ = sc_trainer.evaluate(dataloader_test, isTestset=True, load_best_model=True)
    logger.info("({}) ACC:{:.4f}".format(params.trans_lang, acc))

if __name__ == "__main__":
    params = get_params()
    if params.task == 'dst':
        train_dst(params)
    elif params.task == 'sc':
        train_sc(params)