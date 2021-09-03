import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import codecs


import logging
logger = logging.getLogger()

from sklearn.metrics import accuracy_score, f1_score
import pdb


class DST_Trainer(object):
    def __init__(self, params, dst_model):
        self.dst_model = dst_model
        self.lr = params.lr
        self.params = params

        # Adam optimizer
        self.optimizer = torch.optim.Adam(dst_model.parameters(), lr=self.lr, weight_decay=params.weight_decay)
        self.loss_fn1 = nn.CrossEntropyLoss()
        self.loss_fn2 = nn.MSELoss()
        self.loss_kl = nn.KLDivLoss()
        self.multi_view = params.multi_view

        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.stop_training_flag = False
        # self.best_avg_acc = 0
        self.best_goal_acc = 0
        self.lambda_1 = params.kl1

    def train_step(self, original, utters, acts_request, acts_slot, acts_value, slot_name, slot_labels,
                   request_labels, systems):
        self.dst_model.train()

        original_food_value_pred, original_price_range_value_pred, original_area_value_pred, original_request_value_pred, \
        food_value_pred, price_range_value_pred, area_value_pred, request_value_pred = self.dst_model(
            original, utters, acts_request, acts_slot, acts_value, slot_name, "en", systems)

        # slot value labels
        # slot_labels: (bsz, 3)
        food_label = slot_labels[:, 0]  # (bsz, 1)
        price_range_label = slot_labels[:, 1]  # (bsz, 1)
        area_label = slot_labels[:, 2]  # (bsz, 1)

        self.optimizer.zero_grad()

        original_food_pred_loss = self.loss_fn1(original_food_value_pred, food_label)
        food_pred_loss = self.loss_fn1(food_value_pred, food_label)
        food_kl1_loss = self.loss_kl(F.log_softmax(original_food_value_pred, dim=-1),
                                     F.softmax(food_value_pred, dim=-1))

        original_price_range_pred_loss = self.loss_fn1(original_price_range_value_pred, price_range_label)
        price_range_pred_loss = self.loss_fn1(price_range_value_pred, price_range_label)
        price_kl1_loss = self.loss_kl(F.log_softmax(original_price_range_value_pred, dim=-1),
                                      F.softmax(price_range_value_pred, dim=-1))

        original_area_pred_loss = self.loss_fn1(original_area_value_pred, area_label)
        area_pred_loss = self.loss_fn1(area_value_pred, area_label)
        area_kl1_loss = self.loss_kl(F.log_softmax(original_area_value_pred, dim=-1),
                                     F.softmax(area_value_pred, dim=-1))
        # request label
        # request_labels: (bsz, 7)
        original_request_pred_loss = self.loss_fn2(original_request_value_pred, request_labels)
        request_pred_loss = self.loss_fn2(request_value_pred, request_labels)
        request_kl1_loss = self.loss_kl(F.log_softmax(original_request_value_pred, dim=-1),
                                        F.softmax(request_value_pred, dim=-1))

        original_loss = original_food_pred_loss + original_price_range_pred_loss + original_area_pred_loss + original_request_pred_loss
        codeswitch_loss = food_pred_loss + price_range_pred_loss + area_pred_loss + request_pred_loss

        kl1_loss = food_kl1_loss + price_kl1_loss + area_kl1_loss + request_kl1_loss
        if self.multi_view == True:
            total_loss = original_loss + self.lambda_1 * kl1_loss
        else:
            total_loss = original_loss + codeswitch_loss

        total_loss.backward()
        self.optimizer.step()

        return original_food_pred_loss.item(), original_price_range_pred_loss.item(), original_area_pred_loss.item(), original_request_pred_loss.item(),total_loss.item(),kl1_loss.item()

    def evaluate(self, dataloader, isTestset=False, load_best_model=False):
        with torch.no_grad():
            if load_best_model == True:
                # load best model
                best_model_path = os.path.join(self.params.dump_path, "best_model.pth")
                logger.info("Loading best model from %s" % best_model_path)
                best_model = torch.load(best_model_path)
                self.dst_model = best_model["dialog_state_tracker"]

            self.dst_model.eval()

            # collect predictions and labels
            y_food, y_price, y_area, y_request = [], [], [], []
            pred_food, pred_price, pred_area, pred_request = [], [], [], []
            dialogue_indices = []
            utterances = []
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, (
            dialgue_idx, original, utters, acts_request, acts_slot, acts_values, slot_names, turn_slot_labels,
            turn_request_labels, systems) in pbar:
                # slot labels
                turn_slot_labels = turn_slot_labels.data.cpu().numpy()
                y_food.append(turn_slot_labels[:, 0])
                y_price.append(turn_slot_labels[:, 1])
                y_area.append(turn_slot_labels[:, 2])
                # request labels
                turn_request_labels = turn_request_labels.data.cpu().numpy()
                y_request.append(turn_request_labels)

                food_value_pred, price_range_value_pred, area_value_pred, request_value_pred,_,_,_,_= self.dst_model(
                    utters, original,acts_request, acts_slot, acts_values, slot_names, self.params.trans_lang,systems)

                # slot value prediction
                dialogue_indices.extend(dialgue_idx)
                pred_food.append(food_value_pred.detach().data.cpu().numpy())
                pred_price.append(price_range_value_pred.detach().data.cpu().numpy())
                pred_area.append(area_value_pred.detach().data.cpu().numpy())
                pred_request.append(request_value_pred.detach().data.cpu().numpy())

                # utterances.append(utters)

            # evaluate
            y_food = np.concatenate(y_food, axis=0)
            y_price = np.concatenate(y_price, axis=0)
            y_area = np.concatenate(y_area, axis=0)
            y_request = np.concatenate(y_request, axis=0)
            pred_food = np.concatenate(pred_food, axis=0)
            pred_food = np.argmax(pred_food, axis=1)
            pred_price = np.concatenate(pred_price, axis=0)
            pred_price = np.argmax(pred_price, axis=1)
            pred_area = np.concatenate(pred_area, axis=0)
            pred_area = np.argmax(pred_area, axis=1)

            pred_request = np.concatenate(pred_request, axis=0)
            pred_request = (pred_request > 0.5) * 1.0

            assert len(y_food) == len(y_price) == len(y_area) == len(y_request) == len(pred_food) == len(pred_price) == len(
                pred_area) == len(pred_request) == len(dialogue_indices)

            joint_goal_total, joint_goal_correct = 0, 0
            goal_total, goal_correct = 0, 0
            request_total, request_correct = 0, 0
            food_acc, price_acc, area_acc = 0, 0, 0

            for i in range(len(y_food)):
                y_food_ = y_food[i]
                y_price_ = y_price[i]
                y_area_ = y_area[i]

                dialog_idx = dialogue_indices[i]
                pred_food_ = pred_food[i]
                pred_price_ = pred_price[i]
                pred_area_ = pred_area[i]

                if i == 0: assert dialog_idx == 0

                if dialog_idx != 0:
                    if pre_pred_food_ != self.params.food_class - 1 and pred_food_ == self.params.food_class - 1:
                        pred_food_ = pre_pred_food_

                    if pre_pred_price_ != self.params.price_range_class - 1 and pred_price_ == self.params.price_range_class - 1:
                        pred_price_ = pre_pred_price_

                    if pre_pred_area_ != self.params.area_class - 1 and pred_area_ == self.params.area_class - 1:
                        pred_area_ = pre_pred_area_

                joint_goal_total += 1
                if y_food_ == pred_food_ and y_price_ == pred_price_ and y_area_ == pred_area_:
                    joint_goal_correct += 1


                goal_total += 1
                if y_food_ == pred_food_:
                    food_acc += 1
                    goal_correct += 1

                goal_total += 1
                if y_price_ == pred_price_:
                    price_acc += 1
                    goal_correct += 1

                goal_total += 1
                if y_area_ == pred_area_:
                    area_acc += 1
                    goal_correct += 1

                pre_pred_food_ = pred_food_
                pre_pred_price_ = pred_price_
                pre_pred_area_ = pred_area_

                y_request_ = y_request[i]
                pred_request_ = pred_request[i]
                request_total += 1
                if np.array_equal(y_request_, pred_request_) == True:
                    request_correct += 1

            # 三个slot分别acc
            food_acc_ = food_acc * 1.0 / joint_goal_total
            price_acc_ = price_acc * 1.0 / joint_goal_total
            area_acc_ = area_acc * 1.0 / joint_goal_total

            if load_best_model == True:
                print('food acc:{:.2f},price acc:{:.2f},area acc:{:.2f}'.format(food_acc_, price_acc_, area_acc_))

            joint_goal_acc = joint_goal_correct * 1.0 / joint_goal_total
            goal_acc = goal_correct * 1.0 / goal_total
            request_acc = request_correct * 1.0 / request_total
            avg_acc = (joint_goal_acc + request_acc) / 2

            if isTestset == False:
                if goal_acc > self.best_goal_acc:
                    self.best_goal_acc = goal_acc
                    self.no_improvement_num = 0
                    self.save_model()
                else:
                    self.no_improvement_num += 1
                    logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))

            if self.no_improvement_num >= self.early_stop:
                self.stop_training_flag = True

        return goal_acc, request_acc, joint_goal_acc, avg_acc, self.stop_training_flag

    def save_model(self):
        """
        save the best model (achieve best f1 on slot prediction)
        """
        saved_path = os.path.join(self.params.dump_path, "best_model.pth")
        torch.save({
            "dialog_state_tracker": self.dst_model
        }, saved_path)

        logger.info("Best model has been saved to %s" % saved_path)

class SC_Trainer(object):
    def __init__(self,params, sc_model):
        self.sc_model = sc_model
        self.lr = params.lr
        self.params = params
        self.optimizer = torch.optim.Adam(sc_model.parameters(), lr=self.lr, weight_decay=params.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1)
        self.loss_fn1 = nn.CrossEntropyLoss()
        self.loss_fn2 = nn.MSELoss()
        self.loss_fn3 = nn.CosineEmbeddingLoss()
        self.loss_kl = nn.KLDivLoss()
        self.lr_decay = 5
        self.cur_lr_decay_num = 0
        self.early_stop = params.early_stop
        self.loss_func = params.loss_func
        self.no_improvement_num = 0
        self.stop_training_flag = False
        self.best_acc = 0
        self.kl1 = params.kl1

    def train_step(self, sentence, sc_sentence, label):
        flag= torch.ones(len(label)).cuda()
        self.sc_model.train()
        pred, pred_cs, h, h_cs = self.sc_model(sentence,sc_sentence,label)
        self.optimizer.zero_grad()
        orig_loss = self.loss_fn1(pred, label)
        cs_loss = self.loss_fn1(pred_cs,label)
        mse_loss = self.kl1 * self.loss_fn2(h, h_cs)
        cos_loss = self.kl1 * self.loss_fn3(h,h_cs,flag)
        kl_loss = self.kl1 * self.loss_kl(F.log_softmax(pred), F.softmax(pred_cs))
        if self.loss_func == 'kl':
            total_loss = orig_loss + kl_loss
        elif self.loss_func == 'mse':
            total_loss = orig_loss + mse_loss
        elif self.loss_func == 'cos':
            total_loss = orig_loss + cos_loss
        elif self.loss_func == 'mix':
            total_loss = orig_loss + cs_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), kl_loss.item()

    def evaluate(self, dataloader, isTestset=False, load_best_model=False):
        with torch.no_grad():
            if load_best_model == True:
                best_model_path = os.path.join(self.params.dump_path,"best_model.pth")
                logger.info("Loading best model from %s" % best_model_path)
                best_model = torch.load(best_model_path)
                self.sc_model = best_model["sentiment_classification"]

            self.sc_model.eval()
            labels = []
            preds = []
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i,(sentence, cs_sentence, label) in pbar:
                label = label.data.cpu()
                labels.append(label)

                pred,_,_,_ = self.sc_model(sentence, cs_sentence, label)
                preds.append(pred.detach().data.cpu())

            #evaluate
            preds = np.concatenate(preds, axis=0)
            labels = np.concatenate(labels,axis=0)
            preds = np.argmax(preds,axis=1)

            correct, total = 0, 0
            for i in range(len(preds)):
                pred_ = preds[i]
                label_ = labels[i]

                if pred_ == label_:
                    correct += 1

                total += 1

            acc = correct * 1.0 / total

            if isTestset == False:
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.no_improvement_num = 0
                    self.save_model()
                else:
                    self.no_improvement_num += 1
                    logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))

            # if self.no_improvement_num >= self.lr_decay:
            #     if self.cur_lr_decay_num <= 2:
            #         self.lr_scheduler.step()
            #         self.cur_lr_decay_num += 1
            #         self.no_improvement_num = 0


            if self.no_improvement_num >= self.early_stop:
                self.stop_training_flag = True
                self.cur_lr_decay_num = 0

        return acc, self.stop_training_flag

    def save_model(self):
        """
        save the best model
        """
        saved_path = os.path.join(self.params.dump_path, "best_model.pth")
        torch.save({
            "sentiment_classification": self.sc_model
        }, saved_path)

        logger.info("Best model has been saved to %s" % saved_path)