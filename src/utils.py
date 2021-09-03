import os
import subprocess
import pickle
import logging
import time
import random
import codecs
import json
import torch

from datetime import timedelta
import numpy as np
from tqdm import tqdm
import pdb

import logging
logger = logging.getLogger()

def init_experiment(params, logger_filename):
    """
    Initialize the experiment:
    - save parameters
    - create a logger
    """
    # save parameters
    get_saved_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)

    return logger

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def create_logger(filepath):
    # create log formatter
    log_formatter = LogFormatter()
    
    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def get_saved_path(params):
    """
    create a directory to store the experiment
    """
    dump_path = "./" if params.dump_path == "" else params.dump_path
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    assert os.path.isdir(dump_path)

    # create experiment path if it does not exist
    if params.task == 'sc':
        # create experiment path if it does not exist
        exp_path = os.path.join(dump_path, params.task, params.domain, params.exp_name)
    else:
        exp_path = os.path.join(dump_path, params.task, params.exp_name)
    if not os.path.exists(exp_path):
        subprocess.Popen("mkdir -p %s" % exp_path, shell=True).wait()
    
    # generate id for this experiment
    if params.exp_id == "":
        chars = "0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(0, 3))
            if not os.path.isdir(os.path.join(exp_path, exp_id)):
                break
    else:
        exp_id = params.exp_id
    # update dump_path
    params.dump_path = os.path.join(exp_path, exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()
    assert os.path.isdir(params.dump_path)

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

    with codecs.open(params.ontology_mapping_path, 'r', 'utf8') as f:
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

        dialogue_idx, original, utterance, acts_inform_slots, acts_slot_type, acts_slot_value, test_label, turn_label, system = each_turn
        # 为什么？
        labels = test_label if isTestset == True else turn_label

        for label_slot, label_value in labels.items():

            binarized_slots.append(label_slot)
            if label_value == "none":  # food,price range,area value的在class_type_dict的位置[0,0,0]
                binarized_slot_values.append(len(class_type_dict[label_slot]))  # label for none
            else:
                if type(label_value) is list:  # request的位置[0,0,0,0,0,0,0]
                    r_label = [0] * params.request_class
                    r_indices = [class_type_dict[label_slot].index(item) for item in label_value]
                    for idx in r_indices:
                        r_label[idx] = 1
                    binarized_request_values = r_label
                else:
                    binarized_slot_values.append(class_type_dict[label_slot].index(label_value))

        current_turn = (
        dialogue_idx, original, utterance, acts_inform_slots, acts_slot_type, acts_slot_value, binarized_slots,
        binarized_slot_values, binarized_request_values, system)

        binarized_turns.append(current_turn)

    return binarized_turns