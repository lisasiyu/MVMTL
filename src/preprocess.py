from copy import deepcopy
import codecs
import json
import csv
import re
import string
import os
import pickle

import logging
import random
import pdb
logger = logging.getLogger()

def load_woz_data(file_path, language, dialogue_ontology, mapping=None, dynamic_mix=False, dynamic_ratio=0.75):
    """
    This method loads WOZ dataset as a collection of utterances.

    Testing means load everything, no split.
    """
    with codecs.open(file_path, 'r', 'utf8') as f:
        woz_json = json.load(f)

    turns = []
    dialogue_count = len(woz_json)

    logger.info("loading from file {} totally {} dialogues".format(file_path, dialogue_count))
    for idx in range(0, dialogue_count):

        current_dialogue = process_woz_dialogue(woz_json[idx]["dialogue"], language, dialogue_ontology,mapping=mapping,
                                                dynamic_mix=dynamic_mix, dynamic_ratio=dynamic_ratio)
        turns.extend(current_dialogue)

    return turns


def load_woz_train_data(file_path,  language, dialogue_ontology, file_saliency= None,mapping=None, dynamic_mix=False,
                        dynamic_ratio=0.75):
    """
    This method loads WOZ dataset as a collection of utterances.

    Testing means load everything, no split.
    """
    with codecs.open(file_path, 'r', 'utf8') as f:
        woz_json = json.load(f)

    turns = []
    dialogue_count = len(woz_json)

    logger.info("loading from file {} totally {} dialogues".format(file_path, dialogue_count))
    with codecs.open(file_saliency, 'r', 'utf8') as f1:
        saliency = json.load(f1)

    for idx in range(0, dialogue_count):
        current_dialogue = process_train_woz_dialogue(woz_json[idx]["dialogue"], saliency[idx]["dialogue"], language,
                                                      dialogue_ontology,
                                                      mapping=mapping, dynamic_mix=dynamic_mix,
                                                      dynamic_ratio=dynamic_ratio)
        turns.extend(current_dialogue)
    return turns


def process_woz_dialogue(woz_dialogue, language, dialogue_ontology,mapping=None, dynamic_mix=False,
                         dynamic_ratio=0.75):
    """
    Returns a list of (tuple, belief_state) for each turn in the dialogue.
    """
    # initial belief state
    # belief state to be given at each turn
    if language == "english" or language == "en":

        if mapping is not None:
            if mapping['food'] == 'essen':
                mapping["dontcare"] = "es ist egal"

                null_bs = {}
                null_bs["essen"] = "none"
                null_bs["preisklasse"] = "none"
                null_bs["gegend"] = "none"
                null_bs["request"] = []
                informable_slots = ["essen", "preisklasse", "gegend"]
                pure_requestables = ["postleitzahl", "telefon", "adresse"]
                en_informable_slots = ["food", "price range", "area"]
                en_pure_requestables = ["address", "phone", "postcode"]

            elif mapping['food'] == "cibo":
                mapping["dontcare"] = "non importa"

                null_bs = {}
                null_bs["cibo"] = "none"
                null_bs["prezzo"] = "none"
                null_bs["area"] = "none"
                null_bs["request"] = []
                informable_slots = ["cibo", "prezzo", "area"]
                pure_requestables = ["codice postale", "telefono", "indirizzo"]
                en_informable_slots = ["food", "price range", "area"]
                en_pure_requestables = ["address", "phone", "postcode"]

            else:
                raise Exception('Please check your mapping!')

        else:
            null_bs = {}
            null_bs["food"] = "none"
            null_bs["price range"] = "none"
            null_bs["area"] = "none"
            null_bs["request"] = []
            informable_slots = ["food", "price range", "area"]
            pure_requestables = ["address", "phone", "postcode"]

    elif (language == "italian" or language == "it"):
        null_bs = {}
        null_bs["cibo"] = "none"
        null_bs["prezzo"] = "none"
        null_bs["area"] = "none"
        null_bs["request"] = []
        informable_slots = ["cibo", "prezzo", "area"]
        pure_requestables = ["codice postale", "telefono", "indirizzo"]

    elif (language == "german" or language == "de"):
        null_bs = {}
        null_bs["essen"] = "none"
        null_bs["preisklasse"] = "none"
        null_bs["gegend"] = "none"
        null_bs["request"] = []
        informable_slots = ["essen", "preisklasse", "gegend"]
        pure_requestables = ["postleitzahl", "telefon", "adresse"]
    else:
        null_bs = {}
        pure_requestables = None

    prev_belief_state = deepcopy(null_bs)
    dialogue_representation = []

    for idx, turn in enumerate(woz_dialogue):

        current_DA = turn["system_acts"]

        if language == "english" or language == "en":
            trans_train = turn["trans-train"]
        else:
            trans_train = ''

        current_req = []
        current_conf_slot = []
        current_conf_value = []

        for each_da in current_DA:
            if language == "en" and mapping is not None:
                if each_da in en_informable_slots:
                    current_req.append(each_da)
                elif each_da in en_pure_requestables:
                    current_conf_slot.append("request")
                    current_conf_value.append(each_da)
                else:
                    if type(each_da) is list:
                        current_conf_slot.append(each_da[0])
                        current_conf_value.append(each_da[1])
            else:
                if each_da in informable_slots:
                    current_req.append(each_da)
                elif each_da in pure_requestables:
                    current_conf_slot.append("request")
                    current_conf_value.append(each_da)
                else:
                    if type(each_da) is list:
                        current_conf_slot.append(each_da[0])
                        current_conf_value.append(each_da[1])

        if mapping is not None and language == 'en':
            current_req = [mapping[req] for req in current_req]
            current_conf_slot = [mapping[conf_slot] for conf_slot in current_conf_slot]
            current_conf_value = [mapping[conf_value] for conf_value in current_conf_value]
            #lsy改
            if type(current_conf_value) == list and current_conf_value != []:
                current_conf_value = random.choice(current_conf_value)

        current_transcription = turn["transcript"]
        original_transcription = turn["transcript"]
        current_system = turn["system_transcript"]

        # exclude = set(string.punctuation)
        # exclude.remove("'")

        if mapping == None or language != "en":
            current_transcription = current_transcription.lower()
            current_system = current_system.lower()
        else:
            if dynamic_mix:
                for key, value in mapping.items():
                    if random.randint(1, 100) <= 100 * dynamic_ratio:
                        if type(value) == list:
                            value = random.choice(value)
                        if len(key.split()) > 1:
                            if key == "price range":  ## could be price ranges in the utterance
                                current_transcription = current_transcription.replace("price ranges", value)
                                current_system = current_system.replace("price ranges", value)
                            current_transcription = current_transcription.replace(key, value)
                            current_system = current_system.replace(key, value)
                        else:
                            splits = current_transcription.split()
                            system_splits = current_system.split()
                            for i, word in enumerate(splits):
                                if word == key:
                                    if type(value) == list:
                                        splits[i] = random.choice(value)
                                    else:
                                        splits[i] = value
                            for i, word in enumerate(system_splits):
                                if word == key:
                                    if type(value) == list:
                                        system_splits[i] = random.choice(value)
                                    else:
                                        system_splits[i] = value
                            current_transcription = " ".join(splits)
                            current_system = " ".join(system_splits)
            else:
                for key, value in mapping.items():
                    if len(key.split()) > 1:
                        if type(value) == list:
                            value = random.choice(value)
                        if key == "price range":  ## could be price ranges in the utterance
                            current_transcription = current_transcription.replace("price ranges", value)
                            current_system = current_system.replace("price ranges", value)
                        current_transcription = current_transcription.replace(key, value)
                        current_system = current_system.replace(key, value)
                    else:
                        splits = current_transcription.split()
                        system_splits = current_system.split()
                        for i, word in enumerate(splits):
                            if word == key:  # lsy修改
                                if type(value) == list:
                                    splits[i] = random.choice(value)
                                else:
                                    splits[i] = value
                        for i, word in enumerate(system_splits):
                            if word == key:  # lsy修改
                                if type(value) == list:
                                    system_splits[i] = random.choice(value)
                                else:
                                    system_splits[i] = value
                        current_transcription = " ".join(splits)
                        current_system = " ".join(system_splits)

        current_labels = turn["turn_label"]

        turn_bs = deepcopy(null_bs)
        current_bs = deepcopy(prev_belief_state)

        # print "=====", prev_belief_state
        if "request" in prev_belief_state:
            del prev_belief_state["request"]

        current_bs["request"] = []  # reset requestables at each turn

        legal_flag = True

        for label in current_labels:
            (c_slot, c_value) = label
            c_value = c_value.strip()

            # remove those illegal slot value
            if language == "en" and (c_value not in dialogue_ontology[c_slot]["en"]):
                legal_flag = False
                break

            if language == "en" and mapping is not None:
                if mapping[c_slot] in informable_slots:
                    current_bs[mapping[c_slot]] = c_value
                    turn_bs[mapping[c_slot]] = c_value
                elif mapping[c_slot] == "request":
                    current_bs["request"].append(c_value)
                    turn_bs["request"].append(c_value)
            else:
                if c_slot in informable_slots:
                    current_bs[c_slot] = c_value
                    turn_bs[c_slot] = c_value
                elif c_slot == "request":
                    current_bs["request"].append(c_value)
                    turn_bs["request"].append(c_value)

        # if mapping != None and language == "en":
        #     current_bs["request"].replace()
        #     turn_bs["request"].replace()

        if legal_flag == True:
            dialogue_representation.append((idx, original_transcription,current_transcription, current_req, current_conf_slot,
                                            current_conf_value, deepcopy(current_bs), deepcopy(turn_bs),
                                            current_system))

            prev_belief_state = deepcopy(current_bs)

    return dialogue_representation


def process_train_woz_dialogue(woz_dialogue, sal_dialogue, language, dialogue_ontology, mapping=None, dynamic_mix=False,
                               dynamic_ratio=0.75):
    """
    Returns a list of (tuple, belief_state) for each turn in the dialogue.
    """
    # initial belief state
    # belief state to be given at each turn
    if language == "english" or language == "en":

        if mapping is not None:
            if mapping['food'] == 'essen':
                mapping["dontcare"] = "es ist egal"

                null_bs = {}
                null_bs["essen"] = "none"
                null_bs["preisklasse"] = "none"
                null_bs["gegend"] = "none"
                null_bs["request"] = []
                informable_slots = ["essen", "preisklasse", "gegend"]
                pure_requestables = ["postleitzahl", "telefon", "adresse"]
                en_informable_slots = ["food", "price range", "area"]
                en_pure_requestables = ["address", "phone", "postcode"]

            elif mapping['food'] == "cibo":
                mapping["dontcare"] = "non importa"

                null_bs = {}
                null_bs["cibo"] = "none"
                null_bs["prezzo"] = "none"
                null_bs["area"] = "none"
                null_bs["request"] = []
                informable_slots = ["cibo", "prezzo", "area"]
                pure_requestables = ["codice postale", "telefono", "indirizzo"]
                en_informable_slots = ["food", "price range", "area"]
                en_pure_requestables = ["address", "phone", "postcode"]

            else:
                raise Exception('Please check your mapping!')

        else:
            null_bs = {}
            null_bs["food"] = "none"
            null_bs["price range"] = "none"
            null_bs["area"] = "none"
            null_bs["request"] = []
            informable_slots = ["food", "price range", "area"]
            pure_requestables = ["address", "phone", "postcode"]

    elif (language == "italian" or language == "it"):
        null_bs = {}
        null_bs["cibo"] = "none"
        null_bs["prezzo"] = "none"
        null_bs["area"] = "none"
        null_bs["request"] = []
        informable_slots = ["cibo", "prezzo", "area"]
        pure_requestables = ["codice postale", "telefono", "indirizzo"]

    elif (language == "german" or language == "de"):
        null_bs = {}
        null_bs["essen"] = "none"
        null_bs["preisklasse"] = "none"
        null_bs["gegend"] = "none"
        null_bs["request"] = []
        informable_slots = ["essen", "preisklasse", "gegend"]
        pure_requestables = ["postleitzahl", "telefon", "adresse"]
    else:
        null_bs = {}
        pure_requestables = None

    prev_belief_state = deepcopy(null_bs)
    dialogue_representation = []

    idx = 0

    for turn, sal_turn in zip(woz_dialogue, sal_dialogue):

        sal_food = sal_turn["food"]
        sal_price = sal_turn["price"]
        sal_area = sal_turn["area"]
        sal_request = sal_turn["request"]

        current_DA = turn["system_acts"]

        if language == "english" or language == "en":
            trans_train = turn["trans-train"]
        else:
            trans_train = ''

        current_req = []
        current_conf_slot = []
        current_conf_value = []

        for each_da in current_DA:
            if language == "en" and mapping is not None:
                if each_da in en_informable_slots:
                    current_req.append(each_da)
                elif each_da in en_pure_requestables:
                    current_conf_slot.append("request")
                    current_conf_value.append(each_da)
                else:
                    if type(each_da) is list:
                        current_conf_slot.append(each_da[0])
                        current_conf_value.append(each_da[1])
            else:
                if each_da in informable_slots:
                    current_req.append(each_da)
                elif each_da in pure_requestables:
                    current_conf_slot.append("request")
                    current_conf_value.append(each_da)
                else:
                    if type(each_da) is list:
                        current_conf_slot.append(each_da[0])
                        current_conf_value.append(each_da[1])

        if mapping is not None and language == 'en':
            current_req = [mapping[req] for req in current_req]
            current_conf_slot = [mapping[conf_slot] for conf_slot in current_conf_slot]
            current_conf_value = [mapping[conf_value] for conf_value in current_conf_value]
            if type(current_conf_value) == list and current_conf_value != []:
                current_conf_value = random.choice(current_conf_value)

        current_transcription = turn["transcript"]
        original_transcription = turn["transcript"]
        current_system = turn["system_transcript"]

        splits = current_transcription.split()

        for i, word in enumerate(splits):
            sal_food_ = float(sal_food[i])
            if random.randint(1, 100) <= 100 * sal_food_:
                for key, value in mapping.items():
                    if word == key:
                        if type(value) == list:
                            splits[i] = random.choice(value)
                        else:
                            splits[i] = value
        current_transcription_food = " ".join(splits)

        splits = current_transcription.split()
        sal_price_ = float(sal_price[i])
        for i, word in enumerate(splits):
            if random.randint(1, 100) <= 100 * float(sal_price_):
                for key, value in mapping.items():
                    if word == key:
                        if type(value) == list:
                            splits[i] = random.choice(value)
                        else:
                            splits[i] = value
        current_transcription_price = " ".join(splits)

        splits = current_transcription.split()

        sal_area_ = float(sal_area[i])
        for i, word in enumerate(splits):
            if random.randint(1, 100) <= 100 * float(sal_area_):
                for key, value in mapping.items():
                    if word == key:
                        if type(value) == list:
                            splits[i] = random.choice(value)
                        else:
                            splits[i] = value
        current_transcription_area = " ".join(splits)

        splits = current_transcription.split()
        sal_request_ = float(sal_request[i])
        for i, word in enumerate(splits):
            if random.randint(1, 100) <= 100 * float(sal_request_):
                for key, value in mapping.items():
                    if word == key:
                        if type(value) == list:
                            splits[i] = random.choice(value)
                        else:
                            splits[i] = value
        current_transcription_request = " ".join(splits)

        system_splits = current_system.split()

        for key, value in mapping.items():
            if random.randint(1, 100) <= 100 * dynamic_ratio:
                if type(value) == list:
                    value = random.choice(value)
                if len(key.split()) > 1:
                    if key == "price range":  ## could be price ranges in the utterance
                        current_system = current_system.replace("price ranges", value)
                        current_transcription_food = current_transcription_food.replace("price ranges", value)
                        current_transcription_price = current_transcription_price.replace("price ranges", value)
                        current_transcription_area = current_transcription_area.replace("price ranges", value)
                        current_transcription_request = current_transcription_request.replace("price ranges", value)
                    current_system = current_system.replace(key, value)
                    current_transcription_food = current_transcription_food.replace(key, value)
                    current_transcription_price = current_transcription_price.replace(key, value)
                    current_transcription_area = current_transcription_area.replace(key, value)
                    current_transcription_request = current_transcription_request.replace(key, value)
                else:
                    # splits = current_transcription.split()
                    # system_splits = current_system.split()
                    # for i, word in enumerate(splits):
                    #     if word == key:
                    #         if type(value) == list:
                    #             splits[i] = random.choice(value)
                    #         else:
                    #             splits[i] = value
                    for i, word in enumerate(system_splits):
                        if word == key:
                            if type(value) == list:
                                system_splits[i] = random.choice(value)
                            else:
                                system_splits[i] = value
                    current_system = " ".join(system_splits)

        current_labels = turn["turn_label"]

        turn_bs = deepcopy(null_bs)
        current_bs = deepcopy(prev_belief_state)

        # print "=====", prev_belief_state
        if "request" in prev_belief_state:
            del prev_belief_state["request"]

        current_bs["request"] = []  # reset requestables at each turn

        legal_flag = True

        for label in current_labels:
            (c_slot, c_value) = label
            c_value = c_value.strip()

            # remove those illegal slot value
            if language == "en" and (c_value not in dialogue_ontology[c_slot]["en"]):
                legal_flag = False
                break

            if language == "en" and mapping is not None:
                if mapping[c_slot] in informable_slots:
                    current_bs[mapping[c_slot]] = c_value
                    turn_bs[mapping[c_slot]] = c_value
                elif mapping[c_slot] == "request":
                    current_bs["request"].append(c_value)
                    turn_bs["request"].append(c_value)
            else:
                if c_slot in informable_slots:
                    current_bs[c_slot] = c_value
                    turn_bs[c_slot] = c_value
                elif c_slot == "request":
                    current_bs["request"].append(c_value)
                    turn_bs["request"].append(c_value)

        # if mapping != None and language == "en":
        #     current_bs["request"].replace()
        #     turn_bs["request"].replace()

        transcription = (current_transcription_food, current_transcription_price, current_transcription_area,
                         current_transcription_request)

        if legal_flag == True:
            dialogue_representation.append((idx, original_transcription,transcription, current_req, current_conf_slot,
                                            current_conf_value, deepcopy(current_bs), deepcopy(turn_bs),
                                            current_system))

            prev_belief_state = deepcopy(current_bs)
        idx += 1

    return dialogue_representation
#preprocess for sensitive classification
def load_sentence_data(file_path, language, mapping=None, dynamic_mix=False,dynamic_ratio=0.75):
    turns =[]
    pos_path = file_path + "/positive/"
    for file in os.listdir(pos_path):
        filepath = pos_path + file
        temp = []
        for line in open(filepath):
            temp.append(line)
        temp = ''.join(temp)
        turn = process_sc_dialogue(temp, 1, language, mapping=mapping, dynamic_mix=dynamic_mix, dynamic_ratio=dynamic_ratio)
        turns.extend(turn)

    neg_path = file_path + "/negative/"
    for file in os.listdir(neg_path):
        filepath = neg_path + file
        temp = []
        for line in open(filepath):
            temp.append(line)
        temp = ''.join(temp)
        turn = process_sc_dialogue(temp, 0, language, mapping=mapping, dynamic_mix=dynamic_mix, dynamic_ratio=dynamic_ratio)
        turns.extend(turn)
    logger.info("loading from file {} totally {} data".format(file_path, len(turns)))
    return turns

def process_sc_dialogue(orig_sentence,label, language, mapping=None, dynamic_mix=False,dynamic_ratio=0.75):
    split_sentence = orig_sentence.split()
    if language == "en" and mapping is not None:
        if dynamic_mix:
            for i, word in enumerate(split_sentence):
                if random.randint(1, 100) <= dynamic_ratio * 100:
                    for key, value in mapping.items():
                        if key == word:
                            if type(value) == list:
                                split_sentence[i] = random.choice(value)
                            else:
                                split_sentence[i] = value
            cs_sentence = " ".join(split_sentence)
        else:
            for i, word in enumerate(split_sentence):
                for key, value in mapping.items():
                    if key == word:
                        if type(value) == list:
                            split_sentence[i] = random.choice(value)
                        else:
                            split_sentence[i] = value
            cs_sentence = " ".join(split_sentence)
    else:
        cs_sentence = orig_sentence

    representation =[]
    representation.append((orig_sentence, cs_sentence, label))
    return representation
