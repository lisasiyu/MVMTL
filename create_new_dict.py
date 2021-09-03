import pickle
import pdb
import codecs
import json
import argparse
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=int, default=200)
args = parser.parse_args()

dict_path = 'data/dst/en-de.0-5000.txt'
with open(dict_path, 'r', encoding='utf-8') as fsimi:
	simi_lines = [line.strip().split() for line in fsimi.readlines()]
	simi_dict = defaultdict(lambda :[])
	for line in simi_lines:
		simi_dict[line[0]].append(line[1])
# map_path = 'data/dst/dst_vocab/en2it_onto_for_mix.dict'
# with open(map_path, 'rb') as fmap:
# 	map_dict = pickle.load(fmap)
# new_dict = dict(simi_dict,**map_dict)
output = open('data/dst/dst_vocab/en2de_muse_for_mix.dict', 'wb')
pickle.dump(dict(simi_dict), output)

def create_dict(dict_sour, lang_dire, dict_type, candi_num):

	dict_path = dict_sour+'_simi_dict/simi_dict_'+lang_dire+'_'+dict_type+'.txt'
	map_path = 'data/dst/dst_vocab/'+lang_dire+'_onto_for_mix.dict'
	file_path = 'data/dst/dst_data/tok_woz_train_en.json'

	out_path = 'data/new_mapping_dict/'+lang_dire+'_'+dict_sour+'_'+dict_type+'_'+str(args.threshold)+'_'+str(candi_num)+'candi'+'_for_mix.dict'

	with open(dict_path, 'r', encoding='utf-8') as fsimi:
		simi_lines = [line.strip().split('\t') for line in fsimi.readlines()]
		simi_lines = [[line[0]] + [pair.split()[0] for pair in line[1:]] for line in simi_lines]
		simi_dict = {}
		for line in simi_lines:
			simi_dict[line[0]] = line[1:candi_num+1]

	with codecs.open(file_path, 'r', 'utf8') as f:
	    woz_json = json.load(f)
	    dialogue_count = len(woz_json)
	    all_trans = []
	    for i in range(0, dialogue_count):
	        for j, turn in enumerate(woz_json[i]["dialogue"]):
	        	current_transcription = turn["transcript"]
	        	all_trans.append(current_transcription)

	all_sents = ' '.join(all_trans)
	all_sents = all_sents.split()
	all_sents = Counter(all_sents)
	all_sents = all_sents.most_common()

	useful_dict = {}
	count = 0
	for word_pair in all_sents:
		if word_pair[0] in simi_dict.keys():
			useful_dict[word_pair[0]] = simi_dict[word_pair[0]]
			count += 1
		if count == args.threshold:
			break


	with open(map_path, 'rb') as fmap:
		map_dict = pickle.load(fmap)
		# for key in map_dict.keys():
		# 	map_dict[key] = [map_dict[key]]
		for key in useful_dict.keys():
			if key not in map_dict.keys():
				map_dict[key] = useful_dict[key]
	print('Example line:')
	print(list(useful_dict.items())[:10])
	with open(out_path, 'wb') as fdict:
		pickle.dump(map_dict, fdict)

# dict_sour = 'euro'
# lang_dire = 'en2it'
# dict_type = 'semi'
# candi_num = 4
# create_dict(dict_sour, lang_dire, dict_type, candi_num)
# candi_num = 1
# create_dict(dict_sour, lang_dire, dict_type, candi_num)
#
#
# dict_sour = 'euro'
# lang_dire = 'en2it'
# dict_type = 'iden'
# candi_num = 4
# create_dict(dict_sour, lang_dire, dict_type, candi_num)
# candi_num = 1
# create_dict(dict_sour, lang_dire, dict_type, candi_num)
#
# dict_sour = 'euro'
# lang_dire = 'en2de'
# dict_type = 'semi'
# candi_num = 4
# create_dict(dict_sour, lang_dire, dict_type, candi_num)
# candi_num = 1
# create_dict(dict_sour, lang_dire, dict_type, candi_num)
#
# dict_sour = 'euro'
# lang_dire = 'en2de'
# dict_type = 'iden'
# candi_num = 4
# create_dict(dict_sour, lang_dire, dict_type, candi_num)
# candi_num = 1
# create_dict(dict_sour, lang_dire, dict_type, candi_num)