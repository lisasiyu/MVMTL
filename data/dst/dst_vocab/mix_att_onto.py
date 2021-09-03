import pickle
lang = 'it'
with open("en2"+lang+"_onto_for_mix.dict", "rb") as f:
    mapping_onto = pickle.load(f)
with open("en2"+lang+"_attn_for_mix.dict", "rb") as f:
    mapping_attn = pickle.load(f)
mapping_for_mix = dict(mapping_onto, **mapping_attn)
with open("en2"+lang+"_both_for_mix.dict", "wb") as f:
    pickle.dump(mapping_for_mix, f)