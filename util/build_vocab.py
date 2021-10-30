import json
import string
from collections import Counter
from random import shuffle

word_fred = Counter()
table = str.maketrans(dict.fromkeys(string.punctuation))

with open('../dataset/vlsp_train/train_captions.json', 'r') as f:
    all_data = json.load(f)
    for data in all_data:
        labels = data['captions'].split('\n')
        for label in labels:
            label = label.lower()
            label = label.translate(table)
            label = label.split(' ')
            for word in label:
                # if word.isalpha():
                word_fred.update([word])
vlsp_vocab = [w for w in word_fred.keys() if word_fred[w] >= 2]

with open('../dataset/coco_vn/captions_val2014.json', 'r') as f:
    coco_data = json.load(f)
    for idx in range(len(coco_data['annotations'])):
        labels = coco_data['annotations'][idx]['caption']
        labels = labels.lower()
        labels = labels.translate(table)
        label = labels.split(' ')
        for word in label:
            if word.isalpha():
                word_fred.update([word])

with open('../dataset/coco_vn/captions_train2014.json', 'r') as f:
    coco_data = json.load(f)
    for idx in range(len(coco_data['annotations'])):
        labels = coco_data['annotations'][idx]['caption']
        labels = labels.lower()
        labels = labels.translate(table)
        label = labels.split(' ')
        for word in label:
            if word.isalpha():
                word_fred.update([word])

coco_vocab = [w for w in word_fred.keys() if word_fred[w] > 9]
# word_vocab = [w for w in word_fred.keys()]
final_vocab = vlsp_vocab + coco_vocab
final_vocab = list(set(final_vocab))
shuffle(final_vocab)
with open('vocab_coco.txt', 'w') as file:
    for word in final_vocab:
        file.write(word + '\n')
