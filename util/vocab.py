import json
import string
from collections import Counter
from random import shuffle
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("../tachtu/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
word_fred = Counter()
table = str.maketrans(dict.fromkeys(string.punctuation))
with open('/data/cuongnm1/dataset/vlsp/train_captions.json', 'r') as f:
    all_data = json.load(f)
    for data in all_data:
        labels = data['captions'].split('\n')
        for label in labels:
            label = label.lower()
            label = label.translate(table)
            label = rdrsegmenter.tokenize(label)[0]
            # label = label.split(' ')
            for word in label:
                # if word.isalpha():
                word_fred.update([word])
word_vocab = [w for w in word_fred.keys() if word_fred[w] > 1]
shuffle(word_vocab)
with open('vocab_tachtu.txt', 'w') as file:
    for word in word_vocab:
        file.write(word + '\n')

