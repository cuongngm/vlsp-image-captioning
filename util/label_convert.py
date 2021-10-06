import collections
import torch
import string
from vncorenlp import VnCoreNLP


class LabelConvert:
    def __init__(self, vocab_file, max_length=50):
        vocab = []
        with open(vocab_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                vocab.append(line)
        self.vocab = vocab
        self.vocab_mapper = {'<EOS>': 1, '<SOS>': 2, '<PAD>': 0, '<UNK>': 3}
        for i, item in enumerate(self.vocab):
            self.vocab_mapper[item] = i + 4
        self.vocab_inverse_mapper = {v: k for k, v in self.vocab_mapper.items()}
        self.EOS = self.vocab_mapper['<EOS>']
        self.SOS = self.vocab_mapper['<SOS>']
        self.PAD = self.vocab_mapper['<PAD>']
        self.UNK = self.vocab_mapper['<UNK>']
        self.num_class = len(self.vocab) + 4
        self.max_length = max_length
        # self.rdrsegmenter = VnCoreNLP("tachtu/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

    def encode(self, text):
        """ convert text to label index, add <SOS>, <EOS>, and do max_len padding
                Args:
                    text (str or list of str): texts to convert.
                Returns:
                    torch.LongTensor targets:max_length × batch_size
                """
        table = str.maketrans(dict.fromkeys(string.punctuation))
        if isinstance(text, str):
            text = text.lower()
            text = text.translate(table)
            # words = self.rdrsegmenter.tokenize(text)[0]
            words = text.split(' ')
            text = [self.vocab_mapper[item] if item in self.vocab else self.vocab_mapper['<UNK>'] for item in words]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]
            nb = len(text)
            targets = torch.zeros(nb, self.max_length + 2)
            targets[:, :] = self.PAD
            for i in range(nb):
                targets[i][0] = self.SOS
                targets[i][1: len(text[i]) + 1] = text[i]
                targets[i][len(text[i]) + 1] = self.EOS
            # text = targets
            text = targets.transpose(0, 1).contiguous()
            text = text.long()
        return torch.LongTensor(text)
        # return text

    def decode(self, t):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """

        # texts = list(self.dict.keys())[list(self.dict.values()).index(t)]
        if isinstance(t, torch.Tensor):
            texts = self.vocab_inverse_mapper[t.item()]
        else:
            texts = self.vocab_inverse_mapper[t]
        return texts
