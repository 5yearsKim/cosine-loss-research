import torch

class Collator:
    def __init__(self, tknzr):
        self.tknzr = tknzr

    def __call__(self, data):
        sent1, sent2, score = zip(*data)
        total_sent = []
        for s1, s2 in zip(sent1, sent2):
            total_sent.append(s1)
            total_sent.append(s2)
        score = torch.tensor(score)
        inputs = self.tknzr(total_sent, return_tensors='pt', padding=True)

        return inputs, score 