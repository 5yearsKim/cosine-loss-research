import csv
from torch.utils.data import IterableDataset, Dataset

class SimilarityData(Dataset):
    def __init__(self, file_from=[], debug_print=False):
        super().__init__()
        self.file_from = file_from
        self.debug_print = debug_print
        holder = []
        for path in self.file_from:
            delimeter = '\t' if path.endswith('.tsv') else ','
            with open(path, 'r') as fr:
                iter_csv = csv.DictReader(fr, delimiter=delimeter, quoting=csv.QUOTE_NONE)
                for row in iter_csv:
                    sent1, sent2 = row['sentence1'], row['sentence2']
                    score = self._process_score(row['score'])
                    holder.append(sent1, sent2, score)
        self.data = holder

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return  self.data[index]

    # def __iter__(self):
    #     for path in self.file_from:
    #         delimeter = '\t' if path.endswith('.tsv') else ','
    #         with open(path, 'r') as fr:
    #             iter_csv = csv.DictReader(fr, delimiter=delimeter, quoting=csv.QUOTE_NONE)
    #             for row in iter_csv:
    #                 sent1, sent2 = row['sentence1'], row['sentence2']
    #                 score = self._process_score(row['score'])
    #                 if self.debug_print:
    #                     print(sent1,'|', sent2,'|', score)
    #                 yield sent1, sent2, score

    def _process_score(self, score):
        score = float(score)
        label = score / 5
        if label > 1 or label < 0:
            print('score out of range')
            return 0
        return label