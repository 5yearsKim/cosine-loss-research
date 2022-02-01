from . import SentenceEvaluator, SimilarityFunction
import numpy as np
from typing import List
import torch.nn as nn
from .utils import pearson_r, spearman_correlation 
import torch

class EmbeddingSimilarityEvaluator(SentenceEvaluator):
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], batch_size: int = 16, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = False, write_csv: bool = True):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]

        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    @classmethod
    def from_input_examples(cls, examples, **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        labels = self.scores

        embeddings1 = torch.tensor(embeddings1)
        embeddings2 = torch.tensor(embeddings2)
        labels = torch.tensor(labels)

        sim = self.cos_sim(embeddings1, embeddings2)

        pearson = pearson_r(sim, labels)
        spearman = spearman_correlation(sim, labels)

        print(pearson, spearman)

        return spearman.item()
        
        # cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        # manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        # euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        # dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]


