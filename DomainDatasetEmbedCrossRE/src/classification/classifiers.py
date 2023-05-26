import torch
import torch.nn as nn
from src.classification.embeddings import get_marker_embeddings
import numpy as np


#
# Base Classifier
#


class EmbeddingClassifier(nn.Module):
    def __init__(self, emb_model, lbl_model, label_types):
        super().__init__()

        # internal models
        self._emb = emb_model
        self._emb_pooling = get_marker_embeddings
        self._lbl = lbl_model
        self._label_types = label_types

        # move model to GPU if available
        if torch.cuda.is_available():
            self.to(torch.device("cuda"))

    def __repr__(self):
        return f"""<{self.__class__.__name__}: emb_model = {self._emb}>"""

    def train(self, mode=True):
        super().train(mode)
        return self

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)

    def forward(self, sentences, entities_1, entities_2, domains):
        # embed sentences (batch_size, seq_length) -> (batch_size, max_length, emb_dim)
        emb_tokens, att_tokens, encodings = self._emb(sentences)

        if torch.cuda.is_available():
            emb_tokens = emb_tokens.cuda()
            att_tokens = att_tokens.cuda()

        domain_dict = {
            "ai": 0,
            "literature": 1,
            "music": 2,
            "news": 3,
            "politics": 4,
            "science": 5,
        }

        domain_ids = []  # mapping domains to domain ids [batch_size]
        for domain in domains:
            domain_ids.append(domain_dict[domain])

        # domain_ids = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        # Create a new tensor to store the result
        data_ids = torch.zeros_like(att_tokens)
        data_ids = torch.zeros((att_tokens.size(0), att_tokens.size(1), 768))
        
        print(data_ids.shape)

        if torch.cuda.is_available():
            data_ids = data_ids.cuda()

        dataset_embedder = nn.Embedding(len(domain_dict), 768)
        dataset_embeds = dataset_embedder.state_dict()["weight"]

        # Iterate over each element in attention_tokens
        for i in range(att_tokens.size(0)):
            for j in range(att_tokens.size(1)):
                if att_tokens[i, j] == 1:
                    # temp = dataset_embeds[domain_ids[i]]
                    # data_ids[i, j] = dataset_embeds[i]
                    data_ids[i, j] = dataset_embeds[domain_ids[i]]

        print(dataset_embeds.shape)
        print("Data ids shape: ", data_ids.shape)
        # print("Token ids shape: ", emb_tokens.shape)

        emb_tokens = data_ids + emb_tokens

        if torch.cuda.is_available():
            emb_tokens = emb_tokens.cuda()


        emb_sentences = torch.zeros(
            (emb_tokens.shape[0], emb_tokens.shape[2] * 2), device=emb_tokens.device
        )
        # iterate over sentences and pool relevant tokens
        for sidx in range(emb_tokens.shape[0]):
            emb_sentences[sidx, :] = self._emb_pooling(
                emb_tokens[sidx, : torch.sum(att_tokens[sidx]), :],
                encodings[sidx],
                entities_1[sidx],
                entities_2[sidx],
            )

        # set embedding attention mask to cover each sentence embedding
        att_sentences = torch.ones((att_tokens.shape[0], 1), dtype=torch.bool)

        # logits for all tokens in all sentences + padding -inf (batch_size, max_len, num_labels)
        logits = torch.ones(
            (att_sentences.shape[0], att_sentences.shape[1], self._label_types),
            device=emb_sentences.device,
        ) * float("-inf")

        # pass through classifier
        flat_logits_tot = self._lbl(emb_sentences)  # (num_words, num_labels)
        logits[att_sentences, :] = flat_logits_tot  # (batch_size, max_len, num_labels)
        predictions_tot = self.get_labels(logits.detach())

        results = {"labels": predictions_tot, "flat_logits": flat_logits_tot}

        return results

    def get_labels(self, logits):
        # get predicted labels with maximum probability (padding should have -inf)
        labels = torch.argmax(logits, dim=-1)  # (batch_size, max_len)
        # add -1 padding label for -inf logits
        labels[(logits[:, :, 0] == float("-inf"))] = -1

        return labels


#
# Head Classifier
#


class LinearClassifier(EmbeddingClassifier):
    def __init__(self, emb_model, label_types):

        lbl_model = nn.Linear(emb_model.emb_dim * 2, len(label_types))

        super().__init__(
            emb_model=emb_model, lbl_model=lbl_model, label_types=len(label_types)
        )