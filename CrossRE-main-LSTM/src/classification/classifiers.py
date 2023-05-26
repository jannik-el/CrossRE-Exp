import torch
import torch.nn as nn
from src.classification.embeddings import get_marker_embeddings

class EmbeddingClassifier(nn.Module):
    def __init__(self, emb_model, lbl_model, label_types):
        super().__init__()

        self._emb = emb_model
        self._emb_pooling = get_marker_embeddings
        self._lbl = lbl_model
        self._label_types = label_types

        # Add BiLSTM layer
<<<<<<< HEAD
        hidden_size = 768  # Choose an appropriate hidden size
=======
        hidden_size = 768 
>>>>>>> cfb279aabbe76a019786d4ea2a6cee948623abef
        self.bilstm = nn.LSTM(emb_model.emb_dim * 2, hidden_size, bidirectional=True)

        if torch.cuda.is_available():
            self.to(torch.device('cuda'))

    def __repr__(self):
        return f'<{self.__class__.__name__}: emb_model = {self._emb}>'

    def train(self, mode=True):
        super().train(mode)
        return self

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)

    def forward(self, sentences, entities_1, entities_2):
        emb_tokens, att_tokens, encodings = self._emb(sentences)

        emb_sentences = torch.zeros((emb_tokens.shape[0], emb_tokens.shape[2] *2), device=emb_tokens.device)

        for sidx in range(emb_tokens.shape[0]):
            emb_sentences[sidx, :] = self._emb_pooling(emb_tokens[sidx, :torch.sum(att_tokens[sidx]), :], encodings[sidx], entities_1[sidx], entities_2[sidx])

        att_sentences = torch.ones((att_tokens.shape[0], 1), dtype=torch.bool)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Move tensors to the same device as self.bilstm
        emb_sentences = emb_sentences.to(device)
        att_sentences = att_sentences.to(device)

        logits = torch.ones((att_sentences.shape[0], att_sentences.shape[1], self._label_types), device=emb_sentences.device) * float('-inf')

        lstm_out, _ = self.bilstm(emb_sentences.unsqueeze(1))
        lstm_out = lstm_out.squeeze(1)

        lstm_out = lstm_out.to(self._lbl.weight.device)

        # Perform matrix multiplication with modified tensors
        flat_logits_tot = self._lbl(lstm_out)
        # flat_logits_tot = self._lbl(lstm_out.to(self._lbl.weight.device))  # Ensure self._lbl is on the same device as lstm_out

        logits[att_sentences, :] = flat_logits_tot

        predictions_tot = self.get_labels(logits.detach())

        results = {
            'labels': predictions_tot,
            'flat_logits': flat_logits_tot
        }

        return results


    def get_labels(self, logits):
        labels = torch.argmax(logits, dim=-1)
        labels[(logits[:, :, 0] == float('-inf'))] = -1

        return labels


class LinearClassifier(EmbeddingClassifier):
    def __init__(self, emb_model, label_types):
        hidden_size = 1536  # Choose an appropriate hidden size
        lbl_model = nn.Linear(hidden_size, len(label_types))
        # len(label_types) = 17 FYI

        super().__init__(
            emb_model=emb_model, lbl_model=lbl_model, label_types=len(label_types)
        )