import torch
import torch.nn as nn
from src.classification.embeddings import get_marker_embeddings

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
            self.to(torch.device('cuda'))

    def __repr__(self):
        return f'''<{self.__class__.__name__}: emb_model = {self._emb}>'''

    def train(self, mode=True):
        super().train(mode)
        return self

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)

    def forward(self, sentences, entities_1, entities_2):
        # embed sentences (batch_size, seq_length) -> (batch_size, max_length, emb_dim)
        emb_tokens, att_tokens, encodings = self._emb(sentences)

        # prepare sentence embedding tensor (batch_size, emb_dim)
        emb_sentences = torch.zeros((emb_tokens.shape[0], emb_tokens.shape[2] * 2), device=emb_tokens.device)

        # iterate over sentences and pool relevant tokens
        for sidx in range(emb_tokens.shape[0]):
            emb_sentences[sidx, :] = self._emb_pooling(emb_tokens[sidx, :torch.sum(att_tokens[sidx]), :], encodings[sidx], entities_1[sidx], entities_2[sidx])

        # reshape for BiLSTM, expected (batch, seq_len, input_size)
        emb_sentences = emb_sentences.view(emb_sentences.shape[0], -1, self._emb.emb_dim)

        print("Embedding Sentences: ", emb_sentences.shape)

        # pass embeddings through BiLSTM
        lstm_out, _ = self.bilstm(emb_sentences)

        print("Bare Bones LSTM Output:", lstm_out.shape)

        # reshaping for linear layer, back to (batch_size, emb_dim)
        lstm_out = lstm_out[:, -1, :] # get the last output for each sequence in the batch

        print("LSTM after reshaping: ", lstm_out.shape)

        # lstm_out = lstm_out.view(lstm_out.shape[1], lstm_out.shape[0])

        # set embedding attention mask to cover each sentence embedding
        att_sentences = torch.ones((att_tokens.shape[0], 1), dtype=torch.bool)

        # logits for all tokens in all sentences + padding -inf (batch_size, max_len, num_labels)
        logits = torch.ones(
            (att_sentences.shape[0], att_sentences.shape[1], self._label_types), device=lstm_out.device
        ) * float('-inf')

        print("LSTM output: ", lstm_out.shape)

        # pass through classifier
        flat_logits_tot = self._lbl(lstm_out)  # (num_words, num_labels)

        logits[att_sentences, :] = flat_logits_tot  # (batch_size, max_len, num_labels)
        predictions_tot = self.get_labels(logits.detach())

        results = {
            'labels': predictions_tot,
            'flat_logits': flat_logits_tot
        }

        return results

    def get_labels(self, logits):
        # get predicted labels with maximum probability (padding should have -inf)
        labels = torch.argmax(logits, dim=-1)  # (batch_size, max_len)
        # add -1 padding label for -inf logits
        labels[(logits[:, :, 0] == float('-inf'))] = -1

        return labels
    

class LinearClassifier(EmbeddingClassifier):
    def __init__(self, emb_model, label_types, hidden_dim=768, num_layers=2):

        lbl_model = nn.Linear(emb_model.emb_dim * 2, len(label_types))
        # for future reference label types is of length 17

        print("Model Dim:", emb_model.emb_dim)

        super().__init__(
            emb_model=emb_model, lbl_model=lbl_model, label_types=len(label_types)
        )

        # bidirectional LSTM layer
        self.bilstm = nn.LSTM(emb_model.emb_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(len(label_types), hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)


        # move model to GPU if available
        if torch.cuda.is_available():
            self.to(torch.device('cuda'))

    def forward(self, sentences, entities_1, entities_2):
        results = super().forward(sentences, entities_1, entities_2)

        # pass through BiLSTM
        linear_out = results['flat_logits']
        linear_out = linear_out.view(linear_out.shape[0], 1, -1)
        print("linear out: ", linear_out.shape)

        lstm_out, _ = self.bilstm2(linear_out)

        # reshaping for logits, back to (batch_size, emb_dim)
        lstm_out = lstm_out.view(linear_out.shape[0], -1)

        # update the results
        results['flat_logits'] = lstm_out

        return results