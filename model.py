import numpy as np
import torch
import torch.nn as nn


class CaptionNet(nn.Module):
    def __init__(self,
                 vocab_size=None,
                 emb_dim=256,
                 cnn_feature_size=2048,
                 hidden_dim=256,
                 num_lstm_layers=2,
                 device=None):

        super().__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.n_layers = num_lstm_layers

        self.fc_h0 = nn.Linear(cnn_feature_size, hidden_dim)
        self.fc_c0 = nn.Linear(cnn_feature_size, hidden_dim)
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_lstm_layers)
        self.fc_logits = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image_vectors, captions_ix, teacher_forcing_ratio=1.0):
        """
        Apply the network in training mode.
        :param image_vectors: torch tensor, содержащий выходы inseption. Те, из которых будем генерить текст
                shape: [batch, cnn_feature_size]
        :param captions_ix:
                таргет описания картинок в виде матрицы
        :param teacher_forcing_ratio:
                доля токенов с teacher forcing
        :returns: логиты для сгенерированного текста описания, shape: [batch, word_i, n_tokens]

        Обратите внимание, что мы подаем сети на вход сразу все префиксы описания
        и просим ее к каждому префиксу сгенерировать следующее слово!
        """

        batch_size = captions_ix.shape[0]
        captions_len = captions_ix.shape[1]

        # LSTM state
        hidden = self.fc_h0(image_vectors).repeat(self.n_layers, 1, 1)  # shape: [n_layers, batch_size, hidden_dim]
        cell = self.fc_c0(image_vectors).repeat(self.n_layers, 1, 1)  # shape: [n_layers, batch_size, hidden_dim]

        captions_emb = self.emb(captions_ix)  # shape: [batch_size, captions_len, emb_dim]

        logits = torch.zeros(batch_size, captions_len, self.vocab_size).to(self.device)

        # <SOS> token
        input = captions_emb[:, 0, :].unsqueeze(0)

        for t in range(1, captions_len):
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
            output = self.fc_logits(output.squeeze(0))
            logits[:, t, :] = output

            teacher_force = np.random.random() < teacher_forcing_ratio

            # next token
            if teacher_force:
                input = captions_emb[:, t, :].unsqueeze(0)
            else:
                top_pred = output.argmax(-1)
                input = self.emb(top_pred).unsqueeze(0)

        return logits
