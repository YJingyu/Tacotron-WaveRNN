import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from infolog import log


class WaveRNN(nn.Module):
    def __init__(self, bits, hop_size, num_mels, device):
        super().__init__()
        self.device = device
        self.h_size = 2**bits
        self.upsample = nn.Upsample(scale_factor=hop_size, mode='linear', align_corners=True)
        self.I = nn.Linear(1 + num_mels, self.h_size)
        self.rnn1 = nn.GRU(self.h_size, self.h_size, batch_first=True)
        self.rnn2 = nn.GRU(self.h_size, self.h_size, batch_first=True)
        self.fc1 = nn.Linear(self.h_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.h_size)

        self.num_params()

    def forward(self, x, mels):
        # upsample
        mels = self.upsample(mels)
        mels = mels.transpose(1, 2)

        batch_size = x.size(0)
        h1 = torch.zeros(1, batch_size, self.h_size).to(self.device)
        h2 = torch.zeros(1, batch_size, self.h_size).to(self.device)

        # I
        x = torch.cat([x.unsqueeze(-1), mels], dim=2)
        x = self.I(x)
        res = x

        # rnn1
        x, _ = self.rnn1(x, h1)
        x = x + res
        res = x

        # rnn2
        x, _ = self.rnn2(x, h2)
        x = x + res

        # fc1 & relu
        x = F.relu(self.fc1(x))

        # fc2 & softmax
        x = F.log_softmax(self.fc2(x), dim=-1)

        return x

    def generate(self, mels):
        self.eval()

        output = []
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():
            start = time.time()

            mels = torch.FloatTensor(mels).to(self.device).unsqueeze(0)
            mels = self.upsample(mels)
            mels = mels.transpose(1, 2)

            x = torch.zeros(1, 1).to(self.device)
            h1 = torch.zeros(1, self.h_size).to(self.device)
            h2 = torch.zeros(1, self.h_size).to(self.device)

            seq_len = mels.size(1)
            for i in range(seq_len):
                # I
                x = torch.cat([x, mels[:, i, :]], dim=1)
                x = self.I(x)

                # rnn1
                h1 = rnn1(x, h1)
                x = x + h1

                # rnn2
                h2 = rnn2(x, h2)
                x = x + h2

                # fc1 & relu
                x = F.relu(self.fc1(x))

                # fc2 & softmax
                x = F.softmax(self.fc2(x), dim=1).view(-1)

                # categorical distribution
                distrib = torch.distributions.Categorical(x)
                sample = 2 * distrib.sample().float() / (self.h_size - 1.) - 1.

                output.append(sample)
                x = torch.FloatTensor([[sample]]).to(self.device)

                if (i + 1) % 1000 == 0:
                    log('%i/%i -- Speed: %i samples/sec' % (i + 1, seq_len, (i + 1) / (time.time() - start)))

        self.train()
        return torch.stack(output).cpu().numpy()

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters])
        log('Trainable Parameters: %i' % parameters)
