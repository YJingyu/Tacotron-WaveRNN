import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from datasets.audio import save_wavernn_wav
from hparams import hparams_debug_string
from infolog import log
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from wavernn.model import WaveRNN


class AudiobookDataset(Dataset):
    def __init__(self, ids, path):
        self.ids = ids
        self.path = path

    def __getitem__(self, index):
        id = self.ids[index]
        m = np.load(f'{self.path}/mels/{id}.npy')
        x = np.load(f'{self.path}/quant/{id}.npy')
        return m, x

    def __len__(self):
        return len(self.ids)


class CustomCollator():
    def __init__(self, hparams):
        self.bits = hparams.wavernn_bits
        self.pad = hparams.wavernn_pad
        self.hop_size = hparams.hop_size

    def __call__(self, batch):
        mel_win = 5 + 2 * self.pad
        seq_len = self.hop_size * mel_win

        mels = []
        coarse = []
        for x in batch:
            max_offset = x[0].shape[-1] - mel_win
            mel_offset = np.random.randint(0, max_offset)
            sig_offset = mel_offset * self.hop_size
            mels.append(x[0][:, mel_offset:(mel_offset + mel_win)])
            coarse.append(x[1][sig_offset:(sig_offset + seq_len + 1)])

        mels = torch.FloatTensor(np.stack(mels).astype(np.float32))
        coarse = torch.LongTensor(np.stack(coarse).astype(np.int64))

        x_input = 2 * coarse[:, :seq_len].float() / (2**self.bits - 1.) - 1.
        y_coarse = coarse[:, 1:]

        return x_input, mels, y_coarse


def test_generate(model, step, input_dir, ouput_dir, sr, samples=3):
    filenames = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.npy')]
    for i in tqdm(range(samples)):
        mel = np.load(os.path.join(input_dir, filenames[i])).T
        save_wavernn_wav(model.generate(mel), f'{ouput_dir}/{step // 1000}k_steps_{i}.wav', sr)


def train(args, log_dir, input_dir, hparams):
    test_dir = os.path.join(args.base_dir, 'tacotron_output', 'eval')
    save_dir = os.path.join(log_dir, 'wavernn_pretrained')
    eval_dir = os.path.join(log_dir, 'eval-dir')
    eval_wav_dir = os.path.join(eval_dir, 'wavs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'wavernn_model.pyt')

    log('Checkpoint path: {}'.format(checkpoint_path))
    log('Loading training data from: {}'.format(input_dir))
    log('Using model: {}'.format(args.model))
    log(hparams_debug_string())

    # device
    device = torch.device('cuda' if args.use_cuda else 'cpu')

    # Load Dataset
    with open(f'{input_dir}/dataset_ids.pkl', 'rb') as f:
        dataset = AudiobookDataset(pickle.load(f), input_dir)

    collate = CustomCollator(hparams)
    batch_size = hparams.wavernn_batch_size * hparams.wavernn_gpu_num
    data_loader = DataLoader(dataset, collate_fn=collate, batch_size=batch_size, shuffle=True, pin_memory=args.use_cuda)

    # Initialize Model
    model = WaveRNN(hparams.wavernn_bits, hparams.hop_size, hparams.num_mels, device).to(device)

    # Load Model
    if not os.path.exists(checkpoint_path):
        log('Created new model!!!', slack=True)
        torch.save({'state_dict': model.state_dict(), 'global_step': 0}, checkpoint_path)
    else:
        log('Loading model from {}'.format(checkpoint_path), slack=True)

    # Load Parameters
    if args.use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if args.use_cuda:
        model = nn.DataParallel(model).to(device)

    step = checkpoint['global_step']
    log('Starting from {} step'.format(step), slack=True)

    optimiser = optim.Adam(model.parameters(), lr=hparams.wavernn_lr_rate)
    criterion = nn.NLLLoss().to(device)

    # Train
    for e in range(args.wavernn_train_epochs):
        running_loss = 0.
        start = time.time()

        for i, (x, m, y) in enumerate(data_loader):
            x, m, y = x.to(device), m.to(device), y.to(device)

            y_hat = model(x, m).transpose(1, 2)

            loss = criterion(y_hat.unsqueeze(-1), y.unsqueeze(-1))

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            step += 1
            speed = (i + 1) / (time.time() - start)

            log('Step {:7d} [{:.3f} step/sec, avg_loss={:.5f}]'.format(step, speed, avg_loss), end='\r')

        # Save Checkpoint and Eval Wave
        if (e + 1) % 30 == 0:
            log('\nSaving model at step {}'.format(step), end='', slack=True)

            if args.use_cuda:
                torch.save({'state_dict': model.module.state_dict(), 'global_step': step}, checkpoint_path)
                test_generate(model.module, step, test_dir, eval_wav_dir, hparams.sample_rate)
            else:
                torch.save({'state_dict': model.state_dict(), 'global_step': step}, checkpoint_path)
                test_generate(model, step, test_dir, eval_wav_dir, hparams.sample_rate)

        log('\nFinished {} epoch. Starting next epoch...'.format(e + 1))


def wavernn_train(args, log_dir, hparams):
    input_dir = os.path.join(args.base_dir, 'wavernn_data')

    train(args, log_dir, input_dir, hparams)
