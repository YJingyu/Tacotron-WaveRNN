import os
import sys

sys.path.append('..')

import librosa
import librosa.display
import matplotlib.pyplot as plt

from hparams import hparams


def main():
    os.chdir('../')

    # Setting
    plt.rcParams['font.size'] = 24
    plt.rcParams['font.family'] = 'sans-serif'
    plt.figure(figsize=(32, 16))

    # Get path
    cpp_dir = os.path.join('cpp')
    rtl_dir = os.path.join('rtl')
    cpp_path = os.path.join(cpp_dir, 'output.wav')
    rtl_path = os.path.join(rtl_dir, 'output.wav')
    output_path = os.path.join(rtl_dir, 'plot.png')
    os.makedirs(rtl_dir, exist_ok=True)

    print(f'Loading cpp_wav from {cpp_path}')
    print(f'Loading rtl_wav from {rtl_path}')

    # Load wavs
    cpp = librosa.load(cpp_path)
    rtl = librosa.load(rtl_path)

    # Plot
    librosa.display.waveplot(cpp[0], sr=hparams.sample_rate, label='C++')
    librosa.display.waveplot(rtl[0], sr=hparams.sample_rate, label='RTL')
    plt.legend()

    # Save file
    plt.savefig(output_path, format='png', dpi=50, transparent=True, bbox_inches='tight', pad_inches=0)

    print(f'Saving output into {output_path}')
    print('Finish!!!')


if __name__ == '__main__':
    main()
