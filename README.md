# Tacotron-WaveRNN
Tacotron + WaveRNN synthesis

<p align="center">
<img src="https://user-images.githubusercontent.com/25808682/50381000-b151ed00-06be-11e9-9e6c-369b396ef741.png" width="75%" />
</p>

Makes use of:
 - Tacotron: https://github.com/Rayhane-mamah/Tacotron-2
 - WaveRNN: https://github.com/fatchord/WaveRNN

 You'll at least need python3, PyTorch 0.4.1, Tensorflow and librosa.

## Preprocess
```
python3 preprocess.py --model='WaveRNN'
```
Default parameters:

|name|default||
|----|----|----|
|--base_dir|||
|--hparams||ex) 'wavernn_gpu_num=4, wavernn_batch_size=16'|
|--model|'Tacotron'|'Tacotron', 'WaveRNN'|
|--dataset|'LJSpeech-1.1'|'LJSpeech-1.0', 'LJSpeech-1.1', 'M-AILABS'|

Others, look at this [file](https://github.com/h-meru/Tacotron-WaveRNN/blob/master/preprocess.py#L78)...

## Training
```
python3 train.py --model='Tacotron-2' --GTA --use_cuda
```

If you would like to train separately...
```
# Tacotron
python3 train.py --model='Tacotron'

# Tacotron synth
python3 synthesize.py --model='Tacotron' --mode='synthesis' --GTA

# WaveRNN
python3 train.py --model='WaveRNN' --use_cuda
```
Default parameters:

|name|default||
|----|----|----|
|--base_dir|||
|--hparams||ex) 'wavernn_gpu_num=4, wavernn_batch_size=16'|
|--model|'Tacotron-2'|'Tacotron-2', 'Tacotron', 'WaveRNN'|
|--mode|'synthesis'|'eval', 'synthesis', 'live'|
|--init|False|True, False|
|--slack_url||{your slack wabhook url...}|
|--use_cuda|False|True, False|

Others, look at this [file](https://github.com/h-meru/Tacotron-WaveRNN/blob/master/train.py#L103)...

## Synthesis
```
python3 synthesize.py --model='Tacotron-2' --text_list={your text file}
```
Default parameters:

|name|default||
|----|----|----|
|--base_dir|||
|--hparams||ex) 'wavernn_gpu_num=4, wavernn_batch_size=16'|
|--model|'Tacotron-2'|'Tacotron-2', 'Tacotron', 'WaveRNN'|
|--mode|'eval'|'eval', 'synthesis', 'live'|
|--text_list||{your text file...}|
|--use_cuda|False|True, False|

Others, look at this [file](https://github.com/h-meru/Tacotron-WaveRNN/blob/master/synthesize.py#L52)...

## Pretrained Model(old)
https://github.com/h-meru/Tacotron-WaveRNN/files/2444777/wavernn_model.zip

## Samples(old)
https://github.com/h-meru/Tacotron-WaveRNN/files/2444792/Samples_730k.zip
