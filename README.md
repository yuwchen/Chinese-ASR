# Chinese-ASR

This repository describe how to finetune the pretrained Chinese ASR model from speechbrain on your own dataset.  
Link to huggingface: https://huggingface.co/speechbrain/asr-transformer-aishell



## Prepare your data

### Step 1: Prepare the train.csv, dev.csv, and test.csv  

The train.csv, dev.csv, and test.csv need to be in the following format:  

Header: ID,                duration,                wav,             transcript  
       (index of the file, duration of the wavfile, path to wavfile, ground-truth content)  
example:  
```
ID,duration,wav,transcript
0,3.12,data/test/A_1_17_F.wav,跑遍 每 个 剧组 的 庆功宴
1,1.9272562358276644,data/test/A_1_17_M.wav,跑遍 每 个 剧组 的 庆功宴
2,3.408,data/test/A_1_18_F.wav,不 排除 是 游客临时 起意
3,1.9620861678004535,data/test/A_1_18_M.wav,不 排除 是 游客临时 起意
``` 

#### Calculate the duration of utterances: 
```
import librosa

duration = librosa.get_duration(filename=/path/to/wavfile)
```

#### Chinese word segmentation 
```
from ckiptagger import data_utils, construct_dictionary, WS

ws = WS("/path/to/ckip/data_dir")
transcript = '要斷詞的句子'
transcript = ws([transcript])[0]

```
ckiptagger: [Github](https://github.com/ckiplab/ckiptagger) 


#### Convert traditional Chinese to simplified Chinese  

```
import opencc

converter = opencc.OpenCC('t2s.json')
transcript = '要轉成簡體的句子'
transcript = converter.convert(transcript)
```
opencc: [Github](https://github.com/BYVoid/OpenCC) 

### Step 2: Put your data in the "data" folder

. The file path should match the one in the .csv file  
. If you want to put the data in other folder, you need to edit the hparams.yaml files 

```
# hparams/hparams_train.yaml & hparams/hparams_test.yaml

line 23: data_folder: !ref data  # change the "data" to your folder path

```


## Finetuning

### Step 1: Download the pretrained ASR model 

| LinkA (original author) | LinkB | 
|:------:|:------:| 
|[google drive](https://drive.google.com/drive/folders/1noVw2hCwMIEt6Ovn4wt6DvrxqB2tT-Q1?usp=sharing)|google drive|

. Save the downloaded model (CKPT+2021-04-20+23-20-18+00 and tokenizer.ckpt) in the **output/model** folder  

. If you want to save the model in other folder, you need to edit the hparams.yaml files  

```
# hparams/hparams_train.yaml & hparams/hparams_test.yaml

line 15: output_folder: !ref output                   # change the "output" to your folder name
line 19: save_folder: !ref <output_folder>/model      # the code will load the model in "output/model" folder

```

. If you don't put the pretrained model, the model will train from scratch. 

### Step 2: Edit the training parameters

```
# hparams/hparams_train.yaml

Line 35: number_of_epochs: 100 
# Because the pretrained model end at 50 epoch, the model will fintune (number_of_epochs - 50) epochs (e.g. 50 epochs.)
# If you train from scratch (i.e. didn't put the pretrained mdoel in the model folder), the model will train 100 epochs.

```

### Step 3: Run the code

```
python train.py hparams/hparams_train.yaml --train_data=data/train.csv --valid_data=data/dev.csv --test_data=data/test.csv
```
. The finetuned model will be save in the output/model folder  
. The predicted results of testing data will be save in the output/predicted.txt  
. If you want to save the prediciton in other fils, you need to edit the hparams.yaml  

```
# hparams/hparams_train.yaml
Line 17: cer_file: !ref <output_folder>/predicted.txt # change <output_folder>/predicted.txt to the desired output file path
```

## Inferences  

Run ASR using the model described in the hparams_test.yaml (e.g. the model in output/model):  

```
python test.py hparams/hparams_test.yaml --test_data=data/test.csv

# hparams/hparams_test.yaml
Line 15: output_folder: !ref output
Line 19: save_folder: !ref <output_folder>/model
```
