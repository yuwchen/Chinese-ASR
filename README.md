# Chinese-ASR

This repository is a step-by-step tutorial of finetuning the pretrained Chinese ASR model from speechbrain on your own dataset.  
Link to huggingface: https://huggingface.co/speechbrain/asr-transformer-aishell


## Download the pretrained ASR model
| LinkA (original author) | LinkB | 
|:------:|:------:| 
|[google drive](https://drive.google.com/drive/folders/1noVw2hCwMIEt6Ovn4wt6DvrxqB2tT-Q1?usp=sharing)|google drive|


## data preprocessing

Prepare the train.csv, dev.csv, and test.csv  
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
transcript = ws([transcript])[0]

```
ckiptagger: [Github](https://github.com/ckiplab/ckiptagger) 


#### Convert traditional Chinese to simplified Chinese  

```
import opencc
converter = opencc.OpenCC('t2s.json')
transcript = converter.convert(transcript)
```


## Finetuning



## Inferences
