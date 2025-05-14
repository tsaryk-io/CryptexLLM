<div align="center">
  <h2><b>Time-LLM for Cryptocurrency Price Prediction</b></h2>
</div>



<div align="center">

<p>A fork of Time-LLM adapted for cryptocurrency price forecasting</p>

**[<a href="https://arxiv.org/abs/2310.01728">Original Paper Page</a>]**
**[<a href="https://www.youtube.com/watch?v=6sFiNExS3nI">YouTube Talk 1</a>]**
**[<a href="https://www.youtube.com/watch?v=L-hRexVa32k">YouTube Talk 2</a>]**
**[<a href="https://medium.com/towards-data-science/time-llm-reprogram-an-llm-for-time-series-forecasting-e2558087b8ac">Medium Blog</a>]**
</div>

<p align="center">

<img src="./figures/logo.png" width="70">

</p>

---

>
> This repository is a fork of [Time-LLM](https://github.com/KimMeen/Time-LLM), adapted specifically for cryptocurrency price prediction. We leverage the Time-LLM framework to forecast Bitcoin prices using historical OHLCV (Open, High, Low, Close, Volume) data.
>

## Introduction
Time-LLM is a reprogramming framework to repurpose LLMs for general time series forecasting with the backbone language models kept intact.
Notably, the authors show that time series analysis (e.g., forecasting) can be cast as yet another "language task" that can be effectively tackled by an off-the-shelf LLM.


<p align="center">
<img src="./figures/framework.png" height = "360" alt="" align=center />
</p>

- Time-LLM comprises two key components: (1) reprogramming the input time series into text prototype representations that are more natural for the LLM, and (2) augmenting the input context with declarative prompts (e.g., domain expert knowledge and task instructions) to guide LLM reasoning.

<p align="center">
<img src="./figures/method-detailed-illustration.png" height = "190" alt="" align=center />
</p>

## Datasets
You can access the well pre-processed datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity (ECL), Traffic, ILI, and M4) from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), or using ```gdown``` with the following command:

```bash
pip install gdown
gdown 1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP
```

Then place the downloaded contents under `./dataset`.

## Cryptex Dataset

The [Cryptex dataset](http://crypto.cs.iit.edu/datasets/download.html) contains high-resolution OHLCV (Open, High, Low, Close, Volume) time series data for the BTC-USDT trading pair, extracted from the Binance.us exchange between September 2019 and July 2023. It features:
- **Time granularity:** Down to 1-second candlesticks
- **Data columns:** timestamp, open, close, high, low, volume
- **Timestamp format:** UNIX time

Subsets (daily/hourly) are already available under ```./dataset```


## Requirements
Use Python 3.11 from MiniConda

- torch==2.2.2
- accelerate==0.28.0
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.12.0
- tqdm==4.65.0
- peft==0.4.0
- transformers==4.31.0
- deepspeed==0.14.0
- sentencepiece==0.2.0

To install all dependencies:
```
pip install -r requirements.txt
```

## Training
1. Download datasets and place them under `./dataset`.
2. Edit domain-specific prompts in ```./dataset/prompt_bank```.
3. A few experiment setups with different configurations are provided in `./scripts/TimeLLM_Cryptex.sh`. Edit the script to run an experiment of your choosing with:

```bash
bash ./scripts/TimeLLM_Cryptex.sh
```
4. Trained models are saved to ```./trained_models```.

### Key hyperparameters
- **seq_len:** Input sequence length.
- **label_len:** Label length.
- **pred_len:** Output prediction window.
- **features:** Forecasting task. Options: ```M``` for multivariate predict multivariate, ```S``` for univariate predict univariate, and ```MS``` for multivariate predict univariate.
- **patch_len:** Patch length. 16 by default, reduce for short term tasks.
- **stride:** Stride for patching. 8 by default, reduce for short term tasks.

Please refer to `run_main.py` for more detailed descriptions of each hyperparameter. Time-LLM also defaults to supporting Llama-7B and includes compatibility with two additional smaller PLMs (GPT-2 and BERT). Simply adjust `--llm_model` and `--llm_dim` to switch backbones.



## Inference
1. For **standard forecasting** (predicting future timestamps starting from the end of the input file), run ```inference.py``` using the following script:
```bash
bash ./scripts/inference.sh
```
Inside the script, specify:
- **model_path:** Path to the trained model.
- **data_path:** Path to the dataset.
- **model_id:** Model ID given during training.

This script outputs predictions to terminal.


2. For **autoregressive forecasting**, run ```inference_ar.py``` using the following script:
```bash
bash ./scripts/inference_ar.sh
```
Inside the script, also specify:
- **output_path:** Output file name in the following format: `./predictions/<filename>.csv`


This script autoregressively generates multi-step forecasts and saves output CSV in `./predictions` with the following format:
```
timestamp | open | close | high | low | volume | close_predicted_1 | ... | close_predicted_n
```
Where ```n``` is the prediction length ```pred_len```. For example, with `seq_len=24` and `pred_len=6`:
- Row 24 will contain 6 future predictions made using rows 0-23
- Row 25 will contain 6 future predictions made using rows 1-24
- Row 26 will contain 6 future predictions made using rows 2-25

And so on...



## Acknowledgments
This project is based on the original [Time-LLM](https://github.com/KimMeen/Time-LLM) paper and implementation:

```bibtex
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}

```
We thank the authors for their foundational work.


