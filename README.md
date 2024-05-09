
## How to run
Just run the main.py file. 
```bash
python main.py
```
You will see the result figure after running, when you turn off the current figure the next figure will be displayed.

## The role of each file
#### main.py: 
```bash
Read datasets; 
Data preprocessing;
Create model;
Training and evaluating models(CNN, GRU, LSTM, Transformer);
Print and plot results;
Compare different models' performance.
```
#### CNN_Model/model.py: 
Build a CNN model.

#### GRU_Model/model.py: 
Build a GRU model.

#### LSTM_Model/model.py: 
Build a LSTM model.

#### Transformer_Model/model.py: 
Build a Transformer model.



## Description of the dataset
#### suspicious tweets.csv
Contains 2 columns: messaga(Content of tweets), label(1:non-suspicious, 0:suspicious)
This dataset was downloaded from https://www.kaggle.com/datasets/syedabbasraza/suspicious-tweets

## package required:
```bash

- numpy==1.19.2
- pandas==1.1.3
- matplotlib==3.3.2
- scikit-learn==0.23.2
- tensorflow==2.3.1

```

