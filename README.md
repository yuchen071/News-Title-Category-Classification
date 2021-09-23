# News Title Category Classification
## Instructions
You are given a `train.csv` file that contains the corresponding label for BBC news with title and content. You are required to implement a recurrent neural network (RNN, LSTM or GRU) and transformer to correctly classify the news in `test.csv`. An example output format `sample_output.csv` of the script is provided in the `new_data` folder.

## Requirements
The root folder should be structured as follows:
```
root
  ├─ 📁news data
  |  ├─ 📄 test.csv
  |  └─ 📄 train.csv
  ├─ 📄 RNN.py
  └─ 📄 Transformer.py
```
### Dependecies
* torchtext==0.9.0
* tqdm==4.62.2
* pandas==1.1.3
* torch==1.8.0
* spacy==3.1.3
* matplotlib==3.3.4
* numpy==1.19.2

## Train
Run the following code to train with RNN:  
```
python rnn.py
```

Run the following code to train with Transformer:
```
python transformer.py
```

Both scripts should produce `output.csv` files which contain the news title ID and the predicted category of the news title from `test.csv`.  
