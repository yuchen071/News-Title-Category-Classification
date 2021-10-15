# News Title Category Classification
## About the Project
Two different models (GRU RNN and Transformer) are implemented to classify news categories from their title. A `train.csv` file contains news titles from the BBC with their corresponding labels. 

## Requirements
The root folder should be structured as follows:
```
📁 root
  ├─ 📁 news data
  |  ├─ 📗 test.csv
  |  └─ 📗 train.csv
  ├─ 📄 RNN.py
  └─ 📄 Transformer.py
```
### Dependecies
```
torchtext==0.9.0
tqdm==4.62.2
pandas==1.1.3
torch==1.8.0
spacy==3.1.3
matplotlib==3.3.4
numpy==1.19.2
```

## Usage
### Train
Run the following code to train with RNN:  
```
python rnn.py
```

Run the following code to train with Transformer:
```
python transformer.py
```

Both scripts should produce `output.csv` files which contain the news title ID and the predicted category of the news title from `test.csv`.  

### Parameters
Global parameters can be tinkered in the script:

RNN: 
```python
PATH_TRAIN = "path/to/news_data_train.csv"
PATH_TEST = "path/to/news_data_test.csv"

MAX_SEQ         # text sequence length cutoff, set to 0 for auto max text len
HID_DIM         # hidden dimension of the rnn
RNN_LAYERS      # gru layers
DROP            # dropout

EPOCHS          # epochs
LR              # learning rate
BATCH_SIZE      # batch size
CLIP_GRAD       # clip_grad_norm_
```

Transformer:
```python
PATH_TRAIN = "news_data/train.csv"
PATH_TEST = "news_data/test.csv"

MAX_SEQ         # text sequence length cutoff, set to 0 for auto max text len
NUM_HID         # number of hidden nodes in NN part of trans_encode
NUM_HEAD        # number of attention heads for trans_encode
NUM_LAYERS      # number of trans_encoderlayer in trans_encode
DROPOUT         # dropout

EPOCHS          # epochs
LR              # learning rate
BATCH_SIZE      # batch size
CLIP_GRAD       # clip_grad_norm_
```

## Results
### RNN
* Epochs: 300
* Learning rate: 1e-4
* Batch size: 900

| Loss | Accuracy |
| -- | -- |
| ![rnn_loss](https://github.com/yuchen071/News-Title-Category-Classification/blob/main/results/rnn_loss.png) | ![rnn_acc](https://github.com/yuchen071/News-Title-Category-Classification/blob/main/results/rnn_accuracy.png) |

### Transformer
* Epochs: 300
* Learning rate: 8e-5
* Batch size: 900

| Loss | Accuracy |
| -- | -- |
| ![tra_loss](https://github.com/yuchen071/News-Title-Category-Classification/blob/main/results/transformer_loss.png) | ![tra_acc](https://github.com/yuchen071/News-Title-Category-Classification/blob/main/results/transformer_accuracy.png)