# Cryptocurrency Predictors

A self hosted program to train, evaluate and deploy keras model bots that predict cryptocurrency values using publically available data

## How to Run

1. Clone the repo
   ```bash
   git clone https://github.com/Arihasaheadache/crypto-bot.git
   ```
2. Navigate to project directory
   ```bash
   cd crypto-bot/src
   ```
3. Run install.py (Python 3.11 or above is recommended)
   ```python
   python install.py
   ```
4. Once dependencies are installed, run main.py to see results of pre-trained models
   ```python
   python main.py
   ```

## How to Train Models

The program fetches new data and trains the model on it, for best results run the training file every month or so. 

```python
python train.py
```
You should see the models in `/models` directory

#### P.S. Check out the ipynb (jupyter notebook) file for data visualisation of the models
