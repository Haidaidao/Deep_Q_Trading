# Multi-DQN: Integrating Time-Scaled Machine Learning Approaches for Enhanced Stock Market Investment Decisions

## Abstract

Predicting stock market trends is a major challenge for scientists due to the noisy and unstable nature of the data. While there have been many studies based on machine learning models, these typically rely on information from a single timeframe, failing to utilize valuable data from varying timeframes. Multi-timeframe trading is a critical and effective method in the financial trading domain. This paper presents a new approach to stock investment decision-making by combining multiple timeframes with ensemble machine learning models. Our method uniquely integrates short-term, mid-term, and long-term analytical perspectives to enhance the decision-making process in the volatile stock investment field. Initially, we use reinforcement learning algorithms to navigate and capitalize on short-term market fluctuations. For mid-term and long-term scopes, different machine learning techniques are applied to identify trends and market patterns from historical data.

## Authors

- Dao Dai Hai
- Le Dang Minh Khoi
- Tran Huynh Ngoc Diep
- Tran Thao Quyen

## Acknowledgement

This repository is a extension from the original github repository of the Paper: [Multi-DQN: an Ensemble of Deep Q-Learning Agents for Stock Market Forecasting - Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna](https://doi.org/10.1016/j.eswa.2020.113820)

Original source code: [multidqn/deep-q-trading](https://github.com/multidqn/deep-q-trading)
 
## Scripts Description

### These files are needed to run the main code:

- **main.py**: the entry point of the application;
- **deepQTrading.py**: used to organize our data in walks and set up the agents, also start learning the short-term agent and determine the long-term and mid-term's market trend;
- **spEnv.py**: the environment used for the agents;
- **global_config.py**: save global value;
- **trend.py**: Method of using TrendWA to find market trends for mid-term and long-term observation;
- **macd.py**: Method of using MACD to find market trends;
- **callback.py**: a module used to log and trace the results.

### Other tools

- **ensemble.py**: can be used to generate the threshold ensemble from the main agents;
- **plotResults.py**: can be used to generate the final ensemble for the LONG+SHORT agent (after running ensemble.py).

## Requirements

- Python 3.6 - 3.7
- Tensorflow2: `pip install tensorflow==1.14.0`
- Keras: `pip install keras==2.3.1`
- H5py: `pip install h5py==2.10.0`
- Keras-RL: `pip install keras-rl`
- OpenAI Gym: `pip install gym`
- Pandas: `pip install pandas`
- sklearn `pip install scikit-learn`
- trendet `pip install trendet`
- XGBoost `pip instasl xgboost`

**Note:**

- Some other dependency may show up, installed it if its required
- You can try`pip install -r requirements.txt` first before installing the mentioned dependency one by one

## Usage

The code needs three positional parameters to be correctly executed:
```python main.py <numberOfActions> <isOnlyShort> <ensembleFolder>```

- To run the **FULL** agent you need to run: `python main.py 3 0 <ensembleFolder>`
- To run the **ONLY LONG** agent you need to run: `python main.py 2 0 <ensembleFolder>`
- To run the **ONLY SHORT** agent you need to run: `python main.py 2 1 <ensembleFolder>`

where the parameter `<ensembleFolder>` is used to set the name of the folder in which you'll get your results.

Our experiment only using full agent `python main.py 3 0 <ensembleFolder>` where all action: Long, Short, Opt-out is in the short-term agent prediction.
