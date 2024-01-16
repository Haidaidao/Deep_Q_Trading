# Multi-DQN: Integrating Time-Scaled Machine Learning Approaches for Enhanced Stock Market Investment Decisions                                                                                                           

## Abstract 

Predicting stock market trends is a major challenge for scientists due to the noisy and unstable nature of the data. While there have been many studies based on machine learning models, these typically rely on information from a single timeframe, failing to utilize valuable data from varying timeframes. Multi-timeframe trading is a critical and effective method in the financial trading domain. This paper presents a new approach to stock investment decision-making by combining multiple timeframes with ensemble machine learning models. Our method uniquely integrates short-term, mid-term, and long-term analytical perspectives to enhance the decision-making process in the volatile stock investment field. Initially, we use reinforcement learning algorithms to navigate and capitalize on short-term market fluctuations. For mid-term and long-term scopes, different machine learning techniques are applied to identify trends and market patterns from historical data.

## Authors

- Dao Dai Hai
- Le Dang Minh Khoi
- Tran Huynh Ngoc Diep
- Tran Thao Quyen


# Info 

## Description

#### These files are needed to run the main code:
* **main.py**: the entry point of the application;
* **deepQTrading.py**: used to organize our data in walks and set up the agents;
* **spEnv.py**: the environment used for the agents;
* **global_config.py**: save global value;
* **trend.py**: Method of using trendet to find market trends;
* **macd.py**: Method of using MACD to find market trends;
* **callback.py**: a module used to log and trace the results.

#### Other tools:
* **ensemble.py**: can be used to generate the threshold ensemble from the main agents;
* **splitEnsemble.py**: can be used to generate the final ensemble for the LONG+SHORT agent (after running ensemble.py).


If you want to adapt the code and use it for more markets, you can use the file **utils/parserWeek.py**, to create a weekly resolution dataset.<br>
On the other hand, the file **utils/plotResults.py** can be used to generate a .pdf containing several plots, useful to get information on the testing phase of the algorithm.


## Requirements
* Python 3
* Tensorflow : `pip install tensorflow`
* Keras: `pip install keras`
* Keras-RL: `pip install keras-rl2`
* OpenAI Gym: `pip install gym`
* Pandas: `pip install pandas`

## Usage
The code needs three positional parameters to be correctly executed:<br>
`python main.py <numberOfActions> <isOnlyShort> <ensembleFolder>`<br>
<br>

* To run the **FULL** agent you need to run: `python main.py 3 0 ensembleFolder`
* To run the **ONLY LONG** agent you need to run: `python main.py 2 0 ensembleFolder`
* To run the **ONLY SHORT** agent you need to run: `python main.py 2 1 ensembleFolder`

where the paramenter **ensembleFolder** is used to set the name of the folder in which you'll get your results.
