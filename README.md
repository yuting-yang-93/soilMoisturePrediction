# Deep Soil Moisture Prediction
Soil moisture is considered to be one of the key factors in the decision-making for
agricultural water management. Unlike surface soil moisture, deep soil moisture
information is much harder to obtain since it can only be retrieved from in situ
sensors. However, most studies focus on forecasting the deep soil moisture based
on its historical data. Therefore, the purpose of this project is to predict hourly
deep soil moisture at depths of 20, 50 and 100 cm by using surface soil moisture
for sites, where no in situ sensors are available. 

## Requirement
### Environment
```
python: 3.6
```
### IDE
```
Pycharm
jupyter notebook
Google Colab
```

### Pacakges Installation
```
  pandas
  numpy
  matplotlib
  plotly
  pyecharts
  pygeogrids: 0.2.5
  pytesmo: 0.6.11
  ismn: 0.3
  scikit-learn: 0.18.1
  keras: 2.2.5
```

## Usage
```
1. Store the ismn data for each station in data folder
-- e.g. ./data/SCAN/

2. run pieline_generatedData.py
-- merge the ismn data by each station 
-- scrap the SoilGrid data from webpage
-- scrap the coordinate information for each station and store it as CSV file.
-- merge ismn and SoilGrid by coordinate information and store it as CSV file.

3. run pipeline.py
-- predict the deep soil moisture by Linear Regression, Exponentially Weighted Linear Regression and deep learning LSTM model.
-- In terms of LSTM model, Bayesian Optimization Techinique is used for hyper-parameter tuning. 
   The optimal weight for a set of hyper-parametres will be stored in the local path.
   The train-val-loss plot will be also generated after given epochs to check whether overfitting.
```
## License
MIT @yuting-yang-93

## Questions
-- if you have any question, you can open a issue or email yuting.yang93@gmail.com
-- if you have any good suggestions, you can PR or email me.
