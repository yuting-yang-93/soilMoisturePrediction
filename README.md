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
### Basic packages
```
  pandas
  numpy
```

### Data collection
```
  pygeogrids: 0.2.5
  pytesmo: 0.6.11
  ismn: 0.3
```
#### Data preprocessing
```
  scikit-learn: 0.18.1
```
### Model
```
  keras: 2.2.5
```
### Plot
```
  keras: 2.2.5
```

## Usage
```
1. Store the ismn data for each station in data folder
-- e.g. ./data/SCAN/

2. run pieline_generatedData.py
-- merge the ismn data by each station 
-- scrap the SoilGrid data from webpage
-- scrap the coordinate information for each station and store it as csv file.
-- merge ismn and SoilGrid by coordinate information and store it as csv file.

3. run pipeline.py

```
