# soilMoisturePrediction
Soil moisture is considered to be one of the key factors in the decision-making for
agricultural water management. Unlike surface soil moisture, deep soil moisture
information is much harder to obtain since it can only be retrieved from in situ
sensors. However, most studies focus on forecasting the deep soil moisture based
on its historical data. Therefore, the purpose of this project is to predict hourly
deep soil moisture at depths of 20, 50 and 100 cm by using surface soil moisture
for sites, where no in situ sensors are available. 

## Requirement
* Environment
```
python: 3.6
```

* Data preparation and preprocessing
** basic packages: pandas, numpy
** data collection:
```
  pygeogrids: 0.2.5
  pytesmo: 0.6.11
  ismn: 0.3
```
** data preprocessing:
```
  scikit-learn: 0.18.1
```
