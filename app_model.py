import pandas as pd
cars = pd.read_csv('car_data_cleaned.csv')

import datetime
date_time = datetime.datetime.now()
cars['Age'] = date_time.year - cars['Year']

# Ordinal feature encoding
df = cars.copy()
target = 'Selling_Price'

# Separating X and y
X = df.drop('Selling_Price', axis=1)
Y = df['Selling_Price']

# Build random forest model
from xgboost import XGBRegressor
xg = XGBRegressor()
xg_final=xg.fit(X,Y)

# Saving the model
import joblib
joblib.dump(xg_final,'car_price_predictor')
