import streamlit as st
import pandas as pd
import numpy as np
import sklearn.externals as extjoblib
import joblib
from xgboost import XGBRegressor

st.set_page_config(page_title="Car Price Prediciton")
centered_title = "<h1 style='text-align: center;'>CAR PRICE PREDICTION</h1>"
st.write(centered_title, unsafe_allow_html=True)

centered_text = "<p style='text-align: center;'>We will help you predict the price of your Car. Please provide us with the necessary information, and we will generate an estimate of your car's price!</p>"
st.write(centered_text, unsafe_allow_html=True)

image = 'car.jpg'
st.image(image)

st.write("""
Dataset obtained from [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=car+data.csv).

For more information [Click Here](https://github.com/Suzanne-11/Car-Prediction/blob/main/info.txt).
""")

st.sidebar.header('Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/Suzanne-11/Car-Prediction/blob/main/car_data_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Year = st.sidebar.slider('Year', 2000,2023,2017)
        Owner = st.sidebar.slider('NO. OF PREVIOUS OWNERS',0, 10, 2)
        Present_Price = st.sidebar.slider('PRESENT PRICE (In Lakhs)', 0.50,50.00,20.00)
        Kms_Driven = st.sidebar.slider('DISTANCE DRIVEN(In kms)',0, 50000, 20000)
        Fuel_Type = st.sidebar.selectbox('FUEL TYPE (0: Petrol, 1: Diesel, 2: CNG)',(0,1,2))
        Seller_Type = st.sidebar.selectbox('SELLER TYPE (0: Dealer, 1: Individual)',(0,1))
        Transmission = st.sidebar.selectbox('TRANSMISSION (0: Manual, 1: Automatic)',(0,1))
        Age = st.sidebar.slider('AGE OF THE CAR',0, 25, 10)
        
        data = {'Year': Year,
                'Owner': Owner,
                'Present_Price': Present_Price,
                'Kms_Driven': Kms_Driven,
                'Fuel_Type': Fuel_Type,
                'Seller_Type': Seller_Type,
                'Transmission': Transmission,
                'Age':Age
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

cars_raw = pd.read_csv('car_data_cleaned.csv')
cars = cars_raw.drop(columns=['Selling_Price'])
df = pd.concat([input_df,cars],axis=0)

# Displays the user input features
st.subheader('User Input features')
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

model = joblib.load('car_price_predictor')
prediction=model.predict(input_df)
decimal_array = np.round(prediction, decimals=2)

st.write('Click the button below to predict the price of your car')
if st.button('Predict'):
    st.header(f"Predicted Price of the Car is Rs. {decimal_array} Lakhs!")


