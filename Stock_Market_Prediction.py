import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quandl
import sklearn
import streamlit as st
import alpha_vantage

data = quandl.get("NSE/TATAGLOBAL")

#Dataframe Analysis


data['Close - Open'] = data['Close'] - data['Open']
data['High - Low'] = data['High'] - data['Low']

data['Next day Close'] = data['Close'].shift(-1)
data['Y']=pd.np.where(data['Next day Close']>data['Close'],1,-1)
data = data.drop(columns = ['Next day Close'])

#Adding 20 SMA
window_size = 20  
data['20 SMA'] = data['Close'].rolling(window=window_size).mean()
data = data.dropna()
#adding VWAP
def calculate_vwap(data):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    cumulative_volume = data['Total Trade Quantity'].cumsum()
    cumulative_volume_price = (data['Total Trade Quantity'] * typical_price).cumsum()
    vwap = cumulative_volume_price / cumulative_volume
    return vwap

data['VWAP'] = calculate_vwap(data)


#MODEL CREATION
X=data[['Close - Open','High - Low','VWAP']]
y=data['Y']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44)

from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=1)
model = knn.fit(X_train,y_train)
#-----Model Created------

#-------- STREAMLIT CODE--------
st.title("Stock Market Prediction")

if st.sidebar.button("Give Stock Details"):
   user_input = st.text_input("Enter Stock Ticker",'IBM')
   df = quandl.get("WIKI/"+user_input)

   st.subheader("Historical Data")
   st.write(df.describe())

#figure
   st.subheader("Closing Price vs Time chart")
   fig = plt.figure(figsize=(12,6))
   plt.plot(df.Close)
   st.pyplot(fig)

#Details Entry
st.subheader("Let's begin the Prediction!!")
sym = st.text_input("Enter Stock ticker",'IBM')
from alpha_vantage.timeseries import TimeSeries
api_key = 'H0EQ5EILH03LCDRV'
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_intraday(symbol=sym, interval='1min', outputsize='compact')
data['VWAP'] = (data['1. open'] + data['2. high'] + data['3. low'] + data['4. close']) / 4

# Print today's stock details
today_data = data.iloc[0]  # Today's first row
user_open = today_data['1. open']
user_high = today_data['2. high']
user_low = today_data['3. low']
user_close = today_data['4. close']
user_vwap = today_data['VWAP']



if st.sidebar.button("Calculate Predictions"):
    st.write("Today's open:",+user_open)
    st.write("Today's high",+user_high)
    st.write("Today's low:",+user_low)
    st.write("Today's close:",+user_close)
    st.write("Today's VWAP:",+user_vwap)
    close_minus_open = user_close - user_open
    high_minus_low = user_high - user_low




#testing(use this for streamlit inputs)
    custom_data = pd.DataFrame({
    'Close - Open': [close_minus_open],
    'High - Low': [high_minus_low],
    'VWAP':[user_vwap]
    })

# Make predictions on your custom data
    predictions = model.predict(custom_data)

#Printing Results
    def display_result(result, color):
      st.write(f"<span style='color:{color}; font-size:20px;'><b>{result}</b></span>", unsafe_allow_html=True)


    result_buy = "BUY!!"
    result_sell = "SELL!!"

    st.write("#### According to the Predictions you should: ")
    if predictions[0] == 1 :
      display_result(result_buy, "green")
    else:
      display_result(result_sell, "red")
    st.text("for tomorrow's close")

#Printing Disclaimer
st.caption("### Disclaimer: This is only for educational purpose. Do not invest real money from this prediction.")
