# Stock-Market-Prediction
--- 
The stock market offers immense potential for wealth generation but poses significant risks due to its volatile nature. Swing trading, a trading strategy focused on capturing short- to medium-term gains, provides an alternative to intraday trading. It involves holding positions for 3–7 days. This Project focuses on improving swing trading strategies using machine learning, specifically the K-Nearest Neighbors (KNN) algorithm.
---
## Technologies Used
- Python
- Jupyter Notebook
- Numpy
- Pandas
- Matplotlib
- SciKit Learn
---
## APIs
- Quandl - To get historical stock data.
- Alpha Vantage - To get realtime stock prices to make a prediction.
---
## Model Training
- The KNN model is trained on the stock data of "TATAGLOBAL".
- Various features such as Price Change, VWAP, SMA 20, ROC are used to improve the model accuracy.
- Hyperparameter Tuning is done to find the best K value for KNN model.
- The model analyzes the historical stock data and provides signals to buy or sell in 3 degrees, "Average", "Medium" or "High".
  <img width="653" alt="image" src="https://github.com/user-attachments/assets/da45b2d5-4f1e-4295-afe8-cfbadee70e37" />

---
## Results
- Model achieves a accuracy of 77%. The accuracy is very high considering the multiclass signals and the stock market scenario.
- While deep learning models like Long Short-Term Memory (LSTM) networks often excel due to their ability to capture long-term dependencies in sequential data, the K-Nearest Neighbors (KNN) method’s instance-based learning approach offers computational efficiency.
<img width="547" alt="image" src="https://github.com/user-attachments/assets/5c3af5ea-ad04-43cd-b226-24e7afa0ff8e" />


# Thank You

