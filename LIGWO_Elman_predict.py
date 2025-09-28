import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

data = pd.read_csv(r'D:\predict.txt')

x_data_pr = data.loc[:, ['as', 'de', 'rainfall', 'SPI', 'vegetation', 'distance_roads', 'NDBI', 'NDVI', 'soil', 'lithology', 'aspect', 'slope', 'elevation']]

scaler = StandardScaler()
x_data_pr = scaler.fit_transform(x_data_pr)

x_data_pr = np.expand_dims(x_data_pr, axis=1)

model = joblib.load('LIGWO_Elman_model.pkl')

y_predict = model.predict(x_data_pr)
print("predict:", y_predict, y_predict.shape)

roc_auc_plot = model.predict(x_data_pr)
print("prediction_probability:", roc_auc_plot, y_predict.shape)

pd.DataFrame(roc_auc_plot).to_csv(r'D:\result.csv', header=None)
