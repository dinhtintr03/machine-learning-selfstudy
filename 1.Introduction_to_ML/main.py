from sklearn.tree import DecisionTreeRegressor
import pandas as pd

melbourne_file_path = 'D:\\Machine_learning\\1.Introduction_to_ML\\melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price

melbourne_features =  ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]

# print(X.describe())
# print(X.head)

melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")

print(X.head())

print("The predictions are")

print(melbourne_model.predict(X.head()))