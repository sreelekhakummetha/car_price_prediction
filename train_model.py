import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

data = pd.read_csv("car data.csv")

data.replace({
    'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
    'Seller_Type': {'Dealer': 0, 'Individual': 1},
    'Transmission': {'Manual': 0, 'Automatic': 1}
}, inplace=True)

X = data.drop(['Selling_Price', 'Car_Name', 'company'], axis=1)
y = data['Selling_Price']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("✅ Model saved successfully")
