import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = pd.read_csv("car data.csv")

# Encode categorical columns
data.replace({
    'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
    'Seller_Type': {'Dealer': 0, 'Individual': 1},
    'Transmission': {'Manual': 0, 'Automatic': 1}
}, inplace=True)

# DROP string columns
X = data.drop(['Selling_Price', 'Car_Name', 'company'], axis=1)
y = data['Selling_Price']

print(X.dtypes)   # 🔥 should show NO object type

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("R2 Score:", r2_score(y_test, y_pred))
