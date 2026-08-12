import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("Housing.csv")

# 2. Select features and target
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create model
model = LinearRegression()

# 5. Train model
model.fit(X_train, y_train)

# 6. Predict test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# 8. Predict a new house
new_house = pd.DataFrame(
    [[2000, 3, 2]],
    columns=['area', 'bedrooms', 'bathrooms']
)

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])