import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib

# Load dataset
data = pd.read_csv('sales.csv')

# Ensure 'sales' column exists
if 'sales' not in data.columns:
    raise ValueError("Dataset must contain a 'sales' column.")

X = data.drop('sales', axis=1)
y = data['sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))

# Save model
joblib.dump(model, 'xgb_sales_model.pkl')
