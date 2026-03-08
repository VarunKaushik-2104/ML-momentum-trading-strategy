# 1. Imports
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 2. Feature & Target Setup (from Step 1 data)
features = ['Return_5', 'Return_20', 'Return_60', 'Vol_20', 'SMA_Ratio', 'RSI']

# 3. Walk-Forward Training Function
def walk_forward_train(stock_df, train_end_year=2022):
    models = []
    scalers = []

    years = sorted(stock_df['Date'].dt.year.unique())
    years = [y for y in years if y <= train_end_year]

    for yr in years[2:]:  # start after minimum data
        train_df = stock_df[stock_df['Date'].dt.year < yr]
        val_df = stock_df[stock_df['Date'].dt.year == yr]

        X_train = train_df[features]
        y_train = train_df['Target']
        X_val = val_df[features]
        y_val = val_df['Target']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        acc = accuracy_score(y_val, model.predict(X_val_scaled))
        print(f"Year {yr} Validation Accuracy: {acc:.3f}")

        models.append(model)
        scalers.append(scaler)

    return models[-1], scalers[-1]  # final trained model

# Load Processed Data
import pandas as pd

data = pd.read_csv("data/processed_data.csv")

data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# 4. Train Advanced Model Per Stock
# Train-Test Split (STRICT)
train = data[data['Date'] < '2023-01-01']
test  = data[data['Date'] >= '2023-01-01']


advanced_models = {}
advanced_scalers = {}

for stock in train['Stock'].unique():
    print(f"\nTraining advanced model for {stock}")
    stock_df = train[train['Stock'] == stock]
    model, scaler = walk_forward_train(stock_df)
    advanced_models[stock] = model
    advanced_scalers[stock] = scaler

# 5. Test Prediction (Out-of-Sample)
advanced_predictions = []

for stock in test['Stock'].unique():
    df_test = test[test['Stock'] == stock]

    X_test = df_test[features]
    X_test_scaled = advanced_scalers[stock].transform(X_test)

    probs = advanced_models[stock].predict_proba(X_test_scaled)[:, 1]

    temp = df_test[['Date', 'Stock']].copy()
    temp['Predicted_Prob'] = probs
    advanced_predictions.append(temp)

adv_pred_df = pd.concat(advanced_predictions)
adv_pred_df.to_csv('outputs/advanced_predictions.csv', index=False)

