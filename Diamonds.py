import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st


df = pd.read_csv('diamonds.csv')

target = 'price'

plt.figure(figsize=(8, 6))
sns.histplot(df[target], bins=50, kde=True)
plt.title('Distribution of Diamond Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

print(df.info())
print(df.describe())

df = df.drop(['Unnamed: 0'], axis=1)

for column in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], bins=50, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

df = df[(df['price'] < df['price'].quantile(0.99)) & (df['price'] > df['price'].quantile(0.01))]

selected_features = corr[target][corr[target].abs() > 0.5].index.tolist()
selected_features.remove(target)
print(f'Selected Features: {selected_features}')

le = LabelEncoder()
df['cut'] = le.fit_transform(df['cut'])
df['color'] = le.fit_transform(df['color'])
df['clarity'] = le.fit_transform(df['clarity'])

X = df[selected_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Linear Regression': LinearRegression()
}

best_model = None
best_score = float('inf')

for name, model in models.items():
    score = -cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf).mean()
    if score < best_score:
        best_score = score
        best_model = model
    print(f'{name}: {score}')

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print(f'Best Model: {best_model.__class__.__name__}')
print(f'Mean Squared Error on Test Set: {mean_squared_error(y_test, y_pred)}')

def predict_price(input_data):
    input_df = pd.DataFrame([input_data], columns=selected_features)
    return best_model.predict(input_df)[0]

st.title('Diamond Price Prediction')
st.write('Input the features to predict the price of a diamond.')

input_data = {feature: st.number_input(feature) for feature in selected_features}
if st.button('Predict'):
    price = predict_price(input_data)
    st.write(f'Predicted Diamond Price: ${price:.2f}')

