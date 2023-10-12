import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression # Importing Linear Regression Model
from sklearn.model_selection import train_test_split # Importing Model Selection to split dataset into training and testing set
from sklearn.metrics import r2_score
import joblib
from sklearn.metrics import mean_squared_error
df =pd.read_csv("datasets/housing.csv")
def inspect_data(df):
    print('Data Shape')
    print('\n')
    print(df.shape)
    print('\n')
    print('Missing Values: ')
    print(df.isnull().sum())
    print('\n')
    print('Data Types: ')
    print(df.dtypes)
    
# inspect_data(df)
df.drop('ocean_proximity', axis = 1, inplace = True)
df.dropna(inplace = True)
X = df.drop('median_house_value', axis = 1)
# X.head()
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3) 
lr_model = LinearRegression() # Initializing the model
trained_model= lr_model.fit(X_train, y_train)
joblib.dump(trained_model, "trained_model.pkl")
deserialized = joblib.load('trained_model.pkl')
input_data = np.array([[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]]) 

# Make predictions
print("Trained model output:", trained_model.predict(input_data))
print("Deserialized model output:", deserialized.predict(input_data))
y_pred = lr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = lr_model.score(X_test, y_test)

print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)