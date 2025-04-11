from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('train.csv')

# Combine Date and Time into 'createdAt'
df['createdAt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
 
# Add sensor info
df['sensorId'] = 'sensor-1'
df['sensor'] = df['sensorId'].apply(lambda x: {
    'id': x,
    'name': f"Sensor {x.split('-')[-1]}",
    'areaId': 'Area-A',
    'location': 'Room 1'
})

# Rename columns
df.rename(columns={
    'Indoor_temperature_room': 'temp',
    'Relative_humidity_room': 'humidity'
}, inplace=True)

# Format final DataFrame
df_formatted = df[['sensorId', 'temp', 'humidity', 'createdAt', 'sensor']]

# Enhanced data preprocessing
def process_temp_enhanced(df):
    df = df.copy()
    
    # Handle missing values more effectively
    df.dropna(inplace=True)
    
    # Extract sensor information
    df.drop(columns=['sensor'], inplace=True)
    
    # Improved temporal feature extraction
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    df['hour'] = df['createdAt'].dt.hour
    df['minute'] = df['createdAt'].dt.minute
    df['day'] = df['createdAt'].dt.day
    df['weekday'] = df['createdAt'].dt.weekday
    df['month'] = df['createdAt'].dt.month
    
    # Create cyclical features for time variables
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/7)
    
    # Add lagged humidity as it might help predict temperature
    df['humidity_lag'] = df['humidity'].shift(1).fillna(df['humidity'].mean())
    
    # Add temperature and humidity interaction
    df['temp_humidity_interaction'] = df['temp'] * df['humidity']
    
    # Add hour-specific features
    df['is_daytime'] = ((df['hour'] >= 7) & (df['hour'] <= 19)).astype(int)
    
    # Add day of year to capture seasonality
    df['day_of_year'] = df['createdAt'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    # Drop original datetime column
    df.drop(columns=['createdAt'], inplace=True)
    
    # Encode categorical variables
    encoder = LabelEncoder()
    df['sensorId'] = encoder.fit_transform(df['sensorId'])
    
    # Don't drop humidity as it's valuable for temperature prediction
    
    # Normalize numerical features
    scaler = StandardScaler()
    numeric_cols = ['temp', 'day', 'hour', 'minute', 'weekday', 'month', 
                    'humidity', 'humidity_lag', 'temp_humidity_interaction', 'day_of_year']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def split_with_validation(df, target, test_size=0.2, val_size=0.25):
    """Split data into train, validation and test sets"""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df.drop(columns=[target]), df[target], test_size=test_size, random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def find_best_model(X_train, X_val, y_train, y_val):
    """Compare different models to find the best one"""
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(),
        'MLP': MLPRegressor(random_state=42, max_iter=500)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        results[name] = val_r2
        print(f"{name}: Validation R^2 = {val_r2:.4f}")
    
    best_model_name = max(results, key=results.get)
    print(f"\nBest model: {best_model_name} with R^2 = {results[best_model_name]:.4f}")
    
    return models[best_model_name]


# Enhanced pipeline
df_processed = process_temp_enhanced(df_formatted)
X_train, X_val, X_test, y_train, y_val, y_test = split_with_validation(df_processed, 'temp')

# First find the best model type
best_model = find_best_model(X_train, X_val, y_train, y_val)

# Train final model on combined training and validation data
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])
best_model.fit(X_train_full, y_train_full)

# Predict on test set
y_pred = best_model.predict(X_test)

# Calculate final accuracy
final_r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nFinal Test Set R^2 Score: {final_r2:.4f} ({final_r2 * 100:.2f}%)")
print(f"Final Test Set RMSE: {rmse:.4f}")





