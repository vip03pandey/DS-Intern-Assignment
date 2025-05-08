import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

def load_data(file_path='/Users/vipulpandey/DS-Intern-Assignment/data/data.csv'):
    try:
        print("Loading data...")
        df = pd.read_csv(file_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        print("Dataset Shape:", df.shape)
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

def preprocess_data(df):
    if df is None:
        return None
    df.rename(columns=lambda x: x.strip(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col].str.strip(), errors="coerce")
    df.drop(columns=['visibility_index', 'random_variable1', 'random_variable2'], inplace=True)
    df = df[df['equipment_energy_consumption'] >= 0]
    df = df[df['lighting_energy'] >= 0]
    return df

def perform_eda(df):
    if df is None:
        return
    print("\nPerforming Time Series Analysis...")
    print("Dataset Info:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    nan_by_date = df.isna().sum(axis=1)
    nan_by_date = nan_by_date.rename('NA_COUNT').reset_index()
    perc_nan_by_date = nan_by_date.copy()
    perc_nan_by_date['NA_COUNT'] = 100 * perc_nan_by_date['NA_COUNT'] / 25
    fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
    sns.lineplot(data=nan_by_date, x='timestamp', y='NA_COUNT', ax=ax[0])
    ax[0].set_title('Missing Values Count Over Time')
    sns.lineplot(data=perc_nan_by_date[-8700:], x='timestamp', y='NA_COUNT', ax=ax[1])
    ax[1].set_title('% Device Down (Last 8700 Records)')
    ax[1].set_ylabel('% Device Down')
    plt.tight_layout()
    plt.savefig('missing_values_plot.png')
    plt.close()
    zone_temp_cols = [f'zone{i}_temperature' for i in range(1, 10)]
    plt.figure(figsize=(12, 6))
    df[zone_temp_cols].boxplot()
    plt.title('Temperature Distribution Across Zones')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('zone_temperature_boxplot.png')
    plt.close()
    plt.figure(figsize=(12, 6))
    df['lighting_energy'].plot(kind='hist', bins=20, alpha=0.5, color='blue', label='Lighting Energy')
    plt.title('Lighting Energy Distribution')
    plt.xlabel('Lighting Energy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lighting_energy_histogram.png')
    plt.close()

def engineer_features(df, target_col='equipment_energy_consumption'):
    if df is None:
        return None
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    for span in [3, 6, 12, 24]:
        df[f'{target_col}_ewm_{span}'] = df[target_col].ewm(span=span).mean()
    df.fillna(df.median(), inplace=True)
    return df

def perform_cross_validation(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    rmse_scores, mae_scores, r2_scores, train_r2_scores = [], [], [], []
    print("\nPerforming Time Series Cross-Validation with RandomForestRegressor...")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        train_r2_scores.append(train_r2)
        print(f"\nFold {fold} Results:")
        print(f"Training R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {r2:.4f}")
        print(f"R² Score Difference (Train - Test): {train_r2 - r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    print("\nAverage Metrics Across Folds:")
    print(f"Average Training R²: {np.mean(train_r2_scores):.4f} ± {np.std(train_r2_scores):.4f}")
    print(f"Average Test R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print("\nAverage Model Performance Metrics:")
    print(f"Average RMSE: {np.mean(rmse_scores):.2f} (±{np.std(rmse_scores):.2f})")
    print(f"Average MAE: {np.mean(mae_scores):.2f} (±{np.std(mae_scores):.2f})")
    print(f"Average Test R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")
    print(f"Average Train R²: {np.mean(train_r2_scores):.4f} (±{np.std(train_r2_scores):.4f})")
    print(f"Average R² Difference (Train - Test): {np.mean(train_r2_scores) - np.mean(r2_scores):.4f}")
    return scaler

def train_final_model(X, y, scaler):
    print("\nTraining final model on all data...")
    X_scaled = scaler.fit_transform(X)
    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_scaled, y)
    joblib.dump(final_model, 'final_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Final model saved as 'final_model.pkl'")
    print("Scaler saved as 'scaler.pkl'")
    feature_importance = pd.Series(final_model.feature_importances_, index=X.columns)
    feature_importance.sort_values(ascending=False).head(10).plot(kind='bar', figsize=(12, 6))
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    return final_model

def main():
    df = load_data()
    df = preprocess_data(df)
    perform_eda(df)
    target_col = 'equipment_energy_consumption'
    df = engineer_features(df, target_col)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    scaler = perform_cross_validation(X, y)
    train_final_model(X, y, scaler)

if __name__ == "__main__":
    main()