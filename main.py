import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
import argparse
import logging

# Import custom modules
from data_loader import DataLoader
from lstm_model import LSTMModel
from modern_charts import ModernCharts
from modern_dashboard import ModernDashboard

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("walmart_sales_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_lstm_model(data_loader, sequence_length=10, epochs=50):
    """Train LSTM model for time series prediction"""
    logger.info("Starting LSTM model training...")

    # Load and preprocess data
    _, train_data, _, _ = data_loader.load_data()
    merged_train, _ = data_loader.merge_data()

    # Check data types before feature engineering
    data_loader.check_data_types(merged_train, "Merged Training Data")

    # Engineer features
    engineered_data = data_loader.engineer_features(merged_train)

    # Check data types after feature engineering
    data_loader.check_data_types(engineered_data, "Engineered Data")

    # Prepare time series data
    X, y, feature_cols = data_loader.prepare_time_series(
        engineered_data, target_col='Weekly_Sales', sequence_length=sequence_length
    )

    logger.info(f"Prepared time series data with {len(X)} sequences")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    # Split data
    X_train, X_val, y_train, y_val = data_loader.split_train_val(X, y, val_ratio=0.2)

    # Initialize and train model
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_model = LSTMModel(input_shape=input_shape)

    # Build and train model
    lstm_model.build_model()
    history = lstm_model.train(X_train, y_train, X_val, y_val, epochs=epochs)

    # Evaluate model
    metrics = lstm_model.evaluate(X_val, y_val)
    logger.info(f"Model evaluation metrics: {metrics}")

    # Plot training history
    lstm_model.plot_history()

    # Save model
    lstm_model.save_model()

    return lstm_model


def train_random_forest(data_loader):
    """Train Random Forest model for feature importance analysis"""
    logger.info("Starting Random Forest model training...")

    # Load and preprocess data
    _, train_data, _, _ = data_loader.load_data()
    merged_train, _ = data_loader.merge_data()

    # Engineer features
    engineered_data = data_loader.engineer_features(merged_train)

    # Prepare features and target
    features = engineered_data.drop(['Store', 'Dept', 'Date', 'Weekly_Sales'], axis=1)
    target = engineered_data['Weekly_Sales']

    # Handle any remaining NaN values
    features = features.fillna(0)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(features, target)

    logger.info(f"Random Forest model trained with {len(features.columns)} features")

    # Create feature importance plot
    charts = ModernCharts()
    charts.plot_feature_importance(rf_model, features.columns, title='Feature Importance (Random Forest)')

    return rf_model, features.columns


def generate_visualizations(data_loader):
    """Generate various visualizations for analysis"""
    logger.info("Generating visualizations...")

    # Load and preprocess data
    stores, train, _, features = data_loader.load_data()
    merged_train, _ = data_loader.merge_data()

    # Create charts object
    charts = ModernCharts()

    # Generate various plots
    charts.plot_sales_by_store(merged_train)
    charts.plot_sales_by_dept(merged_train)
    charts.plot_sales_trend(merged_train)
    charts.plot_correlation_heatmap(merged_train)

    if 'Type' in merged_train.columns:
        charts.plot_sales_by_store_type(merged_train)

    if 'IsHoliday' in merged_train.columns:
        charts.plot_sales_by_holiday(merged_train)

    # Generate PCA analysis
    charts.plot_pca_analysis(merged_train)

    logger.info("Visualizations generated successfully")


def run_dashboard(data_loader, model=None):
    """Run the interactive dashboard"""
    logger.info("Starting dashboard...")

    # Initialize dashboard
    dashboard = ModernDashboard(data_loader, model)

    # Run dashboard server
    dashboard.run_server(debug=True)

    logger.info("Dashboard is running")


def make_predictions(data_loader, model):
    """Generate predictions for test data"""
    logger.info("Generating predictions...")

    # Load and preprocess data
    _, _, test_data, _ = data_loader.load_data()
    _, merged_test = data_loader.merge_data()

    # Engineer features
    engineered_test = data_loader.engineer_features(merged_test)

    # Prepare time series data for prediction
    X_test, _, feature_cols = data_loader.prepare_time_series(
        engineered_test, target_col=None, sequence_length=10
    )

    # Make predictions
    predictions = model.predict(X_test)

    # Create DataFrame with predictions
    pred_df = pd.DataFrame({
        'Store': engineered_test['Store'].iloc[10:].values,
        'Dept': engineered_test['Dept'].iloc[10:].values,
        'Date': engineered_test['Date'].iloc[10:].values,
        'Predicted_Sales': predictions.flatten()
    })

    # Save predictions
    os.makedirs('output', exist_ok=True)
    pred_df.to_csv('output/predictions.csv', index=False)

    logger.info(f"Predictions generated and saved for {len(pred_df)} records")

    return pred_df


def main():
    """Main function to run the Walmart Sales Prediction project"""
    parser = argparse.ArgumentParser(description='Walmart Sales Prediction')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'visualize', 'predict', 'dashboard'],
                        help='Mode to run the application')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='Path to the data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for LSTM training')

    args = parser.parse_args()

    try:
        # Initialize data loader
        data_loader = DataLoader(data_path=args.data_path)

        if args.mode == 'train':
            # Train LSTM model
            lstm_model = train_lstm_model(data_loader, epochs=args.epochs)

            # Train Random Forest for feature importance
            rf_model, feature_names = train_random_forest(data_loader)

            logger.info("Training completed successfully")

        elif args.mode == 'visualize':
            # Generate visualizations
            generate_visualizations(data_loader)

        elif args.mode == 'predict':
            # Load trained model
            input_shape = (10, 20)  # This should match your actual data dimensions
            lstm_model = LSTMModel(input_shape=input_shape)
            lstm_model.load_model()

            # Make predictions
            predictions = make_predictions(data_loader, lstm_model)

            logger.info("Prediction completed successfully")

        elif args.mode == 'dashboard':
            # Load trained model if available
            try:
                input_shape = (10, 20)  # This should match your actual data dimensions
                lstm_model = LSTMModel(input_shape=input_shape)
                lstm_model.load_model()
                logger.info("Loaded trained model for dashboard")
            except:
                lstm_model = None
                logger.warning("No trained model found, running dashboard without prediction capability")

            # Run dashboard
            run_dashboard(data_loader, lstm_model)

        logger.info("Program completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        print(f"\nERROR: {str(e)}")
        print("\nFor more details, check the log file.")


if __name__ == "__main__":
    main()
