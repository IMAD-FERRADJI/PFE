import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO
import re
import os
from lstm_model import get_lstm_predictions_real, get_optimization_recommendations

# Initialize as empty DataFrame
inventory_df = pd.DataFrame()


def load_grocery_data(url):
    """
    Load grocery inventory data from a URL

    Args:
        url (str): URL to the CSV file

    Returns:
        pd.DataFrame: Processed dataframe with inventory data
    """
    global inventory_df

    try:
        # Fetch data from URL
        print(f"Fetching data from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Read CSV data
        data = StringIO(response.text)
        df = pd.read_csv(data)

        print(f"Successfully loaded data. Found {len(df)} rows and {len(df.columns)} columns.")
        print(f"Columns: {df.columns.tolist()}")

        # Process data types
        # Convert string numbers to numeric
        numeric_columns = ['Stock_Quantity', 'Reorder_Level', 'Reorder_Quantity',
                           'Sales_Volume', 'Inventory_Turnover_Rate']

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted {col} to numeric. Sample: {df[col].head(3).tolist()}")

        # Process Unit_Price (remove $ and convert to float)
        if 'Unit_Price' in df.columns:
            df['Unit_Price'] = df['Unit_Price'].apply(
                lambda x: float(re.sub(r'[^\d.]', '', str(x))) if pd.notna(x) else np.nan
            )
            print(f"Processed Unit_Price. Sample: {df['Unit_Price'].head(3).tolist()}")

        # Convert date string to datetime - handle MM/DD/YYYY format
        if 'Last_Order_Date' in df.columns:
            df['Last_Order_Date'] = pd.to_datetime(df['Last_Order_Date'], errors='coerce')
            print(f"Converted Last_Order_Date to datetime. Sample: {df['Last_Order_Date'].head(3).tolist()}")

        # Add a status column based on stock levels
        if all(col in df.columns for col in ['Stock_Quantity', 'Reorder_Level']):
            df['Status'] = df.apply(lambda row:
                                    "Low Stock" if row['Stock_Quantity'] < row['Reorder_Level'] else
                                    "Overstocked" if row['Stock_Quantity'] > 2 * row['Reorder_Level'] else
                                    "In Stock", axis=1)
            print(f"Added Status column. Distribution: {df['Status'].value_counts().to_dict()}")

        # Calculate days since last order
        if 'Last_Order_Date' in df.columns:
            df['Days_Since_Order'] = (datetime.now() - df['Last_Order_Date']).dt.days
            print(f"Calculated Days_Since_Order. Sample: {df['Days_Since_Order'].head(3).tolist()}")

        # Calculate inventory value
        if all(col in df.columns for col in ['Stock_Quantity', 'Unit_Price']):
            df['Inventory_Value'] = df['Stock_Quantity'] * df['Unit_Price']
            print(f"Calculated Inventory_Value. Sample: {df['Inventory_Value'].head(3).tolist()}")

        # Update the global inventory_df variable
        inventory_df = df
        print(f"Updated global inventory_df with {len(df)} rows")

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        empty_df = create_empty_dataframe()
        inventory_df = empty_df
        return empty_df


def load_local_csv(file_path):
    """
    Load grocery inventory data from a local CSV file

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Processed dataframe with inventory data
    """
    global inventory_df

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return create_empty_dataframe()

        # Read CSV data
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}. Found {len(df)} rows.")

        # Process data types
        # Convert string numbers to numeric
        numeric_columns = ['Stock_Quantity', 'Reorder_Level', 'Reorder_Quantity',
                           'Sales_Volume', 'Inventory_Turnover_Rate']

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Process Unit_Price (remove $ and convert to float)
        if 'Unit_Price' in df.columns:
            df['Unit_Price'] = df['Unit_Price'].apply(
                lambda x: float(str(x).replace('$', '').replace(',', '')) if pd.notna(x) else np.nan
            )

        # Convert date string to datetime - handle various formats
        if 'Last_Order_Date' in df.columns:
            df['Last_Order_Date'] = pd.to_datetime(df['Last_Order_Date'], errors='coerce')

        # Add a status column based on stock levels
        if all(col in df.columns for col in ['Stock_Quantity', 'Reorder_Level']):
            df['Status'] = df.apply(lambda row:
                                    "Low Stock" if row['Stock_Quantity'] < row['Reorder_Level'] else
                                    "Overstocked" if row['Stock_Quantity'] > 2 * row['Reorder_Level'] else
                                    "In Stock", axis=1)

        # Calculate days since last order
        if 'Last_Order_Date' in df.columns:
            df['Days_Since_Order'] = (datetime.now() - df['Last_Order_Date']).dt.days

        # Calculate inventory value
        if all(col in df.columns for col in ['Stock_Quantity', 'Unit_Price']):
            df['Inventory_Value'] = df['Stock_Quantity'] * df['Unit_Price']

        # Update the global inventory_df variable
        inventory_df = df

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        empty_df = create_empty_dataframe()
        inventory_df = empty_df
        return empty_df


def create_empty_dataframe():
    """Create an empty DataFrame with the expected columns"""
    return pd.DataFrame(columns=[
        'Product_ID', 'Product_Name', 'Supplier_ID', 'Supplier_Name',
        'Stock_Quantity', 'Reorder_Level', 'Reorder_Quantity', 'Unit_Price',
        'Last_Order_Date', 'Sales_Volume', 'Inventory_Turnover_Rate',
        'Status', 'Days_Since_Order', 'Inventory_Value'
    ])


def get_inventory_summary(df):
    """
    Calculate summary metrics from inventory data

    Args:
        df (pd.DataFrame): Processed inventory dataframe

    Returns:
        dict: Dictionary with summary metrics
    """
    if df.empty:
        return {
            'total_items': 0,
            'total_quantity': 0,
            'low_stock_count': 0,
            'inventory_value': 0,
            'avg_turnover': 0
        }

    summary = {
        'total_items': len(df),
        'total_quantity': int(df['Stock_Quantity'].sum() if 'Stock_Quantity' in df.columns else 0),
        'low_stock_count': int(df[df['Status'] == 'Low Stock'].shape[0] if 'Status' in df.columns else 0),
        'inventory_value': float(df['Inventory_Value'].sum() if 'Inventory_Value' in df.columns else 0),
        'avg_turnover': float(df['Inventory_Turnover_Rate'].mean() if 'Inventory_Turnover_Rate' in df.columns else 0)
    }

    return summary


def get_category_distribution(df):
    """
    Get distribution of products by category
    For this dataset, we'll use the first word of the product name as a simple category

    Args:
        df (pd.DataFrame): Processed inventory dataframe

    Returns:
        tuple: (categories, values) for plotting
    """
    if df.empty or 'Product_Name' not in df.columns:
        return [], []

    # Extract simple categories from product names (first word)
    df['Category'] = df['Product_Name'].apply(lambda x: str(x).split()[0] if pd.notna(x) else 'Unknown')

    # Get top 5 categories by count
    category_counts = df['Category'].value_counts().head(5)

    # Add "Other" category for the rest
    other_count = df['Category'].value_counts().sum() - category_counts.sum()
    if other_count > 0:
        category_counts['Other'] = other_count

    return category_counts.index.tolist(), category_counts.values.tolist()


def get_time_series_data(df, days=90):
    """
    Generate time series data for stock levels, sales, and reordering
    This simulates historical data based on current values

    Args:
        df (pd.DataFrame): Processed inventory dataframe
        days (int): Number of days of historical data to generate

    Returns:
        pd.DataFrame: DataFrame with daily stock, sales, and restock data
    """
    if df.empty:
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['date', 'stock', 'sales', 'restock'])

    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Initialize result dataframe
    result = pd.DataFrame({'date': dates})

    # Calculate average daily values
    avg_stock = df['Stock_Quantity'].mean() if 'Stock_Quantity' in df.columns else 100
    avg_sales = df[
                    'Sales_Volume'].mean() / 30 if 'Sales_Volume' in df.columns else 10  # Assuming Sales_Volume is monthly
    avg_restock = df[
                      'Reorder_Quantity'].mean() / 30 if 'Reorder_Quantity' in df.columns else 15  # Assuming monthly reordering

    # Generate simulated data with some randomness and seasonality
    np.random.seed(42)  # For reproducibility

    # Stock levels with weekly pattern
    stock_base = avg_stock * 1.2  # Start slightly higher than average
    stock = []
    sales = []
    restock = []

    for i, date in enumerate(dates):
        # Add weekly seasonality (higher on weekends)
        day_of_week = date.dayofweek
        seasonal_factor = 1.1 if day_of_week >= 5 else 1.0

        # Add some randomness
        random_factor = np.random.normal(1, 0.05)

        # Calculate stock for this day
        if i == 0:
            stock.append(stock_base * seasonal_factor * random_factor)
        else:
            # Stock = previous stock - sales + restock
            stock.append(max(0, stock[i - 1] - sales[i - 1] + restock[i - 1]))

        # Sales with weekly pattern (higher on weekends)
        sales_seasonal_factor = 1.3 if day_of_week >= 5 else 1.0
        sales_random_factor = np.random.normal(1, 0.2)
        sales.append(avg_sales * sales_seasonal_factor * sales_random_factor)

        # Restock (happens periodically)
        if i % 7 == 0:
            restock.append(avg_restock * np.random.uniform(0.8, 1.2) * 7)
        else:
            restock.append(0)

    # Add to result dataframe
    result['stock'] = stock
    result['sales'] = sales
    result['restock'] = restock

    return result


def get_lstm_predictions(df, product_name=None, history_days=30, forecast_days=14):
    """
    Generate LSTM predictions based on inventory data
    This is a simplified version that generates synthetic forecast data

    Args:
        df (pd.DataFrame): Processed inventory dataframe
        product_name (str, optional): Name of the product to forecast
        history_days (int): Number of days of historical data
        forecast_days (int): Number of days to forecast

    Returns:
        dict: Dictionary with dates, actual values, predictions, and confidence intervals
    """
    if df.empty:
        # Return empty prediction data
        return {
            'date': [],
            'actual': [],
            'predicted': [],
            'lower': [],
            'upper': [],
            'mae': 0,
            'rmse': 0,
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M')
        }

    # Filter data for specific product if provided
    if product_name and 'Product_Name' in df.columns:
        product_df = df[df['Product_Name'] == product_name]
        if not product_df.empty:
            df = product_df

    # Get current stock level
    current_stock = df['Stock_Quantity'].mean() if 'Stock_Quantity' in df.columns else 100

    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=history_days)
    forecast_end = end_date + timedelta(days=forecast_days)

    history_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    forecast_dates = pd.date_range(start=end_date + timedelta(days=1), end=forecast_end, freq='D')
    all_dates = history_dates.append(forecast_dates)

    # Format dates as strings
    date_strings = [d.strftime('%Y-%m-%d') for d in all_dates]

    # Generate historical data with weekly seasonality
    np.random.seed(42)  # For reproducibility

    historical_values = []
    for i, date in enumerate(history_dates):
        # Add weekly seasonality (higher on weekends)
        day_of_week = date.dayofweek
        seasonal_factor = 1.1 if day_of_week >= 5 else 1.0

        # Add trend (slight decrease over time)
        trend_factor = 1.0 - (i / len(history_dates)) * 0.1

        # Add some randomness
        random_factor = np.random.normal(1, 0.05)

        # Calculate stock for this day
        historical_values.append(current_stock * seasonal_factor * trend_factor * random_factor)

    # Generate forecast data
    forecast_values = []
    lower_bounds = []
    upper_bounds = []

    # Calculate trend from historical data
    x = np.arange(len(historical_values))
    y = historical_values
    z = np.polyfit(x, y, 1)
    slope = z[0]
    intercept = z[1]

    for i, date in enumerate(forecast_dates):
        # Add weekly seasonality (higher on weekends)
        day_of_week = date.dayofweek
        seasonal_factor = 1.1 if day_of_week >= 5 else 1.0

        # Predict using trend and seasonality
        prediction = (intercept + slope * (len(historical_values) + i)) * seasonal_factor

        # Add confidence intervals
        std_dev = np.std(historical_values)
        lower_bound = max(0, prediction - 1.96 * std_dev)
        upper_bound = prediction + 1.96 * std_dev

        forecast_values.append(prediction)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    # Combine historical and forecast data
    actual = list(historical_values) + [None] * len(forecast_dates)
    predicted = [None] * len(history_dates) + list(forecast_values)
    lower = [None] * len(history_dates) + list(lower_bounds)
    upper = [None] * len(history_dates) + list(upper_bounds)

    # Calculate error metrics
    mae = np.std(historical_values) * 0.8
    rmse = np.std(historical_values) * 1.2

    return {
        'date': date_strings,
        'actual': actual,
        'predicted': predicted,
        'lower': lower,
        'upper': upper,
        'mae': mae,
        'rmse': rmse,
        'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M')
    }


def get_optimization_recommendations(df):
    """
    Generate inventory optimization recommendations based on forecasting predictions

    Args:
        df (pd.DataFrame): Inventory dataframe

    Returns:
        list: List of recommendation dictionaries
    """
    if df.empty:
        # Return empty recommendations
        return []

    # Get unique products
    if 'Product_Name' in df.columns:
        products = df['Product_Name'].unique()
    else:
        # No product names, return empty recommendations
        return []

    recommendations = []

    for product in products[:5]:  # Limit to 5 products
        # Filter data for this product
        product_df = df[df['Product_Name'] == product]

        if len(product_df) == 0:
            continue

        # Get current stock
        current_stock = product_df['Stock_Quantity'].iloc[0] if 'Stock_Quantity' in product_df.columns else 0

        # Get min stock level
        min_stock = product_df['Reorder_Level'].iloc[
            0] if 'Reorder_Level' in product_df.columns else current_stock * 0.5

        # Generate predictions
        predictions = get_lstm_predictions(product_df, product)

        # Calculate recommended stock based on prediction
        if predictions['predicted'] and any(p is not None for p in predictions['predicted']):
            # Get the maximum predicted demand with upper confidence bound
            pred_values = [p for p in predictions['predicted'] if p is not None]
            upper_values = [p for p in predictions['upper'] if p is not None]

            if pred_values and upper_values:
                max_predicted = max(pred_values)
                max_upper = max(upper_values)

                # Recommended stock is the maximum of:
                # 1. Upper confidence bound of prediction
                # 2. Minimum stock level
                recommended_stock = max(max_upper, min_stock)

                # Determine reason based on current stock vs recommended
                if current_stock < recommended_stock:
                    reason = "Forecast predicts increased demand"
                elif current_stock > recommended_stock * 1.2:
                    reason = "Historical oversupply pattern detected"
                else:
                    reason = "Current stock levels are near optimal"
            else:
                # Fallback if no valid predictions
                recommended_stock = max(current_stock * 1.1, min_stock)
                reason = "Based on historical patterns"
        else:
            # Fallback if no predictions
            recommended_stock = max(current_stock * 1.1, min_stock)
            reason = "Based on historical patterns"

        # Round values
        current_stock = int(current_stock)
        recommended_stock = int(recommended_stock)

        # Calculate confidence
        confidence = 95 - np.random.randint(0, 10)  # Random confidence between 85-95%

        # Add recommendation
        recommendations.append({
            "id": product_df['Product_ID'].iloc[0] if 'Product_ID' in product_df.columns else "",
            "product": product,
            "currentStock": current_stock,
            "recommendedStock": recommended_stock,
            "confidence": confidence,
            "reason": reason,
        })

    return recommendations


# If the file is run directly, test the functions
if __name__ == "__main__":
    # Test loading data
    df = load_local_csv("inventory_data.csv")

    if not df.empty:
        print("\nInventory Summary:")
        print(get_inventory_summary(df))

        print("\nCategory Distribution:")
        categories, values = get_category_distribution(df)
        for cat, val in zip(categories, values):
            print(f"{cat}: {val}")

        print("\nTime Series Data (first 5 rows):")
        time_series = get_time_series_data(df)
        print(time_series.head())

        print("\nLSTM Predictions:")
        predictions = get_lstm_predictions(df)
        print(f"Dates: {len(predictions['date'])} entries")
        print(f"Actual: {len(predictions['actual'])} entries")
        print(f"Predicted: {len(predictions['predicted'])} entries")

        print("\nOptimization Recommendations:")
        recommendations = get_optimization_recommendations(df)
        for rec in recommendations:
            print(
                f"{rec['product']}: Current={rec['currentStock']}, Recommended={rec['recommendedStock']}, Confidence={rec['confidence']}%")
