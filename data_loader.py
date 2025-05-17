import pandas as pd
import numpy as np
from datetime import datetime
import os


class DataLoader:
    def __init__(self, data_path='./data/'):
        self.data_path = data_path
        self.stores = None
        self.train = None
        self.test = None
        self.features = None
        self.merged_train = None
        self.merged_test = None

    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")

        # Load stores data
        stores_path = 'stores.csv'
        if os.path.exists(stores_path):
            self.stores = pd.read_csv(stores_path)
            print(f"Loaded stores data: {self.stores.shape}")
        else:
            print(f"Warning: {stores_path} not found")
            self.stores = self._generate_sample_stores()

        # Load training data
        train_path = 'train.csv'
        if os.path.exists(train_path):
            self.train = pd.read_csv(train_path)
            # Convert date to datetime
            self.train['Date'] = pd.to_datetime(self.train['Date'])
            print(f"Loaded training data: {self.train.shape}")
        else:
            print(f"Warning: {train_path} not found")
            self.train = self._generate_sample_train()

        # Load test data
        test_path = 'test.csv'
        if os.path.exists(test_path):
            self.test = pd.read_csv(test_path)
            # Convert date to datetime
            self.test['Date'] = pd.to_datetime(self.test['Date'])
            print(f"Loaded test data: {self.test.shape}")
        else:
            print(f"Warning: {test_path} not found")
            self.test = self._generate_sample_test()

        # Load features data
        features_path = 'features.csv'
        if os.path.exists(features_path):
            self.features = pd.read_csv(features_path)
            # Convert date to datetime
            self.features['Date'] = pd.to_datetime(self.features['Date'])
            print(f"Loaded features data: {self.features.shape}")
        else:
            print(f"Warning: {features_path} not found")
            self.features = self._generate_sample_features()

        return self.stores, self.train, self.test, self.features

    def merge_data(self):
        """Merge datasets for training and testing"""
        print("Merging datasets...")

        if self.train is None or self.stores is None or self.features is None:
            self.load_data()

        # Merge training data with stores and features
        self.merged_train = pd.merge(self.train, self.stores, on='Store', how='left')
        self.merged_train = pd.merge(self.merged_train, self.features, on=['Store', 'Date'], how='left')

        # Merge test data with stores and features
        self.merged_test = pd.merge(self.test, self.stores, on='Store', how='left')
        self.merged_test = pd.merge(self.merged_test, self.features, on=['Store', 'Date'], how='left')

        # Handle missing values
        self._handle_missing_values(self.merged_train)
        self._handle_missing_values(self.merged_test)

        print(f"Merged training data shape: {self.merged_train.shape}")
        print(f"Merged test data shape: {self.merged_test.shape}")

        return self.merged_train, self.merged_test

    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Fill missing markdown values with 0
        for col in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Fill other missing values with median or mode
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].median())

        # Convert boolean columns to integers
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)

    def engineer_features(self, df):
        """Engineer features for model training"""
        print("Engineering features...")

        # Create copy to avoid modifying original
        df_engineered = df.copy()

        # Extract date features
        df_engineered['Year'] = df_engineered['Date'].dt.year
        df_engineered['Month'] = df_engineered['Date'].dt.month
        df_engineered['Week'] = df_engineered['Date'].dt.isocalendar().week
        df_engineered['Day'] = df_engineered['Date'].dt.day
        df_engineered['DayOfWeek'] = df_engineered['Date'].dt.dayofweek
        df_engineered['Quarter'] = df_engineered['Date'].dt.quarter

        # Create total markdown feature
        markdown_cols = [col for col in df_engineered.columns if 'MarkDown' in col]
        if markdown_cols:
            df_engineered['TotalMarkDown'] = df_engineered[markdown_cols].sum(axis=1)

        # One-hot encode store type if it exists
        if 'Type' in df_engineered.columns:
            df_engineered = pd.get_dummies(df_engineered, columns=['Type'], prefix='Type')

        # Convert IsHoliday to numeric
        if 'IsHoliday' in df_engineered.columns:
            df_engineered['IsHoliday'] = df_engineered['IsHoliday'].astype(int)

        # Normalize Size if it exists
        if 'Size' in df_engineered.columns:
            df_engineered['Size'] = df_engineered['Size'] / df_engineered['Size'].max()

        # Ensure all numeric columns are float
        numeric_cols = df_engineered.select_dtypes(include=['int64', 'float64']).columns
        df_engineered[numeric_cols] = df_engineered[numeric_cols].astype('float32')

        return df_engineered

    def prepare_time_series(self, df, target_col='Weekly_Sales', sequence_length=10):
        """Prepare data for time series modeling (LSTM)"""
        print("Preparing time series data...")

        # Group by Store and Dept
        groups = df.groupby(['Store', 'Dept'])

        X, y = [], []

        for name, group in groups:
            # Sort by date
            group = group.sort_values('Date')

            # Get the target values
            sales = group[target_col].values if target_col in group.columns else None

            # Get the feature values (exclude target and identifiers)
            features = group.drop(['Store', 'Dept', 'Date'], axis=1)
            if target_col in features.columns:
                features = features.drop([target_col], axis=1)

            # Ensure all columns are numeric
            for col in features.columns:
                if features[col].dtype == 'object':
                    try:
                        features[col] = pd.to_numeric(features[col], errors='coerce')
                    except:
                        features[col] = features[col].astype('category').cat.codes

            # Convert all to float32 for TensorFlow
            features = features.astype('float32')

            feature_cols = features.columns
            features = features.values

            # Create sequences
            if sales is not None:  # For training data
                for i in range(len(group) - sequence_length):
                    X.append(features[i:i + sequence_length])
                    y.append(sales[i + sequence_length])
            else:  # For test data
                for i in range(len(group) - sequence_length):
                    X.append(features[i:i + sequence_length])
                    y.append(0)  # Placeholder

        # Convert to numpy arrays with explicit dtype
        X_array = np.array(X, dtype=np.float32)
        y_array = np.array(y, dtype=np.float32)

        # Check for NaN values
        if np.isnan(X_array).any():
            print("Warning: X contains NaN values. Replacing with 0.")
            X_array = np.nan_to_num(X_array)

        if np.isnan(y_array).any():
            print("Warning: y contains NaN values. Replacing with 0.")
            y_array = np.nan_to_num(y_array)

        return X_array, y_array, feature_cols

    def split_train_val(self, X, y, val_ratio=0.2):
        """Split data into training and validation sets"""
        # Determine split index
        split_idx = int(len(X) * (1 - val_ratio))

        # Split the data
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        return X_train, X_val, y_train, y_val

    def check_data_types(self, df, name="DataFrame"):
        """Check and report data types in the DataFrame"""
        print(f"\n--- Data Types in {name} ---")
        for col in df.columns:
            print(f"{col}: {df[col].dtype}")

            # Check for mixed types
            if df[col].dtype == 'object':
                unique_types = set(type(x) for x in df[col].dropna().values)
                print(f"  - Unique types: {unique_types}")

            # Check for NaN values
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"  - NaN count: {nan_count} ({nan_count / len(df) * 100:.2f}%)")

        print("-------------------------\n")

    def _generate_sample_stores(self):
        """Generate sample stores data for testing"""
        stores = []
        for i in range(1, 46):
            store_type = np.random.choice(['A', 'B', 'C'])
            size = np.random.randint(50000, 200000)
            stores.append({'Store': i, 'Type': store_type, 'Size': size})

        return pd.DataFrame(stores)

    def _generate_sample_train(self):
        """Generate sample training data for testing"""
        train_data = []
        start_date = datetime(2010, 2, 5)
        end_date = datetime(2012, 11, 1)

        # Generate weekly dates
        dates = pd.date_range(start=start_date, end=end_date, freq='W')

        for store in range(1, 6):
            for dept in range(1, 4):
                for date in dates:
                    is_holiday = np.random.choice([True, False], p=[0.1, 0.9])
                    weekly_sales = np.random.randint(5000, 50000)

                    train_data.append({
                        'Store': store,
                        'Dept': dept,
                        'Date': date,
                        'Weekly_Sales': weekly_sales,
                        'IsHoliday': is_holiday
                    })

        return pd.DataFrame(train_data)

    def _generate_sample_test(self):
        """Generate sample test data for testing"""
        test_data = []
        start_date = datetime(2012, 11, 2)
        end_date = datetime(2013, 7, 26)

        # Generate weekly dates
        dates = pd.date_range(start=start_date, end=end_date, freq='W')

        for store in range(1, 6):
            for dept in range(1, 4):
                for date in dates:
                    is_holiday = np.random.choice([True, False], p=[0.1, 0.9])

                    test_data.append({
                        'Store': store,
                        'Dept': dept,
                        'Date': date,
                        'IsHoliday': is_holiday
                    })

        return pd.DataFrame(test_data)

    def _generate_sample_features(self):
        """Generate sample features data for testing"""
        features_data = []
        start_date = datetime(2010, 2, 5)
        end_date = datetime(2013, 7, 26)

        # Generate weekly dates
        dates = pd.date_range(start=start_date, end=end_date, freq='W')

        for store in range(1, 6):
            for date in dates:
                is_holiday = np.random.choice([True, False], p=[0.1, 0.9])

                # Generate random features
                temperature = np.random.uniform(0, 100)
                fuel_price = np.random.uniform(2, 5)
                markdown1 = np.random.uniform(0, 1000) if np.random.random() > 0.3 else np.nan
                markdown2 = np.random.uniform(0, 1000) if np.random.random() > 0.3 else np.nan
                markdown3 = np.random.uniform(0, 1000) if np.random.random() > 0.3 else np.nan
                markdown4 = np.random.uniform(0, 1000) if np.random.random() > 0.3 else np.nan
                markdown5 = np.random.uniform(0, 1000) if np.random.random() > 0.3 else np.nan
                cpi = np.random.uniform(200, 250)
                unemployment = np.random.uniform(5, 10)

                features_data.append({
                    'Store': store,
                    'Date': date,
                    'Temperature': temperature,
                    'Fuel_Price': fuel_price,
                    'MarkDown1': markdown1,
                    'MarkDown2': markdown2,
                    'MarkDown3': markdown3,
                    'MarkDown4': markdown4,
                    'MarkDown5': markdown5,
                    'CPI': cpi,
                    'Unemployment': unemployment,
                    'IsHoliday': is_holiday
                })

        return pd.DataFrame(features_data)

    def normalize_features(self, train_df, test_df=None):
        """Normalize numerical features using min-max scaling"""
        print("Normalizing features...")

        # Identify numerical columns (excluding target and identifiers)
        exclude_cols = ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']
        num_cols = [col for col in train_df.columns if
                    train_df[col].dtype in ['int64', 'float64'] and
                    col not in exclude_cols]

        # Create copies to avoid modifying originals
        train_normalized = train_df.copy()
        test_normalized = test_df.copy() if test_df is not None else None

        # Normalize each numerical column
        for col in num_cols:
            # Get min and max from training data
            col_min = train_df[col].min()
            col_max = train_df[col].max()

            # Avoid division by zero
            if col_max > col_min:
                # Normalize training data
                train_normalized[col] = (train_df[col] - col_min) / (col_max - col_min)

                # Normalize test data if provided
                if test_normalized is not None:
                    test_normalized[col] = (test_df[col] - col_min) / (col_max - col_min)

        if test_normalized is not None:
            return train_normalized, test_normalized
        else:
            return train_normalized

    def create_holiday_features(self, df):
        """Create specific features for different holiday types"""
        print("Creating holiday features...")

        # Create copy to avoid modifying original
        df_with_holidays = df.copy()

        # Initialize holiday columns
        df_with_holidays['IsSuperBowl'] = 0
        df_with_holidays['IsLaborDay'] = 0
        df_with_holidays['IsThanksgiving'] = 0
        df_with_holidays['IsChristmas'] = 0

        # Define holiday dates for each year
        holidays = {
            # Super Bowl (first Sunday in February)
            'SuperBowl': [
                '2010-02-07', '2011-02-06', '2012-02-05', '2013-02-03'
            ],
            # Labor Day (first Monday in September)
            'LaborDay': [
                '2010-09-06', '2011-09-05', '2012-09-03', '2013-09-02'
            ],
            # Thanksgiving (fourth Thursday in November)
            'Thanksgiving': [
                '2010-11-25', '2011-11-24', '2012-11-22', '2013-11-28'
            ],
            # Christmas (December 25)
            'Christmas': [
                '2010-12-25', '2011-12-25', '2012-12-25', '2013-12-25'
            ]
        }

        # Convert holiday dates to datetime
        for holiday_type, dates in holidays.items():
            holidays[holiday_type] = [pd.to_datetime(date) for date in dates]

        # Function to check if a date is within a week of a holiday
        def is_holiday_week(date, holiday_dates):
            for holiday_date in holiday_dates:
                # Check if date is within 3 days before or after the holiday
                if abs((date - holiday_date).days) <= 3:
                    return 1
            return 0

        # Mark holiday weeks
        for idx, row in df_with_holidays.iterrows():
            date = row['Date']

            df_with_holidays.at[idx, 'IsSuperBowl'] = is_holiday_week(date, holidays['SuperBowl'])
            df_with_holidays.at[idx, 'IsLaborDay'] = is_holiday_week(date, holidays['LaborDay'])
            df_with_holidays.at[idx, 'IsThanksgiving'] = is_holiday_week(date, holidays['Thanksgiving'])
            df_with_holidays.at[idx, 'IsChristmas'] = is_holiday_week(date, holidays['Christmas'])

        return df_with_holidays

    def create_lag_features(self, df, target_col='Weekly_Sales', lag_periods=[1, 2, 4, 8, 12]):
        """Create lag features for time series analysis"""
        print("Creating lag features...")

        # Create copy to avoid modifying original
        df_with_lags = df.copy()

        # Group by Store and Dept
        groups = []
        for (store, dept), group in df_with_lags.groupby(['Store', 'Dept']):
            # Sort by date
            group = group.sort_values('Date')

            # Create lag features
            for lag in lag_periods:
                lag_col = f'{target_col}_Lag_{lag}'
                group[lag_col] = group[target_col].shift(lag)

            # Add to list of processed groups
            groups.append(group)

        # Combine all groups back into a single DataFrame
        df_with_lags = pd.concat(groups)

        # Fill NaN values in lag columns with 0
        lag_cols = [f'{target_col}_Lag_{lag}' for lag in lag_periods]
        df_with_lags[lag_cols] = df_with_lags[lag_cols].fillna(0)

        return df_with_lags

    def create_rolling_features(self, df, target_col='Weekly_Sales', windows=[4, 8, 12]):
        """Create rolling window features (moving averages, std, etc.)"""
        print("Creating rolling window features...")

        # Create copy to avoid modifying original
        df_with_rolling = df.copy()

        # Group by Store and Dept
        groups = []
        for (store, dept), group in df_with_rolling.groupby(['Store', 'Dept']):
            # Sort by date
            group = group.sort_values('Date')

            # Create rolling features
            for window in windows:
                # Moving average
                group[f'{target_col}_MA_{window}'] = group[target_col].rolling(window=window).mean()

                # Moving standard deviation
                group[f'{target_col}_STD_{window}'] = group[target_col].rolling(window=window).std()

                # Moving min and max
                group[f'{target_col}_MIN_{window}'] = group[target_col].rolling(window=window).min()
                group[f'{target_col}_MAX_{window}'] = group[target_col].rolling(window=window).max()

            # Add to list of processed groups
            groups.append(group)

        # Combine all groups back into a single DataFrame
        df_with_rolling = pd.concat(groups)

        # Fill NaN values in rolling columns with 0
        rolling_cols = []
        for window in windows:
            rolling_cols.extend([
                f'{target_col}_MA_{window}',
                f'{target_col}_STD_{window}',
                f'{target_col}_MIN_{window}',
                f'{target_col}_MAX_{window}'
            ])

        df_with_rolling[rolling_cols] = df_with_rolling[rolling_cols].fillna(0)

        return df_with_rolling