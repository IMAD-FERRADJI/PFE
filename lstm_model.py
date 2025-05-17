import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os


class LSTMModel:
    def __init__(self, input_shape, output_units=1, lstm_units=50, dropout_rate=0.2):
        """
        Initialize the LSTM model.

        Args:
            input_shape: Tuple of (timesteps, features) for input data
            output_units: Number of output units
            lstm_units: Number of LSTM units in each layer
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.output_units = output_units
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None

    def build_model(self):
        """Build LSTM model architecture"""
        model = Sequential()

        # First LSTM layer with return sequences for stacking
        model.add(LSTM(units=self.lstm_units,
                       return_sequences=True,
                       input_shape=self.input_shape))
        model.add(Dropout(self.dropout_rate))

        # Second LSTM layer
        model.add(LSTM(units=self.lstm_units))
        model.add(Dropout(self.dropout_rate))

        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=self.output_units))

        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')

        self.model = model
        print(model.summary())
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=10):
        """Train the LSTM model"""
        if self.model is None:
            self.build_model()

        # Ensure data types are correct
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        # Check for NaN values
        if np.isnan(X_train).any():
            print("Warning: X_train contains NaN values. Replacing with 0.")
            X_train = np.nan_to_num(X_train)

        if np.isnan(y_train).any():
            print("Warning: y_train contains NaN values. Replacing with 0.")
            y_train = np.nan_to_num(y_train)

        if np.isnan(X_val).any():
            print("Warning: X_val contains NaN values. Replacing with 0.")
            X_val = np.nan_to_num(X_val)

        if np.isnan(y_val).any():
            print("Warning: y_val contains NaN values. Replacing with 0.")
            y_val = np.nan_to_num(y_val)

        # Print shapes for debugging
        print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        model_checkpoint = ModelCheckpoint(
            filepath='models/lstm_walmart_sales.h5',
            monitor='val_loss',
            save_best_only=True
        )

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        return self.history

    def predict(self, X):
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")

        # Ensure data type is correct
        X = np.array(X, dtype=np.float32)

        # Check for NaN values
        if np.isnan(X).any():
            print("Warning: X contains NaN values. Replacing with 0.")
            X = np.nan_to_num(X)

        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")

        # Ensure data types are correct
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)

        # Check for NaN values
        if np.isnan(X_test).any():
            print("Warning: X_test contains NaN values. Replacing with 0.")
            X_test = np.nan_to_num(X_test)

        if np.isnan(y_test).any():
            print("Warning: y_test contains NaN values. Replacing with 0.")
            y_test = np.nan_to_num(y_test)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mse = np.mean((y_pred.flatten() - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred.flatten() - y_test))

        # Calculate R-squared
        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_residual = np.sum((y_test - y_pred.flatten()) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        # Calculate WMAE (Weighted Mean Absolute Error) for holiday weighting
        # This is a placeholder - you'll need to adapt this to your actual test data structure

        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R-squared: {r_squared:.4f}")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared
        }

    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            raise ValueError("Model has not been trained yet")

        plt.figure(figsize=(12, 5))

        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout()

        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)

        # Save the plot
        plt.savefig('plots/training_history.png')
        plt.close()

    def save_model(self, filepath='models/lstm_walmart_sales.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model has not been built or trained yet")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='models/lstm_walmart_sales.h5'):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")

        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model