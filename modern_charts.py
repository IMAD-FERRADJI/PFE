import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class ModernCharts:
    def __init__(self, output_dir='plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['xtick.color'] = '#333333'
        plt.rcParams['ytick.color'] = '#333333'

    def plot_sales_by_store(self, df, title='Weekly Sales by Store'):
        """Plot average weekly sales by store"""
        plt.figure(figsize=(12, 6))

        # Calculate average sales by store
        store_sales = df.groupby('Store')['Weekly_Sales'].mean().sort_values(ascending=False)

        # Plot
        ax = sns.barplot(x=store_sales.index, y=store_sales.values, palette='viridis')

        # Customize
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Store', fontsize=12)
        plt.ylabel('Average Weekly Sales ($)', fontsize=12)
        plt.xticks(rotation=90)

        # Add value labels on top of bars
        for i, v in enumerate(store_sales.values):
            ax.text(i, v + 500, f'${int(v):,}', ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sales_by_store.png'), dpi=300)
        plt.close()

    def plot_sales_by_dept(self, df, top_n=10, title='Weekly Sales by Department'):
        """Plot average weekly sales by department (top N)"""
        plt.figure(figsize=(12, 6))

        # Calculate average sales by department
        dept_sales = df.groupby('Dept')['Weekly_Sales'].mean().sort_values(ascending=False).head(top_n)

        # Plot
        ax = sns.barplot(x=dept_sales.index, y=dept_sales.values, palette='magma')

        # Customize
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Department', fontsize=12)
        plt.ylabel('Average Weekly Sales ($)', fontsize=12)

        # Add value labels on top of bars
        for i, v in enumerate(dept_sales.values):
            ax.text(i, v + 500, f'${int(v):,}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sales_by_dept.png'), dpi=300)
        plt.close()

    def plot_sales_trend(self, df, title='Weekly Sales Trend Over Time'):
        """Plot sales trend over time"""
        plt.figure(figsize=(14, 6))

        # Aggregate sales by date
        sales_by_date = df.groupby('Date')['Weekly_Sales'].mean().reset_index()

        # Plot
        plt.plot(sales_by_date['Date'], sales_by_date['Weekly_Sales'],
                 marker='o', markersize=4, linestyle='-', linewidth=1, color='#1f77b4')

        # Highlight holidays
        if 'IsHoliday' in df.columns:
            holiday_dates = df[df['IsHoliday'] == True]['Date'].unique()
            holiday_sales = []

            for date in holiday_dates:
                if date in sales_by_date['Date'].values:
                    idx = sales_by_date[sales_by_date['Date'] == date].index[0]
                    holiday_sales.append((date, sales_by_date.loc[idx, 'Weekly_Sales']))

            if holiday_sales:
                holiday_dates, holiday_values = zip(*holiday_sales)
                plt.scatter(holiday_dates, holiday_values, color='red', s=80,
                            label='Holiday Weeks', zorder=5, alpha=0.7)

        # Customize
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Weekly Sales ($)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add legend if holidays were plotted
        if 'IsHoliday' in df.columns and len(holiday_sales) > 0:
            plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sales_trend.png'), dpi=300)
        plt.close()

    def plot_correlation_heatmap(self, df, title='Feature Correlation Heatmap'):
        """Plot correlation heatmap of numerical features"""
        plt.figure(figsize=(12, 10))

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Calculate correlation matrix
        corr = numeric_df.corr()

        # Plot heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
                    linewidths=0.5, cbar_kws={'shrink': .8})

        # Customize
        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'), dpi=300)
        plt.close()

    def plot_feature_importance(self, model, feature_names, title='Feature Importance'):
        """Plot feature importance for tree-based models"""
        plt.figure(figsize=(12, 6))

        # Get feature importance
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Plot
            plt.bar(range(len(importances)), importances[indices], align='center', color='#1f77b4')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)

            # Customize
            plt.title(title, fontsize=16, pad=20)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Importance', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300)
            plt.close()
        except:
            print("Model doesn't support feature_importances_ attribute")

    def plot_sales_by_store_type(self, df, title='Sales by Store Type'):
        """Plot sales distribution by store type"""
        if 'Type' not in df.columns:
            print("Store type information not available")
            return

        plt.figure(figsize=(10, 6))

        # Calculate average sales by store type
        type_sales = df.groupby('Type')['Weekly_Sales'].mean().reset_index()

        # Plot
        ax = sns.barplot(x='Type', y='Weekly_Sales', data=type_sales, palette='Set2')

        # Customize
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Store Type', fontsize=12)
        plt.ylabel('Average Weekly Sales ($)', fontsize=12)

        # Add value labels on top of bars
        for i, v in enumerate(type_sales['Weekly_Sales']):
            ax.text(i, v + 500, f'${int(v):,}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sales_by_store_type.png'), dpi=300)
        plt.close()

    def plot_sales_by_holiday(self, df, title='Sales Comparison: Holiday vs. Non-Holiday'):
        """Compare sales between holiday and non-holiday weeks"""
        if 'IsHoliday' not in df.columns:
            print("Holiday information not available")
            return

        plt.figure(figsize=(8, 6))

        # Calculate average sales by holiday flag
        holiday_sales = df.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()

        # Map boolean to string for better labels
        holiday_sales['IsHoliday'] = holiday_sales['IsHoliday'].map({True: 'Holiday', False: 'Non-Holiday'})

        # Plot
        ax = sns.barplot(x='IsHoliday', y='Weekly_Sales', data=holiday_sales, palette=['#ff7f0e', '#1f77b4'])

        # Customize
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Week Type', fontsize=12)
        plt.ylabel('Average Weekly Sales ($)', fontsize=12)

        # Add value labels on top of bars
        for i, v in enumerate(holiday_sales['Weekly_Sales']):
            ax.text(i, v + 500, f'${int(v):,}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sales_by_holiday.png'), dpi=300)
        plt.close()

    def plot_pca_analysis(self, df, target_col='Weekly_Sales', n_components=2, title='PCA Analysis'):
        """Plot PCA analysis of features"""
        # Select only numeric columns excluding the target
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col in numeric_df.columns:
            numeric_df = numeric_df.drop(target_col, axis=1)

        # Drop any columns with all NaN values
        numeric_df = numeric_df.dropna(axis=1, how='all')

        # Fill remaining NaN values with column means
        numeric_df = numeric_df.fillna(numeric_df.mean())

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Apply PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)

        # Create DataFrame with principal components
        pca_df = pd.DataFrame(data=principal_components,
                              columns=[f'PC{i + 1}' for i in range(n_components)])

        # Add target variable if available
        if target_col in df.columns:
            pca_df[target_col] = df[target_col].values

        # Plot
        plt.figure(figsize=(10, 8))

        if target_col in df.columns:
            scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'],
                                  c=pca_df[target_col], cmap='viridis',
                                  alpha=0.6, edgecolors='w', linewidth=0.5)
            plt.colorbar(scatter, label=target_col)
        else:
            plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6,
                        edgecolors='w', linewidth=0.5)

        # Customize
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_analysis.png'), dpi=300)
        plt.close()

        # Print explained variance
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2%}")

    def plot_prediction_vs_actual(self, y_true, y_pred, title='Prediction vs Actual'):
        """Plot predicted values against actual values"""
        plt.figure(figsize=(10, 6))

        # Create scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, color='#1f77b4')

        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        # Customize
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Actual Sales', fontsize=12)
        plt.ylabel('Predicted Sales', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Calculate metrics
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)

        # Add metrics as text
        plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'prediction_vs_actual.png'), dpi=300)
        plt.close()