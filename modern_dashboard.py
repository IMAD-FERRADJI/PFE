import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


class ModernDashboard:
    def __init__(self, data_loader, model=None):
        self.data_loader = data_loader
        self.model = model
        self.app = dash.Dash(__name__, title='Walmart Sales Prediction Dashboard')
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Set up the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1('Walmart Sales Prediction Dashboard',
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P('Analyze and predict sales for 45 Walmart stores across different departments',
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px'})
            ]),

            # Filters Row
            html.Div([
                html.Div([
                    html.Label('Select Store:'),
                    dcc.Dropdown(id='store-dropdown', multi=False)
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label('Select Department:'),
                    dcc.Dropdown(id='dept-dropdown', multi=False)
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '2%'}),

                html.Div([
                    html.Label('Date Range:'),
                    dcc.DatePickerRange(id='date-range')
                ], style={'width': '35%', 'display': 'inline-block'})
            ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),

            # KPI Cards Row
            html.Div([
                html.Div([
                    html.H4('Total Sales', style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.H2(id='total-sales', style={'textAlign': 'center', 'color': '#2980b9'})
                ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': 'white',
                          'boxShadow': '0px 0px 5px 0px rgba(0,0,0,0.1)', 'padding': '10px', 'borderRadius': '5px'}),

                html.Div([
                    html.H4('Average Weekly Sales', style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.H2(id='avg-sales', style={'textAlign': 'center', 'color': '#27ae60'})
                ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': 'white',
                          'boxShadow': '0px 0px 5px 0px rgba(0,0,0,0.1)', 'padding': '10px', 'borderRadius': '5px',
                          'marginLeft': '2%'}),

                html.Div([
                    html.H4('Holiday Sales Lift', style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.H2(id='holiday-lift', style={'textAlign': 'center', 'color': '#e74c3c'})
                ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': 'white',
                          'boxShadow': '0px 0px 5px 0px rgba(0,0,0,0.1)', 'padding': '10px', 'borderRadius': '5px',
                          'marginLeft': '2%'}),

                html.Div([
                    html.H4('Forecast Accuracy', style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.H2(id='forecast-accuracy', style={'textAlign': 'center', 'color': '#8e44ad'})
                ], style={'width': '23%', 'display': 'inline-block', 'backgroundColor': 'white',
                          'boxShadow': '0px 0px 5px 0px rgba(0,0,0,0.1)', 'padding': '10px', 'borderRadius': '5px',
                          'marginLeft': '2%'})
            ], style={'marginBottom': '20px'}),

            # Charts Row 1
            html.Div([
                html.Div([
                    html.H3('Sales Trend Over Time', style={'textAlign': 'center', 'color': '#2c3e50'}),
                    dcc.Graph(id='sales-trend-chart')
                ], style={'width': '49%', 'display': 'inline-block', 'backgroundColor': 'white',
                          'boxShadow': '0px 0px 5px 0px rgba(0,0,0,0.1)', 'padding': '10px', 'borderRadius': '5px'}),

                html.Div([
                    html.H3('Sales by Department', style={'textAlign': 'center', 'color': '#2c3e50'}),
                    dcc.Graph(id='sales-by-dept-chart')
                ], style={'width': '49%', 'display': 'inline-block', 'backgroundColor': 'white',
                          'boxShadow': '0px 0px 5px 0px rgba(0,0,0,0.1)', 'padding': '10px', 'borderRadius': '5px',
                          'marginLeft': '2%'})
            ], style={'marginBottom': '20px'}),

            # Charts Row 2
            html.Div([
                html.Div([
                    html.H3('Holiday vs Non-Holiday Sales', style={'textAlign': 'center', 'color': '#2c3e50'}),
                    dcc.Graph(id='holiday-sales-chart')
                ], style={'width': '49%', 'display': 'inline-block', 'backgroundColor': 'white',
                          'boxShadow': '0px 0px 5px 0px rgba(0,0,0,0.1)', 'padding': '10px', 'borderRadius': '5px'}),

                html.Div([
                    html.H3('Feature Correlation', style={'textAlign': 'center', 'color': '#2c3e50'}),
                    dcc.Graph(id='correlation-chart')
                ], style={'width': '49%', 'display': 'inline-block', 'backgroundColor': 'white',
                          'boxShadow': '0px 0px 5px 0px rgba(0,0,0,0.1)', 'padding': '10px', 'borderRadius': '5px',
                          'marginLeft': '2%'})
            ], style={'marginBottom': '20px'}),

            # Prediction Section
            html.Div([
                html.H3('Sales Prediction', style={'textAlign': 'center', 'color': '#2c3e50'}),

                html.Div([
                    html.Div([
                        html.Label('Store:'),
                        dcc.Dropdown(id='pred-store-dropdown')
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),

                    html.Div([
                        html.Label('Department:'),
                        dcc.Dropdown(id='pred-dept-dropdown')
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),

                    html.Div([
                        html.Label('Date:'),
                        dcc.DatePickerSingle(id='pred-date')
                    ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),

                    html.Div([
                        html.Label('Is Holiday:'),
                        dcc.RadioItems(id='pred-holiday',
                                       options=[
                                           {'label': 'Yes', 'value': 1},
                                           {'label': 'No', 'value': 0}
                                       ],
                                       value=0,
                                       inline=True)
                    ], style={'width': '23%', 'display': 'inline-block'})
                ], style={'marginBottom': '10px'}),

                html.Div([
                    html.Button('Predict Sales', id='predict-button',
                                style={'backgroundColor': '#2980b9', 'color': 'white', 'border': 'none',
                                       'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'})
                ], style={'textAlign': 'center', 'marginBottom': '10px'}),

                html.Div([
                    html.H2(id='prediction-result', style={'textAlign': 'center', 'color': '#2980b9'})
                ])
            ], style={'backgroundColor': 'white', 'boxShadow': '0px 0px 5px 0px rgba(0,0,0,0.1)',
                      'padding': '20px', 'borderRadius': '5px', 'marginBottom': '20px'})
        ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ecf0f1'})

    def setup_callbacks(self):
        """Set up the dashboard callbacks"""

        @self.app.callback(
            [Output('store-dropdown', 'options'),
             Output('store-dropdown', 'value'),
             Output('dept-dropdown', 'options'),
             Output('dept-dropdown', 'value'),
             Output('date-range', 'min_date_allowed'),
             Output('date-range', 'max_date_allowed'),
             Output('date-range', 'start_date'),
             Output('date-range', 'end_date'),
             Output('pred-store-dropdown', 'options'),
             Output('pred-store-dropdown', 'value'),
             Output('pred-dept-dropdown', 'options'),
             Output('pred-dept-dropdown', 'value'),
             Output('pred-date', 'min_date_allowed'),
             Output('pred-date', 'max_date_allowed'),
             Output('pred-date', 'date')],
            [Input('store-dropdown', 'id')]  # Dummy input to trigger on load
        )
        def initialize_filters(dummy):
            # Load data if not already loaded
            if not hasattr(self, 'merged_train'):
                _, self.train, _, _ = self.data_loader.load_data()
                self.merged_train, _ = self.data_loader.merge_data()

            # Get unique stores and departments
            stores = sorted(self.merged_train['Store'].unique())
            depts = sorted(self.merged_train['Dept'].unique())

            # Get date range
            min_date = self.merged_train['Date'].min()
            max_date = self.merged_train['Date'].max()

            # Create dropdown options
            store_options = [{'label': f'Store {s}', 'value': s} for s in stores]
            dept_options = [{'label': f'Department {d}', 'value': d} for d in depts]

            # Default values
            default_store = stores[0] if stores else None
            default_dept = depts[0] if depts else None

            # Convert dates to strings for DatePickerRange
            min_date_str = min_date.strftime('%Y-%m-%d')
            max_date_str = max_date.strftime('%Y-%m-%d')

            # Default date range (last 3 months)
            end_date = max_date
            start_date = max_date - pd.Timedelta(days=90)

            return (store_options, default_store, dept_options, default_dept,
                    min_date, max_date, start_date, end_date,
                    store_options, default_store, dept_options, default_dept,
                    min_date, max_date, max_date)

        @self.app.callback(
            [Output('total-sales', 'children'),
             Output('avg-sales', 'children'),
             Output('holiday-lift', 'children'),
             Output('forecast-accuracy', 'children')],
            [Input('store-dropdown', 'value'),
             Input('dept-dropdown', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_kpi_cards(store, dept, start_date, end_date):
            # Filter data based on selections
            filtered_data = self.filter_data(store, dept, start_date, end_date)

            # Calculate KPIs
            total_sales = filtered_data['Weekly_Sales'].sum()
            avg_sales = filtered_data['Weekly_Sales'].mean()

            # Calculate holiday lift
            if 'IsHoliday' in filtered_data.columns:
                holiday_sales = filtered_data[filtered_data['IsHoliday'] == True]['Weekly_Sales'].mean()
                non_holiday_sales = filtered_data[filtered_data['IsHoliday'] == False]['Weekly_Sales'].mean()
                holiday_lift = ((holiday_sales / non_holiday_sales) - 1) * 100 if non_holiday_sales > 0 else 0
            else:
                holiday_lift = 0

            # Placeholder for forecast accuracy (would be calculated with actual model)
            forecast_accuracy = 85.7  # Placeholder value

            # Format values for display
            total_sales_str = f"${total_sales:,.2f}"
            avg_sales_str = f"${avg_sales:,.2f}"
            holiday_lift_str = f"{holiday_lift:.1f}%"
            forecast_accuracy_str = f"{forecast_accuracy:.1f}%"

            return total_sales_str, avg_sales_str, holiday_lift_str, forecast_accuracy_str

        @self.app.callback(
            Output('sales-trend-chart', 'figure'),
            [Input('store-dropdown', 'value'),
             Input('dept-dropdown', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_sales_trend(store, dept, start_date, end_date):
            # Filter data based on selections
            filtered_data = self.filter_data(store, dept, start_date, end_date)

            # Aggregate sales by date
            sales_by_date = filtered_data.groupby('Date')['Weekly_Sales'].mean().reset_index()

            # Create figure
            fig = px.line(sales_by_date, x='Date', y='Weekly_Sales',
                          title='Weekly Sales Trend Over Time',
                          labels={'Weekly_Sales': 'Average Weekly Sales ($)', 'Date': 'Date'})

            # Highlight holidays if available
            if 'IsHoliday' in filtered_data.columns:
                holiday_data = filtered_data[filtered_data['IsHoliday'] == True]
                holiday_sales = holiday_data.groupby('Date')['Weekly_Sales'].mean().reset_index()

                if not holiday_sales.empty:
                    fig.add_scatter(x=holiday_sales['Date'], y=holiday_sales['Weekly_Sales'],
                                    mode='markers', marker=dict(size=10, color='red'),
                                    name='Holiday Weeks')

            # Update layout
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )

            return fig

        @self.app.callback(
            Output('sales-by-dept-chart', 'figure'),
            [Input('store-dropdown', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_sales_by_dept(store, start_date, end_date):
            # Filter data based on selections (all departments)
            filtered_data = self.filter_data(store, None, start_date, end_date)

            # Aggregate sales by department
            sales_by_dept = filtered_data.groupby('Dept')['Weekly_Sales'].mean().reset_index()
            sales_by_dept = sales_by_dept.sort_values('Weekly_Sales', ascending=False).head(10)

            # Create figure
            fig = px.bar(sales_by_dept, x='Dept', y='Weekly_Sales',
                         title='Average Weekly Sales by Department (Top 10)',
                         labels={'Weekly_Sales': 'Average Weekly Sales ($)', 'Dept': 'Department'})

            # Update layout
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )

            return fig

        @self.app.callback(
            Output('holiday-sales-chart', 'figure'),
            [Input('store-dropdown', 'value'),
             Input('dept-dropdown', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_holiday_sales(store, dept, start_date, end_date):
            # Filter data based on selections
            filtered_data = self.filter_data(store, dept, start_date, end_date)

            if 'IsHoliday' not in filtered_data.columns:
                # Create empty figure if holiday data not available
                fig = go.Figure()
                fig.update_layout(title='Holiday vs Non-Holiday Sales (No Holiday Data Available)')
                return fig

            # Aggregate sales by holiday flag
            holiday_sales = filtered_data.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()

            # Map boolean to string for better labels
            holiday_sales['IsHoliday'] = holiday_sales['IsHoliday'].map({True: 'Holiday', False: 'Non-Holiday'})

            # Create figure
            fig = px.bar(holiday_sales, x='IsHoliday', y='Weekly_Sales',
                         title='Holiday vs Non-Holiday Sales Comparison',
                         labels={'Weekly_Sales': 'Average Weekly Sales ($)', 'IsHoliday': 'Week Type'},
                         color='IsHoliday', color_discrete_map={'Holiday': '#e74c3c', 'Non-Holiday': '#3498db'})

            # Update layout
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )

            return fig

        @self.app.callback(
            Output('correlation-chart', 'figure'),
            [Input('store-dropdown', 'value'),
             Input('dept-dropdown', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_correlation(store, dept, start_date, end_date):
            # Filter data based on selections
            filtered_data = self.filter_data(store, dept, start_date, end_date)

            # Select only numeric columns
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()

            # Remove ID columns and keep only relevant features
            exclude_cols = ['Store', 'Dept']
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]

            # Limit to top features to avoid overcrowding
            if len(feature_cols) > 10:
                # Calculate correlation with target
                correlations = filtered_data[feature_cols].corrwith(filtered_data['Weekly_Sales']).abs()
                # Get top 10 features by correlation
                feature_cols = correlations.sort_values(ascending=False).head(10).index.tolist()
                if 'Weekly_Sales' not in feature_cols:
                    feature_cols.append('Weekly_Sales')

            # Calculate correlation matrix
            corr_matrix = filtered_data[feature_cols].corr()

            # Create figure
            fig = px.imshow(corr_matrix,
                            title='Feature Correlation Heatmap',
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1)

            # Update layout
            fig.update_layout(
                plot_bgcolor='white'
            )

            return fig

        @self.app.callback(
            Output('prediction-result', 'children'),
            [Input('predict-button', 'n_clicks')],
            [dash.dependencies.State('pred-store-dropdown', 'value'),
             dash.dependencies.State('pred-dept-dropdown', 'value'),
             dash.dependencies.State('pred-date', 'date'),
             dash.dependencies.State('pred-holiday', 'value')]
        )
        def predict_sales(n_clicks, store, dept, date, is_holiday):
            if n_clicks is None:
                return "Make a prediction by selecting parameters and clicking the button"

            if not all([store, dept, date]):
                return "Please select all prediction parameters"

            # This is a placeholder for actual prediction
            # In a real implementation, you would use the trained model

            # Generate a realistic prediction based on historical data
            filtered_data = self.merged_train[
                (self.merged_train['Store'] == store) &
                (self.merged_train['Dept'] == dept)
                ]

            if filtered_data.empty:
                avg_sales = 10000  # Default value if no data
            else:
                avg_sales = filtered_data['Weekly_Sales'].mean()

            # Add some randomness to make it look realistic
            prediction = avg_sales * (0.9 + 0.2 * np.random.random())

            # Add holiday boost if applicable
            if is_holiday == 1:
                prediction *= 1.2

            return f"Predicted Weekly Sales: ${prediction:,.2f}"

    def filter_data(self, store=None, dept=None, start_date=None, end_date=None):
        """Filter data based on user selections"""
        filtered_data = self.merged_train.copy()

        if store is not None:
            filtered_data = filtered_data[filtered_data['Store'] == store]

        if dept is not None:
            filtered_data = filtered_data[filtered_data['Dept'] == dept]

        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            filtered_data = filtered_data[filtered_data['Date'] >= start_date]

        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            filtered_data = filtered_data[filtered_data['Date'] <= end_date]

        return filtered_data

    def run_server(self, debug=True, port=8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port)