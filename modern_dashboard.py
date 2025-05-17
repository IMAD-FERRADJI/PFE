import sys
import os
import pandas as pd  # Add pandas import
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QTabWidget, QFrame, QStackedWidget, QScrollArea,
                             QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QLineEdit, QGraphicsDropShadowEffect, QSizePolicy)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QIcon, QFont, QPalette

# Import custom widgets and data functions
from modern_charts import LineChartWidget, BarChartWidget, PieChartWidget
from data_loader import load_local_csv, get_lstm_predictions, get_optimization_recommendations, get_inventory_summary, \
    get_category_distribution


class ModernDashboardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Inventory Management")
        self.setMinimumSize(1200, 800)

        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f9fafb;
            }
            QLabel[title="true"] {
                font-size: 24px;
                font-weight: bold;
                color: #111827;
            }
            QLabel[subtitle="true"] {
                font-size: 14px;
                color: #6b7280;
            }
            QFrame[card="true"] {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e5e7eb;
            }
            QTabWidget::pane {
                border: none;
                background-color: transparent;
            }
            QTabBar::tab {
                background-color: #f3f4f6;
                color: #6b7280;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3b82f6;
                color: white;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton[secondary="true"] {
                background-color: white;
                color: #3b82f6;
                border: 1px solid #3b82f6;
            }
            QPushButton[secondary="true"]:hover {
                background-color: #eff6ff;
            }
            QLineEdit {
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 8px;
            }
            QComboBox {
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 8px;
            }
            QTableWidget {
                border: 1px solid #e5e7eb;
                border-radius: 4px;
                gridline-color: #f3f4f6;
            }
            QHeaderView::section {
                background-color: #f9fafb;
                color: #4b5563;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #e5e7eb;
                font-weight: bold;
            }
            QScrollBar:vertical {
                border: none;
                background: #f3f4f6;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #d1d5db;
                border-radius: 4px;
            }
        """)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create header
        header = self.create_header()

        # Create content area
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(20)

        # Create tabs
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        # Create dashboard tab
        dashboard_tab = self.create_dashboard_tab()

        # Create inventory tab
        inventory_tab = self.create_inventory_tab()

        # Create suppliers tab
        suppliers_tab = self.create_suppliers_tab()

        # Create predictive tab
        predictive_tab = self.create_predictive_tab()

        # Add tabs
        tabs.addTab(dashboard_tab, "Dashboard")
        tabs.addTab(inventory_tab, "Inventory")
        tabs.addTab(suppliers_tab, "Suppliers")
        tabs.addTab(predictive_tab, "Predictive Analytics")

        # Add tabs to content layout
        content_layout.addWidget(tabs)

        # Add widgets to main layout
        main_layout.addWidget(header)
        main_layout.addWidget(content)

        # Load data
        self.load_data()

    def create_header(self):
        """Create the application header"""
        header = QFrame()
        header.setStyleSheet("background-color: white; border-bottom: 1px solid #e5e7eb;")
        header.setFixedHeight(64)

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)

        # Logo and title
        logo_container = QWidget()
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setSpacing(10)

        # Use a label as a placeholder for the logo
        logo = QLabel("📦")
        logo.setStyleSheet("font-size: 24px;")

        title = QLabel("Intelligent Inventory Management")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        logo_layout.addWidget(logo)
        logo_layout.addWidget(title)

        # Add to header layout
        header_layout.addWidget(logo_container)
        header_layout.addStretch()

        return header

    def create_dashboard_tab(self):
        """Create the dashboard tab"""
        dashboard = QWidget()
        dashboard_layout = QVBoxLayout(dashboard)
        dashboard_layout.setContentsMargins(0, 0, 0, 0)
        dashboard_layout.setSpacing(20)

        # Dashboard title
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Inventory Dashboard")
        title.setProperty("title", "true")

        last_updated = QLabel("Last updated: Just now")
        last_updated.setStyleSheet("color: #6b7280;")

        title_layout.addWidget(title)
        title_layout.addStretch()
        title_layout.addWidget(last_updated)

        # Metrics cards
        metrics_container = QWidget()
        metrics_layout = QHBoxLayout(metrics_container)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(20)

        # Create metric cards
        self.total_inventory_card = self.create_metric_card("Total Inventory", "0 items", "")
        self.low_stock_card = self.create_metric_card("Low Stock Items", "0 items", "")
        self.incoming_card = self.create_metric_card("Incoming Shipments", "N/A", "")
        self.value_card = self.create_metric_card("Inventory Value", "$0.00", "")

        metrics_layout.addWidget(self.total_inventory_card)
        metrics_layout.addWidget(self.low_stock_card)
        metrics_layout.addWidget(self.incoming_card)
        metrics_layout.addWidget(self.value_card)

        # Charts container
        charts_container = QFrame()
        charts_container.setProperty("card", "true")
        charts_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        charts_layout = QVBoxLayout(charts_container)

        # Charts tabs
        charts_tabs = QTabWidget()
        charts_tabs.setDocumentMode(True)

        # Inventory trends tab
        trends_tab = QWidget()
        trends_layout = QVBoxLayout(trends_tab)

        trends_title = QLabel("Inventory Trends")
        trends_title.setStyleSheet("font-size: 16px; font-weight: bold;")

        trends_subtitle = QLabel("Stock levels, sales, and restocking over time")
        trends_subtitle.setStyleSheet("color: #6b7280; font-size: 12px;")

        # Create line chart for inventory trends
        self.inventory_trend_chart = LineChartWidget()
        self.inventory_trend_chart.setMinimumHeight(300)

        trends_layout.addWidget(trends_title)
        trends_layout.addWidget(trends_subtitle)
        trends_layout.addWidget(self.inventory_trend_chart)

        # Category distribution tab
        category_tab = QWidget()
        category_layout = QVBoxLayout(category_tab)

        category_title = QLabel("Category Distribution")
        category_title.setStyleSheet("font-size: 16px; font-weight: bold;")

        category_subtitle = QLabel("Inventory distribution by product category")
        category_subtitle.setStyleSheet("color: #6b7280; font-size: 12px;")

        # Create pie chart for category distribution
        self.category_chart = PieChartWidget()
        self.category_chart.setMinimumHeight(300)

        category_layout.addWidget(category_title)
        category_layout.addWidget(category_subtitle)
        category_layout.addWidget(self.category_chart)

        # Add tabs to charts tabs
        charts_tabs.addTab(trends_tab, "Inventory Trends")
        charts_tabs.addTab(category_tab, "Category Distribution")

        charts_layout.addWidget(charts_tabs)

        # Add widgets to dashboard layout
        dashboard_layout.addWidget(title_container)
        dashboard_layout.addWidget(metrics_container)
        dashboard_layout.addWidget(charts_container)

        return dashboard

    def create_inventory_tab(self):
        """Create the inventory tab"""
        inventory = QWidget()
        inventory_layout = QVBoxLayout(inventory)
        inventory_layout.setContentsMargins(0, 0, 0, 0)
        inventory_layout.setSpacing(20)

        # Inventory title
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Inventory Management")
        title.setProperty("title", "true")

        add_button = QPushButton("Add Item")
        add_button.setIcon(QIcon("assets/plus.png"))

        title_layout.addWidget(title)
        title_layout.addStretch()
        title_layout.addWidget(add_button)

        # Search and filter
        search_container = QWidget()
        search_layout = QHBoxLayout(search_container)
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(10)

        search_input = QLineEdit()
        search_input.setPlaceholderText("Search inventory...")

        status_filter = QComboBox()
        status_filter.addItems(["All Items", "In Stock", "Low Stock", "Overstocked"])

        category_filter = QComboBox()
        category_filter.addItem("All Categories")

        search_layout.addWidget(search_input, 3)
        search_layout.addWidget(category_filter, 1)
        search_layout.addWidget(status_filter, 1)

        # Inventory table
        table_container = QFrame()
        table_container.setProperty("card", "true")

        table_layout = QVBoxLayout(table_container)

        self.inventory_table = QTableWidget()
        self.inventory_table.setColumnCount(7)
        self.inventory_table.setHorizontalHeaderLabels(
            ["SKU", "Name", "Category", "In Stock", "Min Stock", "Unit Price", "Status"])
        self.inventory_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.inventory_table.setAlternatingRowColors(True)
        self.inventory_table.setEditTriggers(QTableWidget.NoEditTriggers)

        table_layout.addWidget(self.inventory_table)

        # Add widgets to inventory layout
        inventory_layout.addWidget(title_container)
        inventory_layout.addWidget(search_container)
        inventory_layout.addWidget(table_container)

        return inventory

    def create_suppliers_tab(self):
        """Create the suppliers tab"""
        suppliers = QWidget()
        suppliers_layout = QVBoxLayout(suppliers)
        suppliers_layout.setContentsMargins(0, 0, 0, 0)
        suppliers_layout.setSpacing(20)

        # Suppliers title
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Suppliers")
        title.setProperty("title", "true")

        add_button = QPushButton("Add Supplier")
        add_button.setIcon(QIcon("assets/plus.png"))

        title_layout.addWidget(title)
        title_layout.addStretch()
        title_layout.addWidget(add_button)

        # Search
        search_container = QWidget()
        search_layout = QHBoxLayout(search_container)
        search_layout.setContentsMargins(0, 0, 0, 0)

        search_input = QLineEdit()
        search_input.setPlaceholderText("Search suppliers...")

        search_layout.addWidget(search_input)
        search_layout.addStretch()

        # Suppliers table
        table_container = QFrame()
        table_container.setProperty("card", "true")

        table_layout = QVBoxLayout(table_container)

        self.suppliers_table = QTableWidget()
        self.suppliers_table.setColumnCount(4)
        self.suppliers_table.setHorizontalHeaderLabels(["Supplier ID", "Supplier Name", "Products", "Total Items"])
        self.suppliers_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.suppliers_table.setAlternatingRowColors(True)
        self.suppliers_table.setEditTriggers(QTableWidget.NoEditTriggers)

        table_layout.addWidget(self.suppliers_table)

        # Add widgets to suppliers layout
        suppliers_layout.addWidget(title_container)
        suppliers_layout.addWidget(search_container)
        suppliers_layout.addWidget(table_container)

        return suppliers

    def create_predictive_tab(self):
        """Create the predictive analytics tab"""
        predictive = QWidget()
        predictive_layout = QVBoxLayout(predictive)
        predictive_layout.setContentsMargins(0, 0, 0, 0)
        predictive_layout.setSpacing(20)

        # Predictive title
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)

        title_text_container = QWidget()
        title_text_layout = QVBoxLayout(title_text_container)
        title_text_layout.setContentsMargins(0, 0, 0, 0)
        title_text_layout.setSpacing(5)

        title = QLabel("Predictive Analytics")
        title.setProperty("title", "true")

        subtitle = QLabel("LSTM-powered inventory forecasting")
        subtitle.setProperty("subtitle", "true")

        title_text_layout.addWidget(title)
        title_text_layout.addWidget(subtitle)

        # Controls
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        product_label = QLabel("Product:")
        self.product_combo = QComboBox()

        forecast_label = QLabel("Forecast:")
        self.forecast_combo = QComboBox()
        self.forecast_combo.addItems(["7 days forecast", "14 days forecast", "30 days forecast"])
        self.forecast_combo.setCurrentIndex(1)  # Default to 14 days

        retrain_button = QPushButton("Retrain Model")
        retrain_button.setProperty("secondary", "true")
        retrain_button.clicked.connect(self.retrain_model)

        controls_layout.addWidget(product_label)
        controls_layout.addWidget(self.product_combo)
        controls_layout.addWidget(forecast_label)
        controls_layout.addWidget(self.forecast_combo)
        controls_layout.addWidget(retrain_button)

        title_layout.addWidget(title_text_container)
        title_layout.addStretch()
        title_layout.addWidget(controls_container)

        # Model info banner
        self.info_banner = QFrame()
        self.info_banner.setStyleSheet("""
            background-color: #d1fae5;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #a7f3d0;
        """)

        info_layout = QHBoxLayout(self.info_banner)

        self.info_text = QLabel("LSTM model accuracy: 94.2% | Last trained: 2 days ago | MAE: 3.8 units")
        self.info_text.setStyleSheet("color: #065f46; font-weight: bold;")

        info_layout.addWidget(self.info_text)

        # Predictive tabs
        predictive_tabs = QTabWidget()
        predictive_tabs.setDocumentMode(True)

        # Forecast tab
        forecast_tab = QWidget()
        forecast_layout = QVBoxLayout(forecast_tab)

        # Create line chart for forecast
        self.forecast_chart = LineChartWidget()
        self.forecast_chart.setMinimumHeight(300)

        # Legend
        legend_container = QWidget()
        legend_layout = QHBoxLayout(legend_container)
        legend_layout.setContentsMargins(0, 0, 0, 0)

        actual_legend = self.create_legend_item("#10b981", "Actual Data")
        predicted_legend = self.create_legend_item("#3b82f6", "LSTM Prediction")
        confidence_legend = self.create_legend_item("#dbeafe", "Confidence Interval (95%)")

        legend_layout.addWidget(actual_legend)
        legend_layout.addWidget(predicted_legend)
        legend_layout.addWidget(confidence_legend)
        legend_layout.addStretch()

        forecast_layout.addWidget(self.forecast_chart)
        forecast_layout.addWidget(legend_container)

        # Optimization tab
        optimization_tab = QWidget()
        optimization_layout = QVBoxLayout(optimization_tab)

        # Create scroll area for recommendations
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        # Create container for recommendations
        self.recommendations_container = QWidget()
        self.recommendations_layout = QVBoxLayout(self.recommendations_container)
        self.recommendations_layout.setContentsMargins(0, 0, 0, 0)
        self.recommendations_layout.setSpacing(10)
        self.recommendations_layout.addStretch()

        scroll_area.setWidget(self.recommendations_container)

        optimization_layout.addWidget(scroll_area)

        # Add tabs to predictive tabs
        predictive_tabs.addTab(forecast_tab, "Demand Forecast")
        predictive_tabs.addTab(optimization_tab, "Inventory Optimization")

        # Add widgets to predictive layout
        predictive_layout.addWidget(title_container)
        predictive_layout.addWidget(self.info_banner)
        predictive_layout.addWidget(predictive_tabs)

        # Connect signals
        self.product_combo.currentTextChanged.connect(self.update_forecast)
        self.forecast_combo.currentIndexChanged.connect(self.update_forecast)

        return predictive

    def create_metric_card(self, title, value, change):
        """Create a metric card widget"""
        card = QFrame()
        card.setProperty("card", "true")

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        card.setGraphicsEffect(shadow)

        card_layout = QHBoxLayout(card)

        # Text container
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(5)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #6b7280; font-size: 14px;")

        value_label = QLabel(value)
        value_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #111827;")

        change_label = QLabel(change)
        if "+" in change:
            change_label.setStyleSheet("color: #10b981; font-size: 14px; font-weight: bold;")
        elif "-" in change:
            change_label.setStyleSheet("color: #ef4444; font-size: 14px; font-weight: bold;")
        else:
            change_label.setStyleSheet("color: #6b7280; font-size: 14px;")

        text_layout.addWidget(title_label)
        text_layout.addWidget(value_label)
        text_layout.addWidget(change_label)
        text_layout.addStretch()

        # Icon
        icon_label = QLabel("📊")
        icon_label.setStyleSheet("font-size: 24px; color: #9ca3af;")
        icon_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        card_layout.addWidget(text_container, 4)
        card_layout.addWidget(icon_label, 1)

        return card

    def create_legend_item(self, color, text):
        """Create a legend item with color box and text"""
        legend_item = QWidget()
        legend_layout = QHBoxLayout(legend_item)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(5)

        color_box = QFrame()
        color_box.setFixedSize(12, 12)
        color_box.setStyleSheet(f"background-color: {color}; border-radius: 2px;")

        text_label = QLabel(text)
        text_label.setStyleSheet("font-size: 12px;")

        legend_layout.addWidget(color_box)
        legend_layout.addWidget(text_label)

        return legend_item

    def create_recommendation_card(self, recommendation):
        """Create a recommendation card for the optimization tab"""
        card = QFrame()
        card.setProperty("card", "true")

        card_layout = QHBoxLayout(card)

        # Product info
        product_info = QWidget()
        product_layout = QVBoxLayout(product_info)
        product_layout.setContentsMargins(0, 0, 0, 0)
        product_layout.setSpacing(5)

        product_name = QLabel(recommendation["product"])
        product_name.setStyleSheet("font-weight: bold; font-size: 14px;")

        product_reason = QLabel(recommendation["reason"])
        product_reason.setStyleSheet("color: #6b7280; font-size: 12px;")

        product_layout.addWidget(product_name)
        product_layout.addWidget(product_reason)

        # Metrics
        metrics_widget = QWidget()
        metrics_layout = QHBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(20)

        # Current stock
        current_widget = QWidget()
        current_layout = QVBoxLayout(current_widget)
        current_layout.setContentsMargins(0, 0, 0, 0)
        current_layout.setAlignment(Qt.AlignCenter)

        current_label = QLabel("Current")
        current_label.setStyleSheet("color: #6b7280; font-size: 12px;")
        current_label.setAlignment(Qt.AlignCenter)

        current_value = QLabel(str(recommendation["currentStock"]))
        current_value.setStyleSheet("font-weight: bold; font-size: 16px;")
        current_value.setAlignment(Qt.AlignCenter)

        current_layout.addWidget(current_label)
        current_layout.addWidget(current_value)

        # Recommended stock
        recommended_widget = QWidget()
        recommended_layout = QVBoxLayout(recommended_widget)
        recommended_layout.setContentsMargins(0, 0, 0, 0)
        recommended_layout.setAlignment(Qt.AlignCenter)

        recommended_label = QLabel("Recommended")
        recommended_label.setStyleSheet("color: #6b7280; font-size: 12px;")
        recommended_label.setAlignment(Qt.AlignCenter)

        recommended_value = QLabel(str(recommendation["recommendedStock"]))
        if recommendation["recommendedStock"] > recommendation["currentStock"]:
            recommended_value.setStyleSheet("color: #10b981; font-weight: bold; font-size: 16px;")
        else:
            recommended_value.setStyleSheet("color: #f59e0b; font-weight: bold; font-size: 16px;")
        recommended_value.setAlignment(Qt.AlignCenter)

        recommended_layout.addWidget(recommended_label)
        recommended_layout.addWidget(recommended_value)

        # Confidence
        confidence_widget = QFrame()
        confidence_widget.setStyleSheet(f"""
            background-color: #d1fae5;
            border-radius: 4px;
            padding: 4px 8px;
            border: 1px solid #a7f3d0;
        """)
        confidence_layout = QHBoxLayout(confidence_widget)
        confidence_layout.setContentsMargins(8, 4, 8, 4)

        confidence_label = QLabel(f"{recommendation['confidence']}% confidence")
        confidence_label.setStyleSheet("color: #065f46; font-size: 12px; font-weight: bold;")

        confidence_layout.addWidget(confidence_label)

        # Apply button
        apply_button = QPushButton("Apply")
        apply_button.setProperty("secondary", "true")
        apply_button.clicked.connect(lambda: self.apply_recommendation(recommendation["product"]))

        # Add widgets to metrics layout
        metrics_layout.addWidget(current_widget)
        metrics_layout.addWidget(recommended_widget)
        metrics_layout.addWidget(confidence_widget)
        metrics_layout.addWidget(apply_button)

        # Add widgets to card layout
        card_layout.addWidget(product_info, 1)
        card_layout.addWidget(metrics_widget, 2)

        return card

    def load_data(self):
        """Load data and update UI"""
        try:
            print("Starting to load data...")
            # Load inventory data
            df = load_local_csv("inventory_data.csv")

            if df.empty:
                print("No data loaded")
                return

            print("Updating dashboard metrics...")
            # Update dashboard metrics
            try:
                summary = get_inventory_summary(df)

                self.total_inventory_card.findChild(QLabel, "", Qt.FindDirectChildrenOnly).setText(
                    f"{summary['total_quantity']:,} items")
                self.low_stock_card.findChild(QLabel, "", Qt.FindDirectChildrenOnly).setText(
                    f"{summary['low_stock_count']:,} items")
                self.value_card.findChild(QLabel, "", Qt.FindDirectChildrenOnly).setText(
                    f"${summary['inventory_value']:,.2f}")
            except Exception as e:
                print(f"Error updating dashboard metrics: {e}")

            print("Updating inventory table...")
            # Update inventory table
            try:
                self.update_inventory_table(df)
            except Exception as e:
                print(f"Error updating inventory table: {e}")

            print("Updating suppliers table...")
            # Update suppliers table
            try:
                self.update_suppliers_table(df)
            except Exception as e:
                print(f"Error updating suppliers table: {e}")

            print("Updating category chart...")
            # Update category chart
            try:
                categories, values = get_category_distribution(df)
                self.category_chart.update_chart(categories, values)
            except Exception as e:
                print(f"Error updating category chart: {e}")

            print("Updating trend chart...")
            # Update inventory trend chart
            try:
                self.update_trend_chart(df)
            except Exception as e:
                print(f"Error updating trend chart: {e}")

            print("Populating product combo...")
            # Populate product combo
            try:
                self.populate_product_combo(df)
            except Exception as e:
                print(f"Error populating product combo: {e}")

            print("Updating forecast...")
            # Update forecast
            try:
                self.update_forecast()
            except Exception as e:
                print(f"Error updating forecast: {e}")

            print("Updating recommendations...")
            # Update recommendations
            try:
                self.update_recommendations()
            except Exception as e:
                print(f"Error updating recommendations: {e}")

            print("Data loading and UI update completed successfully")

        except Exception as e:
            print(f"Error in load_data: {e}")

    def update_inventory_table(self, df):
        """Update the inventory table with data"""
        self.inventory_table.setRowCount(0)

        if 'Product_ID' not in df.columns:
            return

        for i, row in df.iterrows():
            self.inventory_table.insertRow(i)

            # SKU
            self.inventory_table.setItem(i, 0, QTableWidgetItem(str(row['Product_ID'])))

            # Name
            self.inventory_table.setItem(i, 1, QTableWidgetItem(str(row['Product_Name'])))

            # Category (first word of product name)
            category = str(row['Product_Name']).split()[0] if pd.notna(row['Product_Name']) else 'Unknown'
            self.inventory_table.setItem(i, 2, QTableWidgetItem(category))

            # In Stock
            self.inventory_table.setItem(i, 3, QTableWidgetItem(str(int(row['Stock_Quantity']))))

            # Min Stock
            self.inventory_table.setItem(i, 4, QTableWidgetItem(str(int(row['Reorder_Level']))))

            # Unit Price
            self.inventory_table.setItem(i, 5, QTableWidgetItem(f"${float(row['Unit_Price']):.2f}"))

            # Status
            status_item = QTableWidgetItem(row['Status'])
            if row['Status'] == 'Low Stock':
                status_item.setBackground(QColor('#fee2e2'))
            elif row['Status'] == 'Overstocked':
                status_item.setBackground(QColor('#fef3c7'))
            else:
                status_item.setBackground(QColor('#d1fae5'))

            self.inventory_table.setItem(i, 6, status_item)

    def update_suppliers_table(self, df):
        """Update the suppliers table with data"""
        self.suppliers_table.setRowCount(0)

        if 'Supplier_ID' not in df.columns or 'Supplier_Name' not in df.columns:
            return

        # Group by supplier
        suppliers = {}
        for i, row in df.iterrows():
            supplier_id = str(row['Supplier_ID'])
            supplier_name = str(row['Supplier_Name'])

            if supplier_id not in suppliers:
                suppliers[supplier_id] = {
                    'name': supplier_name,
                    'products': 0,
                    'items': 0
                }

            suppliers[supplier_id]['products'] += 1
            suppliers[supplier_id]['items'] += int(row['Stock_Quantity'])

        # Add to table
        for i, (supplier_id, data) in enumerate(suppliers.items()):
            self.suppliers_table.insertRow(i)

            # Supplier ID
            self.suppliers_table.setItem(i, 0, QTableWidgetItem(supplier_id))

            # Supplier Name
            self.suppliers_table.setItem(i, 1, QTableWidgetItem(data['name']))

            # Supplier ID
            self.suppliers_table.setItem(i, 0, QTableWidgetItem(supplier_id))

            # Supplier Name
            self.suppliers_table.setItem(i, 1, QTableWidgetItem(data['name']))

            # Products
            self.suppliers_table.setItem(i, 2, QTableWidgetItem(str(data['products'])))

            # Total Items
            self.suppliers_table.setItem(i, 3, QTableWidgetItem(str(data['items'])))

    def update_trend_chart(self, df):
        """Update the inventory trend chart"""
        from data_loader import get_time_series_data

        # Get time series data
        time_series = get_time_series_data(df)

        if time_series.empty:
            return

        # Prepare data for chart
        dates = [d.strftime('%Y-%m-%d') for d in time_series['date']]
        stock = time_series['stock'].tolist()
        sales = time_series['sales'].tolist()
        restock = time_series['restock'].tolist()

        # Update chart
        self.inventory_trend_chart.update_chart(
            dates,
            [
                {'name': 'Stock', 'data': stock, 'color': '#10b981'},
                {'name': 'Sales', 'data': sales, 'color': '#3b82f6'},
                {'name': 'Restock', 'data': restock, 'color': '#f59e0b'}
            ]
        )

    def populate_product_combo(self, df):
        """Populate the product combo box"""
        self.product_combo.clear()

        if 'Product_Name' not in df.columns:
            return

        products = df['Product_Name'].unique()
        self.product_combo.addItems(products)

    def update_forecast(self):
        """Update the forecast chart"""
        try:
            # Get selected product
            product_name = self.product_combo.currentText()
            if not product_name:
                print("No product selected for forecast")
                return

            print(f"Generating forecast for product: {product_name}")

            # Get selected forecast horizon
            forecast_text = self.forecast_combo.currentText()
            if "7 days" in forecast_text:
                forecast_days = 7
            elif "30 days" in forecast_text:
                forecast_days = 30
            else:
                forecast_days = 14  # Default

            # Load inventory data if needed
            df = load_local_csv("inventory_data.csv")

            # Get LSTM predictions
            predictions = get_lstm_predictions(df, product_name, 30, forecast_days)

            # Update info text with model metrics
            if 'mae' in predictions and 'rmse' in predictions and 'last_trained' in predictions:
                self.info_text.setText(
                    f"LSTM model accuracy: {100 - (predictions['mae'] / df['Stock_Quantity'].mean() * 100):.1f}% | "
                    f"Last trained: {predictions['last_trained']} | "
                    f"MAE: {predictions['mae']:.1f} units"
                )

            # Limit the data to reduce memory usage
            max_points = 60  # Limit to 60 data points
            if len(predictions['date']) > max_points:
                step = len(predictions['date']) // max_points
                predictions['date'] = predictions['date'][::step]
                predictions['actual'] = predictions['actual'][::step]
                predictions['predicted'] = predictions['predicted'][::step]
                predictions['lower'] = predictions['lower'][::step]
                predictions['upper'] = predictions['upper'][::step]

            # Create actual data series (only where values are not None)
            actual_data = []
            for i, v in enumerate(predictions['actual']):
                if v is not None:
                    actual_data.append({'x': i, 'y': v})

            # Create predicted data series (only where values are not None)
            predicted_data = []
            for i, v in enumerate(predictions['predicted']):
                if v is not None:
                    predicted_data.append({'x': i, 'y': v})

            # Create confidence interval series
            confidence_data = []
            for i, (l, u) in enumerate(zip(predictions['lower'], predictions['upper'])):
                if l is not None and u is not None:
                    confidence_data.append({'x': i, 'y': [l, u]})

            # Update chart
            self.forecast_chart.update_forecast_chart(
                predictions['date'],
                [
                    {'name': 'Actual', 'data': actual_data, 'color': '#10b981'},
                    {'name': 'Predicted', 'data': predicted_data, 'color': '#3b82f6'},
                ],
                confidence_data
            )

            print("Forecast updated successfully")

        except Exception as e:
            print(f"Error in update_forecast: {e}")
            # Set a simple message in case of error
            self.info_text.setText("Error generating forecast. See console for details.")
            self.info_text.setStyleSheet("color: #b91c1c; font-weight: bold;")

    def update_recommendations(self):
        """Update the optimization recommendations"""
        try:
            print("Updating recommendations...")
            # Clear existing recommendations
            for i in reversed(range(self.recommendations_layout.count() - 1)):  # -1 to keep the stretch
                widget = self.recommendations_layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()

            # Load inventory data if needed
            df = load_local_csv("inventory_data.csv")

            # Get optimization recommendations
            recommendations = get_optimization_recommendations(df)

            # Limit to 5 recommendations to reduce memory usage
            recommendations = recommendations[:5]

            # Create recommendation cards
            for recommendation in recommendations:
                card = self.create_recommendation_card(recommendation)
                self.recommendations_layout.insertWidget(self.recommendations_layout.count() - 1, card)

            print(f"Added {len(recommendations)} recommendation cards")

        except Exception as e:
            print(f"Error in update_recommendations: {e}")

    def apply_recommendation(self, product_name):
        """Apply a recommendation by selecting the product and updating the forecast"""
        # Find the product in the combo box
        index = self.product_combo.findText(product_name)
        if index >= 0:
            # Select the product
            self.product_combo.setCurrentIndex(index)
            # Update the forecast (this will happen automatically via the signal)

            # Show a message
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Recommendation Applied",
                f"Showing forecast for {product_name}."
            )

    def retrain_model(self):
        """Retrain the LSTM model"""
        # Load inventory data if needed
        df = load_local_csv("inventory_data.csv")

        # Get selected product
        product_name = self.product_combo.currentText()

        # Show a message that training is in progress
        self.info_text.setText("Training in progress... Please wait.")
        self.info_text.setStyleSheet("color: #1e40af; font-weight: bold;")
        self.info_banner.setStyleSheet("""
            background-color: #dbeafe;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #93c5fd;
        """)

        # Force UI update
        QApplication.processEvents()

        try:
            # Get LSTM predictions (this will retrain the model)
            predictions = get_lstm_predictions(df, product_name, 30, 14)

            # Update info text with model metrics
            if 'mae' in predictions and 'rmse' in predictions and 'last_trained' in predictions:
                self.info_text.setText(
                    f"LSTM model accuracy: {100 - (predictions['mae'] / df['Stock_Quantity'].mean() * 100):.1f}% | "
                    f"Last trained: {predictions['last_trained']} | "
                    f"MAE: {predictions['mae']:.1f} units"
                )
                self.info_text.setStyleSheet("color: #065f46; font-weight: bold;")
                self.info_banner.setStyleSheet("""
                    background-color: #d1fae5;
                    border-radius: 8px;
                    padding: 10px;
                    border: 1px solid #a7f3d0;
                """)

            # Update the forecast
            self.update_forecast()

            # Update recommendations
            self.update_recommendations()

            # Show success message
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Model Trained",
                "LSTM model has been successfully retrained with the latest data."
            )

        except Exception as e:
            # Show error message
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Training Error",
                f"An error occurred while training the model: {str(e)}"
            )

            # Reset info text
            self.info_text.setText("LSTM model accuracy: N/A | Last trained: Never | MAE: N/A")
            self.info_text.setStyleSheet("color: #b91c1c; font-weight: bold;")
            self.info_banner.setStyleSheet("""
                background-color: #fee2e2;
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #fca5a5;
            """)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style to Fusion for a more modern look
    app.setStyle("Fusion")

    window = ModernDashboardApp()
    window.show()

    sys.exit(app.exec_())
