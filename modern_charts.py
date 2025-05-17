from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFrame, QSizePolicy
from PyQt5.QtCore import Qt, QPointF, QRectF, QLineF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont, QFontMetrics, QPainterPath


class BaseChartWidget(QWidget):
    """Base class for chart widgets"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create frame for chart
        self.chart_frame = QFrame()
        self.chart_frame.setFrameShape(QFrame.NoFrame)
        self.chart_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self.chart_frame)

        # Set default colors
        self.background_color = QColor(255, 255, 255)
        self.axis_color = QColor(156, 163, 175)  # #9ca3af
        self.text_color = QColor(107, 114, 128)  # #6b7280
        self.grid_color = QColor(243, 244, 246, 128)  # #f3f4f6 with alpha

        # Set default margins
        self.margin_left = 60
        self.margin_right = 20
        self.margin_top = 20
        self.margin_bottom = 40

        # Set default font
        self.font = QFont("Arial", 9)

        # Initialize data
        self.data = []
        self.x_labels = []

        # Set up painter
        self.chart_frame.paintEvent = self.paintEvent

    def paintEvent(self, event):
        """Paint the chart"""
        try:
            painter = QPainter(self.chart_frame)
            painter.setRenderHint(QPainter.Antialiasing)

            # Fill background
            painter.fillRect(event.rect(), self.background_color)

            # Draw chart
            self.draw_chart(painter, event.rect())

            painter.end()
        except Exception as e:
            print(f"Error in paintEvent: {e}")

    def draw_chart(self, painter, rect):
        """Draw the chart (to be implemented by subclasses)"""
        pass

    def draw_axes(self, painter, rect):
        """Draw x and y axes"""
        try:
            # Set pen for axes
            pen = QPen(self.axis_color)
            pen.setWidth(1)
            painter.setPen(pen)

            # Draw y-axis - using QLineF to avoid type issues
            painter.drawLine(
                QLineF(
                    self.margin_left, self.margin_top,
                    self.margin_left, rect.height() - self.margin_bottom
                )
            )

            # Draw x-axis - using QLineF to avoid type issues
            painter.drawLine(
                QLineF(
                    self.margin_left, rect.height() - self.margin_bottom,
                                      rect.width() - self.margin_right, rect.height() - self.margin_bottom
                )
            )
        except Exception as e:
            print(f"Error in draw_axes: {e}")

    def draw_grid(self, painter, rect, num_y_lines=5):
        """Draw grid lines"""
        try:
            # Set pen for grid
            pen = QPen(self.grid_color)
            pen.setWidth(1)
            painter.setPen(pen)

            # Draw horizontal grid lines
            chart_height = rect.height() - self.margin_top - self.margin_bottom
            step = chart_height / (num_y_lines - 1) if num_y_lines > 1 else chart_height

            for i in range(num_y_lines):
                y = rect.height() - self.margin_bottom - i * step
                # Use QLineF to avoid type issues
                painter.drawLine(
                    QLineF(
                        self.margin_left, y,
                        rect.width() - self.margin_right, y
                    )
                )

            # Draw vertical grid lines (if x labels are available)
            if self.x_labels:
                chart_width = rect.width() - self.margin_left - self.margin_right
                step = chart_width / (len(self.x_labels) - 1) if len(self.x_labels) > 1 else chart_width

                for i in range(len(self.x_labels)):
                    x = self.margin_left + i * step
                    # Use QLineF to avoid type issues
                    painter.drawLine(
                        QLineF(
                            x, self.margin_top,
                            x, rect.height() - self.margin_bottom
                        )
                    )
        except Exception as e:
            print(f"Error in draw_grid: {e}")

    def draw_y_labels(self, painter, rect, min_value, max_value, num_labels=5):
        """Draw y-axis labels"""
        try:
            # Set pen for text
            painter.setPen(self.text_color)
            painter.setFont(self.font)

            # Calculate step
            value_range = max_value - min_value
            step = value_range / (num_labels - 1) if num_labels > 1 else value_range

            # Draw labels
            chart_height = rect.height() - self.margin_top - self.margin_bottom
            y_step = chart_height / (num_labels - 1) if num_labels > 1 else chart_height

            for i in range(num_labels):
                value = max_value - i * step
                y = self.margin_top + i * y_step

                # Format value
                if value >= 1000:
                    text = f"{value / 1000:.1f}k"
                else:
                    text = f"{value:.0f}"

                # Draw text
                painter.drawText(
                    5, int(y + 5),  # Convert to int to avoid type issues
                    text
                )
        except Exception as e:
            print(f"Error in draw_y_labels: {e}")

    def draw_x_labels(self, painter, rect):
        """Draw x-axis labels"""
        try:
            if not self.x_labels:
                return

            # Set pen for text
            painter.setPen(self.text_color)
            painter.setFont(self.font)

            # Calculate step
            chart_width = rect.width() - self.margin_left - self.margin_right
            step = chart_width / (len(self.x_labels) - 1) if len(self.x_labels) > 1 else chart_width

            # Draw only a subset of labels if there are too many
            max_labels = 10
            label_indices = []

            if len(self.x_labels) <= max_labels:
                label_indices = range(len(self.x_labels))
            else:
                # Show first, last, and some in between
                step_size = len(self.x_labels) // (max_labels - 1)
                label_indices = list(range(0, len(self.x_labels), step_size))
                if label_indices[-1] != len(self.x_labels) - 1:
                    label_indices.append(len(self.x_labels) - 1)

            for i in label_indices:
                x = self.margin_left + i * step

                # Get text width
                metrics = QFontMetrics(self.font)
                text_width = metrics.width(self.x_labels[i])

                # Draw text - convert to int to avoid type issues
                painter.drawText(
                    int(x - text_width // 2), rect.height() - 10,
                    self.x_labels[i]
                )
        except Exception as e:
            print(f"Error in draw_x_labels: {e}")


class LineChartWidget(BaseChartWidget):
    """Widget for displaying line charts"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize series data
        self.series = []

    def update_chart(self, x_labels, series):
        """Update chart with new data"""
        try:
            self.x_labels = x_labels
            self.series = series
            self.update()
        except Exception as e:
            print(f"Error in LineChartWidget.update_chart: {e}")

    def update_forecast_chart(self, x_labels, series, confidence_data=None):
        """Update chart with forecast data including confidence intervals"""
        try:
            self.x_labels = x_labels
            self.series = series
            self.confidence_data = confidence_data
            self.update()
        except Exception as e:
            print(f"Error in LineChartWidget.update_forecast_chart: {e}")

    def draw_chart(self, painter, rect):
        """Draw the line chart"""
        try:
            if not self.series or not self.x_labels:
                return

            # Draw axes and grid
            self.draw_axes(painter, rect)

            # Find min and max values
            all_values = []
            for series in self.series:
                for point in series['data']:
                    if isinstance(point, dict) and 'y' in point:
                        all_values.append(point['y'])
                    elif isinstance(point, (int, float)):
                        all_values.append(point)

            # Add confidence interval values if available
            if hasattr(self, 'confidence_data') and self.confidence_data:
                for point in self.confidence_data:
                    if isinstance(point, dict) and 'y' in point and isinstance(point['y'], list):
                        all_values.extend(point['y'])

            if not all_values:
                return

            min_value = min(all_values) * 0.9  # Add some padding
            max_value = max(all_values) * 1.1  # Add some padding

            # Draw grid and labels
            self.draw_grid(painter, rect)
            self.draw_y_labels(painter, rect, min_value, max_value)
            self.draw_x_labels(painter, rect)

            # Draw confidence intervals if available
            if hasattr(self, 'confidence_data') and self.confidence_data:
                self.draw_confidence_intervals(painter, rect, min_value, max_value)

            # Draw each series
            for series in self.series:
                self.draw_series(painter, rect, series, min_value, max_value)
        except Exception as e:
            print(f"Error in LineChartWidget.draw_chart: {e}")

    def draw_confidence_intervals(self, painter, rect, min_value, max_value):
        """Draw confidence intervals"""
        try:
            if not self.confidence_data:
                return

            # Set brush for confidence intervals
            brush = QBrush(QColor(219, 234, 254, 128))  # #dbeafe with alpha
            painter.setBrush(brush)
            painter.setPen(Qt.NoPen)

            # Calculate scales
            chart_width = rect.width() - self.margin_left - self.margin_right
            chart_height = rect.height() - self.margin_top - self.margin_bottom
            x_scale = chart_width / (len(self.x_labels) - 1) if len(self.x_labels) > 1 else chart_width
            y_scale = chart_height / (max_value - min_value) if max_value > min_value else chart_height

            # Create path for the confidence interval
            path = QPainterPath()

            # Start with the first point
            if self.confidence_data:
                first_point = self.confidence_data[0]
                x = self.margin_left + first_point['x'] * x_scale
                y_upper = rect.height() - self.margin_bottom - (first_point['y'][1] - min_value) * y_scale
                path.moveTo(x, y_upper)

            # Add upper bound points
            for point in self.confidence_data:
                x = self.margin_left + point['x'] * x_scale
                y_upper = rect.height() - self.margin_bottom - (point['y'][1] - min_value) * y_scale
                path.lineTo(x, y_upper)

            # Add lower bound points in reverse order
            for point in reversed(self.confidence_data):
                x = self.margin_left + point['x'] * x_scale
                y_lower = rect.height() - self.margin_bottom - (point['y'][0] - min_value) * y_scale
                path.lineTo(x, y_lower)

            # Close the path
            path.closeSubpath()

            # Draw the path
            painter.drawPath(path)
        except Exception as e:
            print(f"Error in LineChartWidget.draw_confidence_intervals: {e}")

    def draw_series(self, painter, rect, series, min_value, max_value):
        """Draw a single data series"""
        try:
            # Set pen for line
            pen = QPen(QColor(series['color']))
            pen.setWidth(2)
            painter.setPen(pen)

            # Calculate scales
            chart_width = rect.width() - self.margin_left - self.margin_right
            chart_height = rect.height() - self.margin_top - self.margin_bottom

            # Check if data is in point format or simple array
            if series['data'] and isinstance(series['data'][0], dict):
                # Point format with x, y coordinates
                points = []

                for point in series['data']:
                    x = self.margin_left + point['x'] * chart_width / (len(self.x_labels) - 1) if len(
                        self.x_labels) > 1 else self.margin_left
                    y = rect.height() - self.margin_bottom - (point['y'] - min_value) * chart_height / (
                                max_value - min_value) if max_value > min_value else rect.height() - self.margin_bottom
                    points.append(QPointF(x, y))

                # Draw lines between points
                for i in range(len(points) - 1):
                    painter.drawLine(points[i], points[i + 1])

                # Draw points
                painter.setBrush(QBrush(QColor(series['color'])))
                for point in points:
                    painter.drawEllipse(point, 4, 4)

            else:
                # Simple array format
                x_scale = chart_width / (len(series['data']) - 1) if len(series['data']) > 1 else chart_width
                y_scale = chart_height / (max_value - min_value) if max_value > min_value else chart_height

                points = []

                for i, value in enumerate(series['data']):
                    x = self.margin_left + i * x_scale
                    y = rect.height() - self.margin_bottom - (value - min_value) * y_scale
                    points.append(QPointF(x, y))

                # Draw lines between points
                for i in range(len(points) - 1):
                    painter.drawLine(points[i], points[i + 1])

                # Draw points
                painter.setBrush(QBrush(QColor(series['color'])))
                for point in points:
                    painter.drawEllipse(point, 4, 4)
        except Exception as e:
            print(f"Error in LineChartWidget.draw_series: {e}")


class BarChartWidget(BaseChartWidget):
    """Widget for displaying bar charts"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize data
        self.values = []

    def update_chart(self, x_labels, values):
        """Update chart with new data"""
        try:
            self.x_labels = x_labels
            self.values = values
            self.update()
        except Exception as e:
            print(f"Error in BarChartWidget.update_chart: {e}")

    def draw_chart(self, painter, rect):
        """Draw the bar chart"""
        try:
            if not self.values or not self.x_labels:
                return

            # Draw axes and grid
            self.draw_axes(painter, rect)

            # Find min and max values
            min_value = 0
            max_value = max(self.values) * 1.1  # Add some padding

            # Draw grid and labels
            self.draw_grid(painter, rect)
            self.draw_y_labels(painter, rect, min_value, max_value)
            self.draw_x_labels(painter, rect)

            # Draw bars
            chart_width = rect.width() - self.margin_left - self.margin_right
            chart_height = rect.height() - self.margin_top - self.margin_bottom
            bar_width = chart_width / len(self.values) * 0.8  # 80% of available space
            spacing = chart_width / len(self.values) * 0.2  # 20% of available space

            # Set brush for bars
            painter.setBrush(QBrush(QColor(136, 132, 216)))  # #8884d8
            painter.setPen(Qt.NoPen)

            for i, value in enumerate(self.values):
                x = self.margin_left + i * (bar_width + spacing) + spacing / 2
                y = rect.height() - self.margin_bottom - value * chart_height / max_value
                height = value * chart_height / max_value

                painter.drawRect(QRectF(x, y, bar_width, height))
        except Exception as e:
            print(f"Error in BarChartWidget.draw_chart: {e}")


class PieChartWidget(BaseChartWidget):
    """Widget for displaying pie charts"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize data
        self.categories = []
        self.values = []

        # Set colors
        self.colors = [
            QColor(0, 136, 254),  # #0088FE
            QColor(0, 196, 159),  # #00C49F
            QColor(255, 187, 40),  # #FFBB28
            QColor(255, 128, 66),  # #FF8042
            QColor(136, 132, 216),  # #8884D8
            QColor(130, 202, 157),  # #82CA9D
            QColor(255, 107, 107),  # #FF6B6B
            QColor(107, 102, 255)  # #6B66FF
        ]

    def update_chart(self, categories, values):
        """Update chart with new data"""
        try:
            self.categories = categories
            self.values = values
            self.update()
        except Exception as e:
            print(f"Error in PieChartWidget.update_chart: {e}")

    def draw_chart(self, painter, rect):
        """Draw the pie chart"""
        try:
            if not self.values or not self.categories:
                return

            # Fill background
            painter.fillRect(rect, self.background_color)

            # Calculate total
            total = sum(self.values)
            if total <= 0:
                return  # Avoid division by zero

            # Calculate center and radius
            center_x = rect.width() / 2
            center_y = rect.height() / 2
            radius = min(center_x, center_y) * 0.7

            # Draw pie slices
            start_angle = 0

            for i, value in enumerate(self.values):
                if i >= len(self.colors):  # Ensure we don't go out of bounds
                    break

                # Calculate angle
                angle = 360 * value / total

                # Set brush for slice
                color = self.colors[i % len(self.colors)]
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.NoPen)

                # Draw slice
                painter.drawPie(
                    int(center_x - radius), int(center_y - radius),
                    int(radius * 2), int(radius * 2),
                    int(start_angle * 16), int(angle * 16)
                )

                # Calculate label position
                mid_angle = start_angle + angle / 2
                label_radius = radius * 0.7
                label_x = center_x + label_radius * 0.8 * np.cos(np.radians(mid_angle))
                label_y = center_y + label_radius * 0.8 * np.sin(np.radians(mid_angle))

                # Draw label
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("Arial", 9, QFont.Bold))

                # Format percentage
                percentage = value / total * 100
                label = f"{percentage:.0f}%"

                # Get text width and height
                metrics = QFontMetrics(painter.font())
                text_width = metrics.width(label)
                text_height = metrics.height()

                # Draw text - convert to int to avoid type issues
                painter.drawText(
                    int(label_x - text_width // 2),
                    int(label_y + text_height // 4),
                    label
                )

                # Update start angle
                start_angle += angle

            # Draw legend
            legend_x = rect.width() - 150
            legend_y = 20

            painter.setFont(QFont("Arial", 9))

            for i, category in enumerate(self.categories):
                if i >= len(self.colors):  # Ensure we don't go out of bounds
                    break

                # Set brush for legend color box
                color = self.colors[i % len(self.colors)]
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.NoPen)

                # Draw color box
                painter.drawRect(int(legend_x), int(legend_y + i * 20), 12, 12)

                # Draw category name
                painter.setPen(self.text_color)
                painter.drawText(int(legend_x + 20), int(legend_y + i * 20 + 10), category)

        except Exception as e:
            print(f"Error in PieChartWidget.draw_chart: {e}")


import numpy as np  # Required for trigonometric functions in PieChartWidget
