"""
Streamlit dashboard for agro demand forecasting visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.database import db_manager
from src.data.scrapers import MockDataGenerator, MockWeatherGenerator
from src.data.pipeline import DataCleaner, FeatureEngineer
from src.models import ProphetForecaster, XGBoostForecaster, LSTMForecaster
from src.business import ProductionSimulator, BusinessImpactAnalyzer

logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Agro Demand Forecasting Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
    }
    .insight-box {
        background-color: #fffacd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FFD700;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class AgroDashboard:
    """
    Main dashboard class for agro demand forecasting
    """
    
    def __init__(self):
        self.db_manager = db_manager
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.production_simulator = ProductionSimulator()
        self.business_analyzer = BusinessImpactAnalyzer()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'simulation_done' not in st.session_state:
            st.session_state.simulation_done = False
    
    def run(self):
        """Main dashboard application"""
        st.markdown('<h1 class="main-header">🌾 Agro Demand Forecasting Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["📊 Data Overview", "🤖 Model Training", "📈 Forecasts", "💼 Business Simulation", "📋 Recommendations"],
            index=0
        )
        
        # Data loading section in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Data Management")
        
        if st.sidebar.button("Load Sample Data", type="primary"):
            self.load_sample_data()
        
        if st.sidebar.button("Train Models", disabled=not st.session_state.data_loaded):
            self.train_models()
        
        if st.sidebar.button("Run Simulation", disabled=not st.session_state.models_trained):
            self.run_business_simulation()
        
        # Status indicators
        self.show_status()
        
        # Main content
        if page == "📊 Data Overview":
            self.show_data_overview()
        elif page == "🤖 Model Training":
            self.show_model_training()
        elif page == "📈 Forecasts":
            self.show_forecasts()
        elif page == "💼 Business Simulation":
            self.show_business_simulation()
        elif page == "📋 Recommendations":
            self.show_recommendations()
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        try:
            with st.spinner("Generating sample data..."):
                # Generate mock product data
                mock_generator = MockDataGenerator()
                products_df = mock_generator.generate_historical_data(days=365)
                
                # Generate mock weather data
                weather_generator = MockWeatherGenerator()
                weather_df = weather_generator.generate_historical_weather(days=365)
                
                # Clean data
                products_clean = self.data_cleaner.clean_product_data(products_df)
                weather_clean = self.data_cleaner.clean_weather_data(weather_df)
                
                # Store in database
                self.db_manager.insert_products(products_clean)
                self.db_manager.insert_weather(weather_clean)
                
                st.session_state.data_loaded = True
                st.session_state.products_data = products_clean
                st.session_state.weather_data = weather_clean
                
            st.success("✅ Sample data loaded successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
    
    def train_models(self):
        """Train forecasting models"""
        try:
            with st.spinner("Training models..."):
                # Get merged data
                merged_data = self.db_manager.get_merged_data()
                
                if merged_data.empty:
                    st.error("No data available for training")
                    return
                
                # Feature engineering
                features_df = self.feature_engineer.create_features(merged_data, 'availability_rate')
                
                # Train Prophet models
                prophet_forecaster = ProphetForecaster()
                prophet_metrics = prophet_forecaster.train_category_models(features_df, 'availability_rate')
                
                # Train XGBoost models
                xgboost_forecaster = XGBoostForecaster()
                xgboost_metrics = xgboost_forecaster.train_category_models(features_df, 'availability_rate')
                
                # Train LSTM models (if TensorFlow available)
                lstm_metrics = {}
                try:
                    lstm_forecaster = LSTMForecaster()
                    lstm_metrics = lstm_forecaster.train_category_models(features_df, 'availability_rate')
                except:
                    st.warning("LSTM models skipped (TensorFlow not available)")
                
                st.session_state.models_trained = True
                st.session_state.prophet_forecaster = prophet_forecaster
                st.session_state.xgboost_forecaster = xgboost_forecaster
                st.session_state.prophet_metrics = prophet_metrics
                st.session_state.xgboost_metrics = xgboost_metrics
                st.session_state.lstm_metrics = lstm_metrics
                
            st.success("✅ Models trained successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error training models: {e}")
    
    def run_business_simulation(self):
        """Run business simulation"""
        try:
            with st.spinner("Running business simulation..."):
                # Get data
                merged_data = self.db_manager.get_merged_data()
                
                if merged_data.empty:
                    st.error("No data available for simulation")
                    return
                
                # Get forecasts
                if 'prophet_forecaster' in st.session_state:
                    prophet_forecasts = st.session_state.prophet_forecaster.predict_all_categories(periods=30)
                else:
                    prophet_forecasts = pd.DataFrame()
                
                # Run simulation
                comparison_results = self.production_simulator.compare_strategies(
                    merged_data, prophet_forecasts, 'availability_rate'
                )
                
                st.session_state.simulation_done = True
                st.session_state.simulation_results = comparison_results
                
            st.success("✅ Business simulation completed!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error running simulation: {e}")
    
    def show_status(self):
        """Show current status in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Status")
        
        status_data = [
            ("Data Loaded", st.session_state.data_loaded),
            ("Models Trained", st.session_state.models_trained),
            ("Simulation Done", st.session_state.simulation_done)
        ]
        
        for label, status in status_data:
            color = "🟢" if status else "🔴"
            st.sidebar.write(f"{color} {label}")
    
    def show_data_overview(self):
        """Show data overview page"""
        st.header("📊 Data Overview")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load sample data first from the sidebar.")
            return
        
        # Get data
        products_data = st.session_state.products_data
        weather_data = st.session_state.weather_data
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", len(products_data))
        
        with col2:
            st.metric("Categories", products_data['category'].nunique())
        
        with col3:
            st.metric("Weather Records", len(weather_data))
        
        with col4:
            st.metric("Date Range", f"{weather_data['date'].min().date()} to {weather_data['date'].max().date()}")
        
        # Product data visualization
        st.subheader("🛒 Product Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_counts = products_data['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index, title="Products by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price trends
            price_trend = products_data.groupby('scraped_at')['current_price'].mean().reset_index()
            fig = px.line(price_trend, x='scraped_at', y='current_price', title="Average Price Trend")
            st.plotly_chart(fig, use_container_width=True)
        
        # Weather data visualization
        st.subheader("🌤️ Weather Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature trend
            temp_trend = weather_data.groupby('date')['temperature_2m'].mean().reset_index()
            fig = px.line(temp_trend, x='date', y='temperature_2m', title="Temperature Trend")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Precipitation distribution
            fig = px.histogram(weather_data, x='precipitation', title="Precipitation Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Data quality report
        st.subheader("📋 Data Quality Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            products_quality = self.data_cleaner.generate_data_quality_report(products_data, 'products')
            st.write("**Product Data Quality**")
            st.json(products_quality)
        
        with col2:
            weather_quality = self.data_cleaner.generate_data_quality_report(weather_data, 'weather')
            st.write("**Weather Data Quality**")
            st.json(weather_quality)
    
    def show_model_training(self):
        """Show model training page"""
        st.header("🤖 Model Training")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first from the sidebar.")
            return
        
        # Model performance comparison
        st.subheader("📊 Model Performance Comparison")
        
        prophet_metrics = st.session_state.prophet_metrics
        xgboost_metrics = st.session_state.xgboost_metrics
        lstm_metrics = st.session_state.lstm_metrics
        
        # Create performance comparison table
        performance_data = []
        
        for category in prophet_metrics.keys():
            if 'mape' in prophet_metrics[category]:
                performance_data.append({
                    'Category': category,
                    'Model': 'Prophet',
                    'MAPE': f"{prophet_metrics[category]['mape']:.4f}",
                    'RMSE': f"{prophet_metrics[category]['rmse']:.4f}",
                    'MAE': f"{prophet_metrics[category]['mae']:.4f}"
                })
        
        for category in xgboost_metrics.keys():
            if 'mape' in xgboost_metrics[category]:
                performance_data.append({
                    'Category': category,
                    'Model': 'XGBoost',
                    'MAPE': f"{xgboost_metrics[category]['mape']:.4f}",
                    'RMSE': f"{xgboost_metrics[category]['rmse']:.4f}",
                    'MAE': f"{xgboost_metrics[category]['mae']:.4f}"
                })
        
        for category in lstm_metrics.keys():
            if 'mape' in lstm_metrics[category]:
                performance_data.append({
                    'Category': category,
                    'Model': 'LSTM',
                    'MAPE': f"{lstm_metrics[category]['mape']:.4f}",
                    'RMSE': f"{lstm_metrics[category]['rmse']:.4f}",
                    'MAE': f"{lstm_metrics[category]['mae']:.4f}"
                })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)
            
            # Performance visualization
            st.subheader("📈 Performance Visualization")
            
            # MAPE comparison by category
            fig = px.bar(performance_df, x='Category', y='MAPE', color='Model', 
                        title="MAPE Comparison by Category", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (XGBoost)
            if 'xgboost_forecaster' in st.session_state:
                st.subheader("🎯 Feature Importance (XGBoost)")
                
                xgboost_forecaster = st.session_state.xgboost_forecaster
                feature_importance = xgboost_forecaster.get_global_feature_importance()
                
                if feature_importance:
                    # Get top 15 features
                    top_features = dict(list(feature_importance.items())[:15])
                    
                    fig = px.bar(x=list(top_features.keys()), y=list(top_features.values()),
                                title="Top 15 Feature Importance", orientation='h')
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_forecasts(self):
        """Show forecasts page"""
        st.header("📈 Demand Forecasts")
        
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first from the sidebar.")
            return
        
        # Generate forecasts
        st.subheader("🔮 30-Day Forecast")
        
        # Get forecasts from Prophet
        if 'prophet_forecaster' in st.session_state:
            prophet_forecaster = st.session_state.prophet_forecaster
            prophet_forecasts = prophet_forecaster.predict_all_categories(periods=30)
            
            if not prophet_forecasts.empty:
                # Forecast visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Select category
                    categories = prophet_forecasts['category'].unique()
                    selected_category = st.selectbox("Select Category", categories)
                    
                    # Filter data
                    category_forecasts = prophet_forecasts[prophet_forecasts['category'] == selected_category]
                    
                    # Plot forecast
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=category_forecasts['date'],
                        y=category_forecasts['predicted_demand'],
                        mode='lines+markers',
                        name='Predicted Demand',
                        line=dict(color='blue')
                    ))
                    
                    if 'confidence_lower' in category_forecasts.columns:
                        fig.add_trace(go.Scatter(
                            x=category_forecasts['date'],
                            y=category_forecasts['confidence_upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=category_forecasts['date'],
                            y=category_forecasts['confidence_lower'],
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(0,100,80,0.2)',
                            name='Confidence Interval',
                            showlegend=True
                        ))
                    
                    fig.update_layout(
                        title=f"Demand Forecast - {selected_category}",
                        xaxis_title="Date",
                        yaxis_title="Predicted Demand"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Forecast summary
                    st.subheader("Forecast Summary")
                    
                    summary_stats = {
                        'Average Demand': f"{category_forecasts['predicted_demand'].mean():.2f}",
                        'Min Demand': f"{category_forecasts['predicted_demand'].min():.2f}",
                        'Max Demand': f"{category_forecasts['predicted_demand'].max():.2f}",
                        'Demand Volatility': f"{category_forecasts['predicted_demand'].std():.2f}"
                    }
                    
                    for metric, value in summary_stats.items():
                        st.metric(metric, value)
        
        # Historical vs Forecast comparison
        st.subheader("📊 Historical vs Forecast Comparison")
        
        # Get historical data
        merged_data = self.db_manager.get_merged_data()
        
        if not merged_data.empty and 'prophet_forecaster' in st.session_state:
            # Select category for comparison
            comparison_category = st.selectbox("Select Category for Comparison", merged_data['category'].unique())
            
            # Filter data
            historical_data = merged_data[merged_data['category'] == comparison_category]
            forecast_data = prophet_forecasts[prophet_forecasts['category'] == comparison_category]
            
            # Create comparison plot
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['availability_rate'],
                mode='lines',
                name='Historical Demand',
                line=dict(color='green')
            ))
            
            # Forecast data
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['predicted_demand'],
                mode='lines',
                name='Forecast',
                line=dict(color='blue', dash='dash')
            ))
            
            fig.update_layout(
                title=f"Historical vs Forecast - {comparison_category}",
                xaxis_title="Date",
                yaxis_title="Demand"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_business_simulation(self):
        """Show business simulation page"""
        st.header("💼 Business Simulation")
        
        if not st.session_state.simulation_done:
            st.warning("⚠️ Please run business simulation first from the sidebar.")
            return
        
        simulation_results = st.session_state.simulation_results
        
        # Strategy comparison table
        st.subheader("📊 Strategy Comparison")
        
        comparison_df = simulation_results['comparison_table']
        st.dataframe(comparison_df, use_container_width=True)
        
        # Key metrics visualization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(comparison_df, x='Strategy', y='Service Level (%)',
                        title="Service Level Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_df, x='Strategy', y='Waste (%)',
                        title="Waste Percentage Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.bar(comparison_df, x='Strategy', y='Total Cost (€)',
                        title="Total Cost Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        # Cost breakdown
        st.subheader("💰 Cost Breakdown Analysis")
        
        cost_cols = ['Storage Cost (€)', 'Shortage Cost (€)', 'Waste Cost (€)']
        cost_df = comparison_df.set_index('Strategy')[cost_cols]
        
        fig = px.bar(cost_df, x=cost_df.index, y=cost_cols,
                    title="Cost Breakdown by Strategy", barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
        
        # Best strategy recommendation
        if 'best_strategy' in simulation_results and simulation_results['best_strategy']:
            best_strategy = simulation_results['best_strategy']
            
            st.markdown(f"""
            <div class="insight-box">
                <h3>🎯 Recommended Strategy: {best_strategy['strategy']}</h3>
                <p><strong>Composite Score:</strong> {best_strategy['score']:.4f}</p>
                <p>This strategy provides the optimal balance between service level, cost efficiency, and waste reduction.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Improvements vs baseline
        if 'improvements' in simulation_results and simulation_results['improvements']:
            st.subheader("📈 Improvements vs Baseline")
            
            improvements = simulation_results['improvements']
            
            for strategy, metrics in improvements.items():
                with st.expander(f"{strategy.upper()} Strategy Improvements"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Cost Reduction", f"{metrics['cost_reduction_pct']:.2f}%")
                    
                    with col2:
                        st.metric("Waste Reduction", f"{metrics['waste_reduction_pct']:.2f}%")
                    
                    with col3:
                        st.metric("Service Level Improvement", f"{metrics['service_level_improvement']:.2f}%")
    
    def show_recommendations(self):
        """Show recommendations page"""
        st.header("📋 Business Recommendations")
        
        if not st.session_state.simulation_done:
            st.warning("⚠️ Please run business simulation first from the sidebar.")
            return
        
        simulation_results = st.session_state.simulation_results
        
        # Generate recommendations
        recommendations = self.business_analyzer.generate_recommendations(simulation_results)
        
        st.subheader("💡 Key Recommendations")
        
        for i, recommendation in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="metric-card">
                <strong>{i}. {recommendation}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # ROI Analysis
        st.subheader("💰 ROI Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            implementation_cost = st.number_input("Implementation Cost (€)", value=10000, min_value=0)
        
        with col2:
            annual_savings = st.number_input("Annual Savings (€)", value=5000, min_value=0)
        
        if st.button("Calculate ROI"):
            roi_results = self.business_analyzer.analyze_roi(implementation_cost, annual_savings)
            
            st.markdown(f"""
            <div class="insight-box">
                <h3>ROI Analysis Results</h3>
                <p><strong>Net Present Value (NPV):</strong> €{roi_results['npv']:.2f}</p>
                <p><strong>Payback Period:</strong> {roi_results['payback_period_years']} years</p>
                <p><strong>ROI:</strong> {roi_results['roi_percentage']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Implementation roadmap
        st.subheader("🗺️ Implementation Roadmap")
        
        roadmap_data = [
            {"Phase": "Phase 1", "Duration": "1-2 months", "Activities": "Data collection and cleaning", "Priority": "High"},
            {"Phase": "Phase 2", "Duration": "2-3 months", "Activities": "Model development and training", "Priority": "High"},
            {"Phase": "Phase 3", "Duration": "1 month", "Activities": "System integration and testing", "Priority": "Medium"},
            {"Phase": "Phase 4", "Duration": "Ongoing", "Activities": "Monitoring and optimization", "Priority": "Medium"}
        ]
        
        roadmap_df = pd.DataFrame(roadmap_data)
        st.dataframe(roadmap_df, use_container_width=True)
        
        # Success metrics
        st.subheader("🎯 Success Metrics")
        
        metrics_data = [
            {"Metric": "Forecast Accuracy", "Target": ">90%", "Current": "85%", "Status": "🟡"},
            {"Metric": "Service Level", "Target": ">95%", "Current": "92%", "Status": "🟡"},
            {"Metric": "Waste Reduction", "Target": ">15%", "Current": "12%", "Status": "🟡"},
            {"Metric": "Cost Reduction", "Target": ">10%", "Current": "8%", "Status": "🟡"}
        ]
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)


def main():
    """Main function to run the dashboard"""
    dashboard = AgroDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
