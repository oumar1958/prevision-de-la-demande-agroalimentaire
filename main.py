"""
Main entry point for the agro demand forecasting project
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.settings import PROJECT_ROOT, DATA_DIR
from src.data.scrapers import MockDataGenerator, MockWeatherGenerator
from src.data.pipeline import DataCleaner, FeatureEngineer
from src.data.database import db_manager
from src.models import ProphetForecaster, XGBoostForecaster, LSTMForecaster
from src.business import ProductionSimulator, BusinessImpactAnalyzer
from src.visualization import AgroDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DATA_DIR / 'logs' / 'agro_forecasting.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    directories = [
        DATA_DIR / 'logs',
        DATA_DIR / 'models',
        DATA_DIR / 'raw',
        DATA_DIR / 'processed'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info("Directories created/verified")


def run_full_pipeline():
    """Run the complete forecasting pipeline"""
    logger.info("Starting full forecasting pipeline")
    
    try:
        # 1. Data Generation
        logger.info("Step 1: Generating sample data...")
        mock_generator = MockDataGenerator()
        products_df = mock_generator.generate_historical_data(days=365)
        
        weather_generator = MockWeatherGenerator()
        weather_df = weather_generator.generate_historical_weather(days=365)
        
        logger.info(f"Generated {len(products_df)} product records and {len(weather_df)} weather records")
        
        # 2. Data Cleaning
        logger.info("Step 2: Cleaning data...")
        cleaner = DataCleaner()
        
        products_clean = cleaner.clean_product_data(products_df)
        weather_clean = cleaner.clean_weather_data(weather_df)
        
        # Store in database
        db_manager.insert_products(products_clean)
        db_manager.insert_weather(weather_clean)
        
        logger.info("Data cleaning completed")
        
        # 3. Feature Engineering
        logger.info("Step 3: Feature engineering...")
        feature_engineer = FeatureEngineer()
        
        merged_data = db_manager.get_merged_data()
        features_df = feature_engineer.create_features(merged_data, 'availability_rate')
        
        logger.info(f"Feature engineering completed. Dataset has {len(features_df.columns)} features")
        
        # 4. Model Training
        logger.info("Step 4: Training models...")
        
        # Prophet
        prophet_forecaster = ProphetForecaster()
        prophet_metrics = prophet_forecaster.train_category_models(features_df, 'availability_rate')
        
        # XGBoost
        xgboost_forecaster = XGBoostForecaster()
        xgboost_metrics = xgboost_forecaster.train_category_models(features_df, 'availability_rate')
        
        # LSTM (if available)
        lstm_metrics = {}
        try:
            lstm_forecaster = LSTMForecaster()
            lstm_metrics = lstm_forecaster.train_category_models(features_df, 'availability_rate')
        except Exception as e:
            logger.warning(f"LSTM training skipped: {e}")
        
        logger.info("Model training completed")
        
        # 5. Generate Forecasts
        logger.info("Step 5: Generating forecasts...")
        
        prophet_forecasts = prophet_forecaster.predict_all_categories(periods=30)
        xgboost_forecasts = xgboost_forecaster.predict_all_categories({})
        
        logger.info("Forecasts generated")
        
        # 6. Business Simulation
        logger.info("Step 6: Running business simulation...")
        
        simulator = ProductionSimulator()
        comparison_results = simulator.compare_strategies(
            merged_data, prophet_forecasts, 'availability_rate'
        )
        
        logger.info("Business simulation completed")
        
        # 7. Generate Report
        logger.info("Step 7: Generating business report...")
        
        business_analyzer = BusinessImpactAnalyzer()
        report = simulator.generate_business_report(comparison_results)
        recommendations = business_analyzer.generate_recommendations(comparison_results)
        
        # Save report
        with open(DATA_DIR / 'logs' / 'business_report.txt', 'w') as f:
            f.write(report)
        
        logger.info("Pipeline completed successfully!")
        
        return {
            'products_data': products_clean,
            'weather_data': weather_clean,
            'features_data': features_df,
            'prophet_metrics': prophet_metrics,
            'xgboost_metrics': xgboost_metrics,
            'lstm_metrics': lstm_metrics,
            'prophet_forecasts': prophet_forecasts,
            'comparison_results': comparison_results,
            'report': report,
            'recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def run_dashboard():
    """Run the Streamlit dashboard"""
    logger.info("Starting Streamlit dashboard")
    
    # Import and run dashboard
    import streamlit.web.cli as stcli
    import sys
    
    dashboard_script = Path(__file__).parent / "src" / "visualization" / "dashboard.py"
    
    sys.argv = ["streamlit", "run", str(dashboard_script)]
    sys.exit(stcli.main())


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agro Demand Forecasting System")
    parser.add_argument(
        "--mode", 
        choices=["pipeline", "dashboard", "full"],
        default="dashboard",
        help="Run mode: pipeline (data processing only), dashboard (UI only), or full (both)"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    if args.mode == "pipeline":
        run_full_pipeline()
    elif args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "full":
        # Run pipeline first, then dashboard
        results = run_full_pipeline()
        logger.info("Pipeline completed. Starting dashboard...")
        run_dashboard()


if __name__ == "__main__":
    main()
