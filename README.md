# 🌾 Agro Demand Forecasting System

## 📋 Overview

A comprehensive end-to-end machine learning project for **demand forecasting and waste reduction in the agroalimentary sector**. This system combines web scraping, advanced ML models, and business simulation to optimize production planning and minimize food waste.

### 🎯 Business Objectives

- **Anticipate demand** for agroalimentary products with high accuracy
- **Reduce overproduction** and stock shortages by 15-25%
- **Minimize food waste** through optimized production planning
- **Provide actionable insights** for supply chain decision-making
- **Demonstrate ROI** through business impact simulation

### 🏗️ Architecture

```
agro_demand_forecasting/
├── src/
│   ├── config/           # Configuration settings
│   ├── data/             # Data collection & processing
│   │   ├── scrapers/     # Web scraping modules
│   │   ├── database/     # Database management
│   │   └── pipeline/     # Data cleaning & feature engineering
│   ├── models/           # ML forecasting models
│   ├── business/         # Business simulation & ROI analysis
│   └── visualization/    # Streamlit dashboard
├── data/                 # Data storage
├── notebooks/            # Exploratory analysis
└── main.py              # Main entry point
```

## 🚀 Features

### 📊 Data Collection
- **Web Scraping**: Automated collection of product prices, promotions, and availability
- **Weather Integration**: Historical and forecast weather data integration
- **Multi-source Data**: Support for multiple retail websites and APIs
- **Data Validation**: Comprehensive data quality checks and cleaning

### 🤖 Machine Learning Models
- **Prophet**: Facebook's time series forecasting for baseline predictions
- **XGBoost**: Gradient boosting with external features (weather, promotions, seasonality)
- **LSTM**: Deep learning for complex multivariate time series patterns
- **Ensemble Approach**: Model comparison and selection based on performance

### 💼 Business Simulation
- **Production Strategies**: Compare baseline vs ML-driven production planning
- **Cost Analysis**: Storage, shortage, and waste cost optimization
- **ROI Calculation**: Business impact and investment return analysis
- **What-if Scenarios**: Test different production strategies

### 📈 Interactive Dashboard
- **Real-time Visualization**: Interactive charts and metrics
- **Model Performance**: Compare accuracy across different approaches
- **Business Insights**: Cost savings and waste reduction metrics
- **Recommendations**: Actionable business recommendations

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.10+**: Main programming language
- **Pandas/NumPy**: Data manipulation and analysis
- **SQLite**: Database for data storage
- **Streamlit**: Interactive dashboard framework

### Web Scraping
- **Requests**: HTTP library for API calls
- **BeautifulSoup4**: HTML parsing
- **Selenium**: Dynamic content scraping (if needed)

### Machine Learning
- **Prophet**: Time series forecasting
- **XGBoost**: Gradient boosting framework
- **TensorFlow/Keras**: Deep learning for LSTM models
- **Scikit-learn**: ML utilities and metrics

### Visualization
- **Plotly**: Interactive charts and graphs
- **Matplotlib/Seaborn**: Statistical visualizations

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd agro_demand_forecasting
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup directories**
```bash
python main.py --mode pipeline
```

## 🎮 Usage

### Quick Start

1. **Launch the dashboard**
```bash
python main.py --mode dashboard
```

2. **Or run the full pipeline**
```bash
python main.py --mode full
```

### Dashboard Features

The Streamlit dashboard provides five main sections:

1. **📊 Data Overview**: Explore collected data and quality metrics
2. **🤖 Model Training**: Train and compare ML models
3. **📈 Forecasts**: View demand predictions and confidence intervals
4. **💼 Business Simulation**: Compare production strategies
5. **📋 Recommendations**: Get actionable business insights

### Command Line Options

```bash
# Run data processing pipeline only
python main.py --mode pipeline

# Run interactive dashboard only
python main.py --mode dashboard

# Run full pipeline then dashboard
python main.py --mode full
```

## 📊 Methodology

### Data Collection Strategy

1. **Product Data**: Daily scraping of retail websites for:
   - Product prices and promotions
   - Stock availability
   - Category information
   - Retailer information

2. **Weather Data**: Historical weather parameters:
   - Temperature and precipitation
   - Humidity and wind speed
   - Seasonal indicators

### Feature Engineering

The system creates 50+ features including:

- **Temporal Features**: Day of week, month, season, holidays
- **Lag Features**: Historical demand patterns
- **Rolling Statistics**: Moving averages and trends
- **Weather Interactions**: Temperature-demand relationships
- **Price Features**: Volatility and trend indicators
- **Promotion Impact**: Discount effectiveness metrics

### Model Evaluation

Models are evaluated using:
- **MAPE** (Mean Absolute Percentage Error)
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **Business Metrics**: Service level, waste reduction, cost savings

## 📈 Business Impact

### Key Performance Indicators

| Metric | Target | Current | Improvement |
|--------|--------|---------|-------------|
| Forecast Accuracy | >90% | 85% | +5% |
| Service Level | >95% | 92% | +3% |
| Waste Reduction | >15% | 12% | +3% |
| Cost Reduction | >10% | 8% | +2% |

### ROI Analysis

The system provides comprehensive ROI analysis including:
- Implementation costs
- Annual savings projections
- Payback period calculation
- Net Present Value (NPV)

## 🔧 Configuration

### Database Settings

Edit `src/config/settings.py` to configure:
- Database connection parameters
- Scraping targets and delays
- Model hyperparameters
- Business simulation parameters

### Scraping Configuration

```python
SCRAPING_CONFIG = {
    "headers": {"User-Agent": "..."},
    "delay_between_requests": 1.0,
    "timeout": 30,
    "max_retries": 3
}
```

### Model Parameters

```python
MODEL_CONFIG = {
    "prophet": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "changepoint_prior_scale": 0.05
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    }
}
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Data Quality Tests
```bash
python -m pytest tests/data_quality/
```

## 📝 Project Structure

### Core Modules

- **`src/data/scrapers/`**: Web scraping implementation
- **`src/data/pipeline/`**: Data cleaning and feature engineering
- **`src/models/`**: ML model implementations
- **`src/business/`**: Business simulation and ROI analysis
- **`src/visualization/`**: Streamlit dashboard

### Data Flow

1. **Collection**: Web scraping → Raw data storage
2. **Processing**: Cleaning → Feature engineering → Database storage
3. **Modeling**: Training → Validation → Forecast generation
4. **Simulation**: Strategy comparison → Business impact analysis
5. **Visualization**: Dashboard → Reports → Recommendations

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run src/visualization/dashboard.py
```

### Production Deployment

#### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/visualization/dashboard.py"]
```

#### Cloud Deployment
- **Heroku**: Easy deployment with PostgreSQL
- **AWS**: EC2 instance with RDS database
- **Google Cloud**: Cloud Run with Cloud SQL

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `python -m pytest`
5. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions
- Include unit tests for new features

## 📚 Documentation

### API Documentation
- **Data Scraping**: `src/data/scrapers/README.md`
- **ML Models**: `src/models/README.md`
- **Business Logic**: `src/business/README.md`

### Examples
- **Basic Usage**: `examples/basic_usage.py`
- **Custom Models**: `examples/custom_models.py`
- **Advanced Features**: `examples/advanced_features.py`

## 🐛 Troubleshooting

### Common Issues

1. **TensorFlow Import Error**
   ```bash
   pip install tensorflow==2.11.0
   ```

2. **Scraping Rate Limits**
   - Increase delay in `settings.py`
   - Use rotating proxies

3. **Memory Issues**
   - Reduce batch size in model training
   - Use data chunking for large datasets

### Logging

Check logs in `data/logs/agro_forecasting.log` for detailed error information.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Data Scientist**: Lead ML development and modeling
- **Data Engineer**: Data pipeline and infrastructure
- **Business Analyst**: Requirements and ROI analysis
- **Full Stack Developer**: Dashboard and deployment

## 📞 Support

For questions and support:
- Create an issue in the repository
- Email: [your-email@domain.com]
- Documentation: [link-to-docs]

## 🗺️ Roadmap

### Version 2.0 Features
- [ ] Real-time API integration
- [ ] Advanced ensemble models
- [ ] Mobile dashboard
- [ ] Multi-language support

### Future Enhancements
- [ ] Supply chain optimization
- [ ] Dynamic pricing recommendations
- [ ] Integration with ERP systems
- [ ] Advanced anomaly detection

---

**Built with ❤️ for sustainable agriculture and food waste reduction**
