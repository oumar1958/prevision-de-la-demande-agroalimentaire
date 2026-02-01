"""
Business simulation module for evaluating the impact of demand forecasting
on production planning, waste reduction, and cost optimization
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.settings import BUSINESS_CONFIG
from src.data.database import db_manager

logger = logging.getLogger(__name__)


class ProductionSimulator:
    """
    Simulates production planning strategies and evaluates business impact
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or BUSINESS_CONFIG
        self.storage_cost = self.config.get("storage_cost_per_unit", 0.05)
        self.shortage_cost = self.config.get("shortage_cost_per_unit", 2.0)
        self.waste_cost = self.config.get("waste_cost_per_unit", 1.5)
        self.capacity_multiplier = self.config.get("production_capacity_multiplier", 1.2)
        
    def simulate_baseline_strategy(self, historical_data: pd.DataFrame, 
                                 target_col: str = 'demand') -> Dict:
        """
        Simulate baseline production strategy based on historical averages
        
        Args:
            historical_data: Historical demand data
            target_col: Target demand column
            
        Returns:
            Simulation results
        """
        logger.info("Simulating baseline production strategy")
        
        # Calculate historical average demand
        avg_demand = historical_data[target_col].mean()
        std_demand = historical_data[target_col].std()
        
        # Baseline strategy: produce average demand + safety stock
        safety_stock = std_demand * 0.5  # 0.5 standard deviations
        baseline_production = avg_demand + safety_stock
        
        # Simulate daily production and inventory
        results = self._simulate_daily_operations(
            historical_data, 
            baseline_production,
            strategy_name="Baseline"
        )
        
        results['strategy_params'] = {
            'avg_demand': avg_demand,
            'std_demand': std_demand,
            'safety_stock': safety_stock,
            'production_level': baseline_production
        }
        
        return results
    
    def simulate_ml_forecast_strategy(self, historical_data: pd.DataFrame,
                                    forecasts: pd.DataFrame,
                                    target_col: str = 'demand') -> Dict:
        """
        Simulate ML-based production strategy using forecasts
        
        Args:
            historical_data: Historical demand data
            forecasts: Forecast data with predictions
            target_col: Target demand column
            
        Returns:
            Simulation results
        """
        logger.info("Simulating ML forecast-based production strategy")
        
        # Merge historical data with forecasts
        if 'date' in historical_data.columns and 'date' in forecasts.columns:
            simulation_data = historical_data.merge(
                forecasts[['date', 'predicted_demand', 'confidence_lower', 'confidence_upper']],
                on='date',
                how='left'
            )
        else:
            # If no date columns, align by index
            simulation_data = historical_data.copy()
            simulation_data['predicted_demand'] = forecasts['predicted_demand'].values[:len(historical_data)]
            simulation_data['confidence_lower'] = forecasts.get('confidence_lower', forecasts['predicted_demand']).values[:len(historical_data)]
            simulation_data['confidence_upper'] = forecasts.get('confidence_upper', forecasts['predicted_demand']).values[:len(historical_data)]
        
        # Strategy: produce based on forecast with confidence interval adjustment
        simulation_data['production_plan'] = simulation_data['predicted_demand']
        
        # Add safety stock based on forecast uncertainty
        forecast_uncertainty = (simulation_data['confidence_upper'] - simulation_data['confidence_lower']) / 2
        safety_stock_factor = 0.3  # 30% of uncertainty as safety stock
        simulation_data['production_plan'] += forecast_uncertainty * safety_stock_factor
        
        # Simulate daily operations
        results = self._simulate_daily_operations(
            simulation_data,
            simulation_data['production_plan'],
            actual_demand_col=target_col,
            strategy_name="ML Forecast"
        )
        
        results['strategy_params'] = {
            'forecast_uncertainty_avg': forecast_uncertainty.mean(),
            'safety_stock_factor': safety_stock_factor,
            'forecast_mae': np.mean(np.abs(simulation_data[target_col] - simulation_data['predicted_demand']))
        }
        
        return results
    
    def simulate_adaptive_strategy(self, historical_data: pd.DataFrame,
                                 forecasts: pd.DataFrame,
                                 target_col: str = 'demand') -> Dict:
        """
        Simulate adaptive production strategy that adjusts based on recent performance
        
        Args:
            historical_data: Historical demand data
            forecasts: Forecast data
            target_col: Target demand column
            
        Returns:
            Simulation results
        """
        logger.info("Simulating adaptive production strategy")
        
        # Merge data
        if 'date' in historical_data.columns and 'date' in forecasts.columns:
            simulation_data = historical_data.merge(
                forecasts[['date', 'predicted_demand']],
                on='date',
                how='left'
            )
        else:
            simulation_data = historical_data.copy()
            simulation_data['predicted_demand'] = forecasts['predicted_demand'].values[:len(historical_data)]
        
        # Initialize adaptive parameters
        alpha = 0.3  # Learning rate for adaptation
        production_plans = []
        
        for i in range(len(simulation_data)):
            if i == 0:
                # First day: use forecast
                production = simulation_data.loc[i, 'predicted_demand']
            else:
                # Calculate recent forecast error
                recent_error = simulation_data.loc[i-1, target_col] - simulation_data.loc[i-1, 'predicted_demand']
                
                # Adjust production based on recent error
                adjustment = alpha * recent_error
                production = simulation_data.loc[i, 'predicted_demand'] + adjustment
                
                # Add safety stock
                production *= 1.1  # 10% safety buffer
            
            production_plans.append(max(0, production))
        
        simulation_data['production_plan'] = production_plans
        
        # Simulate daily operations
        results = self._simulate_daily_operations(
            simulation_data,
            simulation_data['production_plan'],
            actual_demand_col=target_col,
            strategy_name="Adaptive"
        )
        
        results['strategy_params'] = {
            'learning_rate': alpha,
            'safety_buffer': 0.1
        }
        
        return results
    
    def _simulate_daily_operations(self, data: pd.DataFrame, production_plan: pd.Series,
                                 actual_demand_col: str = 'demand',
                                 strategy_name: str = "Unknown") -> Dict:
        """
        Simulate daily production, inventory, and costs
        
        Args:
            data: Input data
            production_plan: Production plan for each day
            actual_demand_col: Column with actual demand
            strategy_name: Name of the strategy
            
        Returns:
            Simulation results
        """
        if isinstance(production_plan, (int, float)):
            # Constant production level
            production_plan = pd.Series([production_plan] * len(data))
        
        # Initialize tracking variables
        inventory = 0
        total_storage_cost = 0
        total_shortage_cost = 0
        total_waste_cost = 0
        total_waste_units = 0
        total_shortage_units = 0
        
        daily_results = []
        
        for i, row in data.iterrows():
            # Daily production
            daily_production = production_plan.iloc[i] if hasattr(production_plan, 'iloc') else production_plan[i]
            actual_demand = row[actual_demand_col]
            
            # Update inventory
            inventory += daily_production
            
            # Satisfy demand
            if inventory >= actual_demand:
                # Sufficient inventory
                sales = actual_demand
                inventory -= actual_demand
                shortage = 0
            else:
                # Stockout
                sales = inventory
                shortage = actual_demand - inventory
                inventory = 0
            
            # Calculate costs
            # Storage cost (cost per unit per day)
            storage_cost = inventory * self.storage_cost
            total_storage_cost += storage_cost
            
            # Shortage cost
            shortage_cost = shortage * self.shortage_cost
            total_shortage_cost += shortage_cost
            total_shortage_units += shortage
            
            # Waste (perishable goods - assume 3-day shelf life)
            # Simplified: 5% of inventory older than 3 days becomes waste
            if i >= 3:
                waste = inventory * 0.05
                inventory -= waste
                waste_cost = waste * self.waste_cost
                total_waste_cost += waste_cost
                total_waste_units += waste
            else:
                waste = 0
                waste_cost = 0
            
            # Record daily results
            daily_results.append({
                'day': i,
                'production': daily_production,
                'actual_demand': actual_demand,
                'sales': sales,
                'inventory_end': inventory,
                'shortage': shortage,
                'waste': waste,
                'storage_cost': storage_cost,
                'shortage_cost': shortage_cost,
                'waste_cost': waste_cost,
                'total_cost': storage_cost + shortage_cost + waste_cost
            })
        
        # Calculate summary metrics
        total_demand = data[actual_demand_col].sum()
        total_sales = sum(d['sales'] for d in daily_results)
        total_production = sum(d['production'] for d in daily_results)
        
        service_level = (total_sales / total_demand) * 100 if total_demand > 0 else 0
        waste_percentage = (total_waste_units / total_production) * 100 if total_production > 0 else 0
        total_cost = total_storage_cost + total_shortage_cost + total_waste_cost
        
        results = {
            'strategy_name': strategy_name,
            'daily_results': pd.DataFrame(daily_results),
            'summary_metrics': {
                'total_demand': total_demand,
                'total_production': total_production,
                'total_sales': total_sales,
                'service_level': service_level,
                'total_waste_units': total_waste_units,
                'waste_percentage': waste_percentage,
                'total_shortage_units': total_shortage_units,
                'total_storage_cost': total_storage_cost,
                'total_shortage_cost': total_shortage_cost,
                'total_waste_cost': total_waste_cost,
                'total_cost': total_cost,
                'cost_per_unit': total_cost / total_production if total_production > 0 else 0
            }
        }
        
        return results
    
    def compare_strategies(self, historical_data: pd.DataFrame,
                          forecasts: pd.DataFrame = None,
                          target_col: str = 'demand') -> Dict:
        """
        Compare different production strategies
        
        Args:
            historical_data: Historical demand data
            forecasts: Forecast data (for ML strategies)
            target_col: Target demand column
            
        Returns:
            Comparison results
        """
        logger.info("Comparing production strategies")
        
        strategies = {}
        
        # Baseline strategy
        strategies['baseline'] = self.simulate_baseline_strategy(historical_data, target_col)
        
        # ML forecast strategy (if forecasts provided)
        if forecasts is not None:
            strategies['ml_forecast'] = self.simulate_ml_forecast_strategy(
                historical_data, forecasts, target_col
            )
            
            strategies['adaptive'] = self.simulate_adaptive_strategy(
                historical_data, forecasts, target_col
            )
        
        # Create comparison table
        comparison_data = []
        for strategy_name, results in strategies.items():
            metrics = results['summary_metrics']
            comparison_data.append({
                'Strategy': strategy_name,
                'Service Level (%)': round(metrics['service_level'], 2),
                'Waste (%)': round(metrics['waste_percentage'], 2),
                'Total Cost (€)': round(metrics['total_cost'], 2),
                'Cost per Unit (€)': round(metrics['cost_per_unit'], 4),
                'Storage Cost (€)': round(metrics['total_storage_cost'], 2),
                'Shortage Cost (€)': round(metrics['total_shortage_cost'], 2),
                'Waste Cost (€)': round(metrics['total_waste_cost'], 2)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate improvements vs baseline
        if 'baseline' in strategies:
            baseline_metrics = strategies['baseline']['summary_metrics']
            improvements = {}
            
            for strategy_name, results in strategies.items():
                if strategy_name != 'baseline':
                    metrics = results['summary_metrics']
                    improvements[strategy_name] = {
                        'cost_reduction_pct': ((baseline_metrics['total_cost'] - metrics['total_cost']) / baseline_metrics['total_cost']) * 100,
                        'waste_reduction_pct': ((baseline_metrics['waste_percentage'] - metrics['waste_percentage']) / baseline_metrics['waste_percentage']) * 100 if baseline_metrics['waste_percentage'] > 0 else 0,
                        'service_level_improvement': metrics['service_level'] - baseline_metrics['service_level']
                    }
        
        comparison_results = {
            'strategies': strategies,
            'comparison_table': comparison_df,
            'improvements': improvements if 'improvements' in locals() else {},
            'best_strategy': self._find_best_strategy(comparison_df)
        }
        
        return comparison_results
    
    def _find_best_strategy(self, comparison_df: pd.DataFrame) -> Dict:
        """
        Find the best strategy based on multiple criteria
        
        Args:
            comparison_df: Comparison table
            
        Returns:
            Best strategy information
        """
        if len(comparison_df) == 0:
            return {}
        
        # Multi-criteria scoring (lower is better for costs, higher for service level)
        scores = {}
        
        for _, row in comparison_df.iterrows():
            strategy = row['Strategy']
            
            # Normalize scores (0-1 scale)
            cost_score = (row['Total Cost (€)'] - comparison_df['Total Cost (€)'].min()) / (comparison_df['Total Cost (€)'].max() - comparison_df['Total Cost (€)'].min())
            waste_score = (row['Waste (%)'] - comparison_df['Waste (%)'].min()) / (comparison_df['Waste (%)'].max() - comparison_df['Waste (%)'].min())
            service_score = (comparison_df['Service Level (%)'].max() - row['Service Level (%)']) / (comparison_df['Service Level (%)'].max() - comparison_df['Service Level (%)'].min())
            
            # Weighted score (cost: 0.4, waste: 0.3, service: 0.3)
            total_score = 0.4 * cost_score + 0.3 * waste_score + 0.3 * service_score
            scores[strategy] = total_score
        
        best_strategy = min(scores.items(), key=lambda x: x[1])
        
        return {
            'strategy': best_strategy[0],
            'score': best_strategy[1],
            'ranking': sorted(scores.items(), key=lambda x: x[1])
        }
    
    def generate_business_report(self, comparison_results: Dict) -> str:
        """
        Generate a comprehensive business report
        
        Args:
            comparison_results: Results from strategy comparison
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("BUSINESS SIMULATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Strategy comparison table
        report.append("STRATEGY COMPARISON")
        report.append("-" * 40)
        comparison_df = comparison_results['comparison_table']
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Best strategy
        if 'best_strategy' in comparison_results and comparison_results['best_strategy']:
            best = comparison_results['best_strategy']
            report.append("BEST STRATEGY")
            report.append("-" * 40)
            report.append(f"Recommended: {best['strategy']}")
            report.append(f"Composite Score: {best['score']:.4f}")
            report.append("")
        
        # Improvements
        if 'improvements' in comparison_results and comparison_results['improvements']:
            report.append("IMPROVEMENTS VS BASELINE")
            report.append("-" * 40)
            for strategy, improvements in comparison_results['improvements'].items():
                report.append(f"{strategy.upper()}:")
                report.append(f"  Cost Reduction: {improvements['cost_reduction_pct']:.2f}%")
                report.append(f"  Waste Reduction: {improvements['waste_reduction_pct']:.2f}%")
                report.append(f"  Service Level Improvement: {improvements['service_level_improvement']:.2f}%")
                report.append("")
        
        # Key insights
        report.append("KEY INSIGHTS")
        report.append("-" * 40)
        
        if len(comparison_df) > 1:
            best_cost = comparison_df.loc[comparison_df['Total Cost (€)'].idxmin()]
            best_service = comparison_df.loc[comparison_df['Service Level (%)'].idxmax()]
            best_waste = comparison_df.loc[comparison_df['Waste (%)'].idxmin()]
            
            report.append(f"• Lowest Cost: {best_cost['Strategy']} (€{best_cost['Total Cost (€)']:.2f})")
            report.append(f"• Highest Service Level: {best_service['Strategy']} ({best_service['Service Level (%)']:.1f}%)")
            report.append(f"• Lowest Waste: {best_waste['Strategy']} ({best_waste['Waste (%)']:.1f}%)")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_strategy_comparison(self, comparison_results: Dict, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot strategy comparison visualizations
        
        Args:
            comparison_results: Results from strategy comparison
            figsize: Figure size
        """
        comparison_df = comparison_results['comparison_table']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Production Strategy Comparison', fontsize=16)
        
        # Service Level
        axes[0, 0].bar(comparison_df['Strategy'], comparison_df['Service Level (%)'])
        axes[0, 0].set_title('Service Level (%)')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Waste Percentage
        axes[0, 1].bar(comparison_df['Strategy'], comparison_df['Waste (%)'])
        axes[0, 1].set_title('Waste Percentage (%)')
        axes[0, 1].set_ylabel('Percentage')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Total Cost
        axes[1, 0].bar(comparison_df['Strategy'], comparison_df['Total Cost (€)'])
        axes[1, 0].set_title('Total Cost (€)')
        axes[1, 0].set_ylabel('Cost (€)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Cost Breakdown
        cost_cols = ['Storage Cost (€)', 'Shortage Cost (€)', 'Waste Cost (€)']
        comparison_df.set_index('Strategy')[cost_cols].plot(kind='bar', stacked=True, ax=axes[1, 1])
        axes[1, 1].set_title('Cost Breakdown')
        axes[1, 1].set_ylabel('Cost (€)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def save_simulation_results(self, comparison_results: Dict, filepath: str) -> bool:
        """
        Save simulation results to file
        
        Args:
            comparison_results: Results from strategy comparison
            filepath: Path to save results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save comparison table
            comparison_results['comparison_table'].to_csv(filepath.replace('.xlsx', '_comparison.csv'), index=False)
            
            # Save detailed results
            detailed_results = []
            for strategy_name, results in comparison_results['strategies'].items():
                daily_df = results['daily_results']
                daily_df['strategy'] = strategy_name
                detailed_results.append(daily_df)
            
            if detailed_results:
                detailed_df = pd.concat(detailed_results, ignore_index=True)
                detailed_df.to_csv(filepath.replace('.xlsx', '_detailed.csv'), index=False)
            
            # Save report
            report = self.generate_business_report(comparison_results)
            with open(filepath.replace('.xlsx', '_report.txt'), 'w') as f:
                f.write(report)
            
            logger.info(f"Simulation results saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving simulation results: {e}")
            return False


class BusinessImpactAnalyzer:
    """
    Analyzes the business impact of demand forecasting improvements
    """
    
    def __init__(self):
        self.simulator = ProductionSimulator()
    
    def analyze_roi(self, implementation_cost: float, 
                   annual_savings: float, years: int = 3) -> Dict:
        """
        Calculate ROI for implementing forecasting system
        
        Args:
            implementation_cost: One-time implementation cost
            annual_savings: Annual cost savings
            years: Analysis period in years
            
        Returns:
            ROI analysis results
        """
        cash_flows = [-implementation_cost] + [annual_savings] * years
        
        # Calculate NPV (assuming 10% discount rate)
        discount_rate = 0.10
        npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))
        
        # Calculate payback period
        cumulative_cash = 0
        payback_period = None
        for i, cf in enumerate(cash_flows):
            cumulative_cash += cf
            if cumulative_cash >= 0 and payback_period is None:
                payback_period = i
        
        # Calculate ROI
        total_return = sum(cash_flows[1:])  # Exclude initial investment
        roi = (total_return - implementation_cost) / implementation_cost * 100
        
        results = {
            'implementation_cost': implementation_cost,
            'annual_savings': annual_savings,
            'analysis_period_years': years,
            'npv': npv,
            'payback_period_years': payback_period,
            'roi_percentage': roi,
            'cash_flows': cash_flows
        }
        
        return results
    
    def generate_recommendations(self, comparison_results: Dict) -> List[str]:
        """
        Generate business recommendations based on simulation results
        
        Args:
            comparison_results: Results from strategy comparison
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        comparison_df = comparison_results['comparison_table']
        
        # Analyze service levels
        min_service = comparison_df['Service Level (%)'].min()
        if min_service < 95:
            recommendations.append(
                f"Service level is below 95% (minimum: {min_service:.1f}%). "
                "Consider increasing safety stock or improving forecast accuracy."
            )
        
        # Analyze waste
        max_waste = comparison_df['Waste (%)'].max()
        if max_waste > 10:
            recommendations.append(
                f"Waste percentage is high (maximum: {max_waste:.1f}%). "
                "Implement dynamic pricing or demand stimulation strategies."
            )
        
        # Analyze costs
        if len(comparison_df) > 1:
            cost_range = comparison_df['Total Cost (€)'].max() - comparison_df['Total Cost (€)'].min()
            if cost_range > 0:
                best_strategy = comparison_df.loc[comparison_df['Total Cost (€)'].idxmin()]
                recommendations.append(
                    f"Adopt '{best_strategy['Strategy']}' strategy for optimal cost management "
                    f"(potential savings: €{cost_range:.2f})."
                )
        
        # General recommendations
        recommendations.extend([
            "Implement continuous monitoring of forecast accuracy",
            "Regularly update production parameters based on performance",
            "Consider seasonal adjustments for production planning",
            "Monitor external factors (weather, promotions) that impact demand"
        ])
        
        return recommendations
