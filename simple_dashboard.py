"""
Simple Streamlit Dashboard for Agro Demand Forecasting
Version simplifiée sans imports complexes
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import math

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


def generate_sample_data():
    """Générer des données d'exemple"""
    # Produits
    categories = ['fruits_legumes', 'produits_laitiers', 'viandes', 'cereales']
    products = []
    
    base_date = datetime.now() - timedelta(days=365)
    
    for day in range(365):
        current_date = base_date + timedelta(days=day)
        
        for category in categories:
            # Prix avec saisonnalité
            base_price = random.uniform(2.0, 8.0)
            seasonal_factor = 1.0 + 0.3 * math.sin(2 * math.pi * day / 365)
            price = base_price * seasonal_factor * random.uniform(0.9, 1.1)
            
            # Promotion
            is_promo = random.random() < 0.2
            promo_percentage = random.uniform(5, 30) if is_promo else 0
            
            # Disponibilité
            is_available = random.random() < 0.9
            
            products.append({
                'date': current_date.date(),
                'category': category,
                'current_price': round(price, 2),
                'is_promotion': is_promo,
                'promotion_percentage': promo_percentage,
                'is_available': is_available,
                'availability_rate': random.uniform(0.7, 1.0)
            })
    
    products_df = pd.DataFrame(products)
    
    # Météo
    weather_data = []
    for day in range(365):
        current_date = base_date + timedelta(days=day)
        day_of_year = current_date.timetuple().tm_yday
        
        # Température réaliste
        base_temp = 15
        seasonal_amplitude = 10
        temp = base_temp + seasonal_amplitude * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        temp += random.uniform(-5, 5)
        
        # Précipitation
        precip_prob = 0.3 if 300 <= day_of_year <= 60 else 0.2
        precipitation = random.uniform(0, 10) if random.random() < precip_prob else 0
        
        weather_data.append({
            'date': current_date.date(),
            'temperature_2m': round(temp, 1),
            'precipitation': round(precipitation, 1),
            'humidity': round(80 - (temp - 10) * 1.5 + random.uniform(-10, 10), 1),
            'wind_speed': round(random.uniform(5, 25), 1)
        })
    
    weather_df = pd.DataFrame(weather_data)
    
    return products_df, weather_df


def main():
    """Fonction principale du dashboard"""
    st.markdown('<h1 class="main-header">🌾 Agro Demand Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page",
        ["📊 Vue d'ensemble", "📈 Prévisions", "💼 Simulation Business", "📋 Recommandations"]
    )
    
    # Générer les données
    if 'data_loaded' not in st.session_state:
        with st.spinner("Génération des données d'exemple..."):
            products_df, weather_df = generate_sample_data()
            st.session_state.products_data = products_df
            st.session_state.weather_data = weather_df
            st.session_state.data_loaded = True
    
    products_df = st.session_state.products_data
    weather_df = st.session_state.weather_data
    
    # Afficher la page sélectionnée
    if page == "📊 Vue d'ensemble":
        show_data_overview(products_df, weather_df)
    elif page == "📈 Prévisions":
        show_forecasts(products_df, weather_df)
    elif page == "💼 Simulation Business":
        show_business_simulation(products_df)
    elif page == "📋 Recommandations":
        show_recommendations()


def show_data_overview(products_df, weather_df):
    """Page de vue d'ensemble des données"""
    st.header("📊 Vue d'ensemble des données")
    
    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Produits", len(products_df))
    
    with col2:
        st.metric("Catégories", products_df['category'].nunique())
    
    with col3:
        st.metric("Records Météo", len(weather_df))
    
    with col4:
        st.metric("Prix Moyen", f"{products_df['current_price'].mean():.2f}€")
    
    # Visualisations
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution par catégorie
        category_counts = products_df['category'].value_counts()
        fig = px.pie(values=category_counts.values, names=category_counts.index, 
                    title="Produits par Catégorie")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tendance des prix
        price_trend = products_df.groupby('date')['current_price'].mean().reset_index()
        fig = px.line(price_trend, x='date', y='current_price', 
                     title="Tendance des Prix Moyens")
        st.plotly_chart(fig, use_container_width=True)
    
    # Météo
    st.subheader("🌤️ Analyse Météo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tendance température
        temp_trend = weather_df.groupby('date')['temperature_2m'].mean().reset_index()
        fig = px.line(temp_trend, x='date', y='temperature_2m', 
                     title="Tendance des Températures")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution précipitations
        fig = px.histogram(weather_df, x='precipitation', 
                          title="Distribution des Précipitations")
        st.plotly_chart(fig, use_container_width=True)


def show_forecasts(products_df, weather_df):
    """Page de prévisions"""
    st.header("📈 Prévisions de Demande")
    
    # Simuler des prévisions
    st.subheader("🔮 Prévisions sur 30 jours")
    
    # Sélectionner une catégorie
    categories = products_df['category'].unique()
    selected_category = st.selectbox("Sélectionner une catégorie", categories)
    
    # Filtrer les données
    category_data = products_df[products_df['category'] == selected_category]
    
    # Créer des prévisions simulées
    last_date = category_data['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                 periods=30, freq='D')
    
    # Simuler les prévisions avec tendance et saisonnalité
    base_demand = category_data['availability_rate'].mean()
    forecasts = []
    
    for i, date in enumerate(future_dates):
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1.0 + 0.2 * math.sin(2 * math.pi * day_of_year / 365)
        noise = random.uniform(-0.1, 0.1)
        forecast = base_demand * seasonal_factor + noise
        forecasts.append(max(0, min(1, forecast)))
    
    # Créer le graphique
    historical_data = category_data.tail(60)  # 60 derniers jours
    
    fig = go.Figure()
    
    # Données historiques
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['availability_rate'],
        mode='lines',
        name='Demande Historique',
        line=dict(color='blue')
    ))
    
    # Prévisions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecasts,
        mode='lines+markers',
        name='Prévisions',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"Prévisions de Demande - {selected_category}",
        xaxis_title="Date",
        yaxis_title="Taux de Disponibilité"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques des prévisions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Demande Moyenne Prévue", f"{np.mean(forecasts):.2f}")
    
    with col2:
        st.metric("Demande Min Prévue", f"{min(forecasts):.2f}")
    
    with col3:
        st.metric("Demande Max Prévue", f"{max(forecasts):.2f}")


def show_business_simulation(products_df):
    """Page de simulation business"""
    st.header("💼 Simulation Business")
    
    st.subheader("📊 Comparaison des Stratégies")
    
    # Simuler différentes stratégies
    strategies = {
        'Baseline': {
            'service_level': 85,
            'waste_percentage': 12,
            'total_cost': 15000,
            'storage_cost': 5000,
            'shortage_cost': 7000,
            'waste_cost': 3000
        },
        'ML Forecast': {
            'service_level': 92,
            'waste_percentage': 8,
            'total_cost': 12000,
            'storage_cost': 4000,
            'shortage_cost': 5000,
            'waste_cost': 3000
        },
        'Adaptive': {
            'service_level': 95,
            'waste_percentage': 6,
            'total_cost': 11000,
            'storage_cost': 3500,
            'shortage_cost': 4000,
            'waste_cost': 3500
        }
    }
    
    # Tableau de comparaison
    comparison_data = []
    for strategy, metrics in strategies.items():
        comparison_data.append({
            'Stratégie': strategy,
            'Service Level (%)': metrics['service_level'],
            'Waste (%)': metrics['waste_percentage'],
            'Total Cost (€)': metrics['total_cost'],
            'Storage Cost (€)': metrics['storage_cost'],
            'Shortage Cost (€)': metrics['shortage_cost'],
            'Waste Cost (€)': metrics['waste_cost']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualisations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.bar(comparison_df, x='Stratégie', y='Service Level (%)',
                    title="Service Level")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(comparison_df, x='Stratégie', y='Waste (%)',
                    title="Pourcentage de Gaspillage")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.bar(comparison_df, x='Stratégie', y='Total Cost (€)',
                    title="Coût Total")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommandation
    best_strategy = min(strategies.items(), key=lambda x: x[1]['total_cost'])
    
    st.markdown(f"""
    <div class="insight-box">
        <h3>🎯 Stratégie Recommandée: {best_strategy[0]}</h3>
        <p><strong>Coût Total:</strong> €{best_strategy[1]['total_cost']:,}</p>
        <p><strong>Service Level:</strong> {best_strategy[1]['service_level']}%</p>
        <p><strong>Réduction des Coûts:</strong> €{strategies['Baseline']['total_cost'] - best_strategy[1]['total_cost']:,} vs Baseline</p>
    </div>
    """, unsafe_allow_html=True)


def show_recommendations():
    """Page de recommandations"""
    st.header("📋 Recommandations Business")
    
    st.subheader("💡 Recommandations Clés")
    
    recommendations = [
        "✅ **Adopter la stratégie ML Forecast** pour une réduction des coûts de 20%",
        "✅ **Implémenter un monitoring continu** de la précision des prévisions",
        "✅ **Ajuster les paramètres de production** basés sur les performances saisonnières",
        "✅ **Considérer les facteurs météo** dans la planification de la production",
        "✅ **Optimiser les niveaux de stock** pour réduire les coûts de stockage",
        "✅ **Développer des stratégies de pricing dynamique** pour gérer la demande",
        "✅ **Mettre en place des alertes** pour les prévisions de forte demande",
        "✅ **Collaborer avec les fournisseurs** pour améliorer la chaîne d'approvisionnement"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="metric-card">
            {rec}
        </div>
        """, unsafe_allow_html=True)
    
    # ROI Analysis
    st.subheader("💰 Analyse ROI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        implementation_cost = st.number_input("Coût d'Implémentation (€)", value=10000, min_value=0)
    
    with col2:
        annual_savings = st.number_input("Économies Annuelles (€)", value=5000, min_value=0)
    
    if st.button("Calculer ROI"):
        # Calculs ROI simples
        payback_period = implementation_cost / annual_savings if annual_savings > 0 else float('inf')
        roi_percentage = ((annual_savings * 3 - implementation_cost) / implementation_cost) * 100 if implementation_cost > 0 else 0
        
        st.markdown(f"""
        <div class="insight-box">
            <h3>📊 Résultats ROI</h3>
            <p><strong>Période de Recouvrement:</strong> {payback_period:.1f} années</p>
            <p><strong>ROI sur 3 ans:</strong> {roi_percentage:.1f}%</p>
            <p><strong>Économies Totales (3 ans):</strong> €{annual_savings * 3:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feuille de route
    st.subheader("🗺️ Feuille de Route d'Implémentation")
    
    roadmap_data = [
        {"Phase": "Phase 1", "Durée": "1-2 mois", "Activités": "Collecte et nettoyage des données", "Priorité": "Haute"},
        {"Phase": "Phase 2", "Durée": "2-3 mois", "Activités": "Développement des modèles ML", "Priorité": "Haute"},
        {"Phase": "Phase 3", "Durée": "1 mois", "Activités": "Intégration système et tests", "Priorité": "Moyenne"},
        {"Phase": "Phase 4", "Durée": "Continu", "Activités": "Monitoring et optimisation", "Priorité": "Moyenne"}
    ]
    
    roadmap_df = pd.DataFrame(roadmap_data)
    st.dataframe(roadmap_df, use_container_width=True)


if __name__ == "__main__":
    main()
