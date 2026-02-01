"""
Advanced Visualizations Module for Agro Demand Forecasting
Visualisations graphiques avancées et interactives
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import math
import random


class AdvancedVisualizations:
    """Classe pour les visualisations avancées"""
    
    def __init__(self, products_df, weather_df):
        self.products_df = products_df
        self.weather_df = weather_df
    
    def render_3d_visualizations(self):
        """Visualisations 3D interactives"""
        st.header("🎨 Visualisations 3D Avancées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_3d_price_surface()
        
        with col2:
            self.render_3d_demand_cloud()
    
    def render_3d_price_surface(self):
        """Graphique 3D de surface des prix"""
        st.subheader("📊 Surface 3D des Prix")
        
        # Générer une grille de données pour la surface 3D
        if 'current_price' in self.products_df.columns:
            # Créer une matrice de prix par catégorie et retailer
            pivot_data = self.products_df.pivot_table(
                values='current_price',
                index='category' if 'category' in self.products_df.columns else self.products_df.index[:10],
                columns='retailer' if 'retailer' in self.products_df.columns else ['Retailer'],
                aggfunc='mean'
            ).fillna(0)
            
            # Créer le graphique 3D
            fig = go.Figure(data=[go.Surface(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Prix Moyen (€)")
            )])
            
            fig.update_layout(
                title="Surface 3D des Prix par Catégorie et Retailer",
                scene=dict(
                    xaxis_title="Retailer",
                    yaxis_title="Catégorie",
                    zaxis_title="Prix (€)"
                ),
                width=500,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Données de prix non disponibles")
    
    def render_3d_demand_cloud(self):
        """Nuage de points 3D de la demande"""
        st.subheader("☁️ Nuage 3D de la Demande")
        
        # Simuler des données 3D si nécessaire
        n_points = min(100, len(self.products_df))
        
        x = np.random.uniform(0, 10, n_points)  # Prix
        y = np.random.uniform(0, 10, n_points)  # Stock
        z = np.random.uniform(0, 1, n_points)   # Demande
        
        # Ajouter des couleurs basées sur la catégorie
        colors = np.random.choice(['red', 'blue', 'green', 'orange'], n_points)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                opacity=0.8
            ),
            text=[f'Point {i}' for i in range(n_points)],
            hovertemplate='<b>%{text}</b><br>Prix: %{x:.2f}<br>Stock: %{y:.2f}<br>Demande: %{z:.2f}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Nuage 3D Prix-Stock-Demande",
            scene=dict(
                xaxis_title="Prix (€)",
                yaxis_title="Stock Level",
                zaxis_title="Demande"
            ),
            width=500,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_animated_charts(self):
        """Graphiques animés"""
        st.header("🎬 Graphiques Animés")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_animated_price_evolution()
        
        with col2:
            self.render_animated_demand_forecast()
    
    def render_animated_price_evolution(self):
        """Animation de l'évolution des prix"""
        st.subheader("📈 Évolution Animée des Prix")
        
        # Créer des données temporelles pour l'animation
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
        
        fig = go.Figure()
        
        # Ajouter des traces pour différentes catégories
        categories = ['fruits_legumes', 'produits_laitiers', 'viandes', 'cereales']
        colors = ['blue', 'red', 'green', 'orange']
        
        for cat, color in zip(categories, colors):
            # Simuler l'évolution des prix
            prices = []
            for date in dates:
                base_price = random.uniform(2, 10)
                seasonal_factor = 1.0 + 0.3 * math.sin(2 * math.pi * date.month / 12)
                price = base_price * seasonal_factor
                prices.append(price)
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines+markers',
                name=cat.replace('_', ' ').title(),
                line=dict(color=color, width=2),
                visible=True
            ))
        
        # Ajouter des frames pour l'animation
        frames = []
        for i, date in enumerate(dates):
            frame_data = []
            for j, cat in enumerate(categories):
                frame_data.append(go.Scatter(
                    x=dates[:i+1],
                    y=[random.uniform(2, 10) * (1.0 + 0.3 * math.sin(2 * math.pi * d.month / 12)) for d in dates[:i+1]],
                    mode='lines+markers',
                    name=cat.replace('_', ' ').title(),
                    line=dict(color=colors[j], width=2)
                ))
            frames.append(go.Frame(data=frame_data, name=str(date)))
        
        fig.frames = frames
        
        # Ajouter des contrôles d'animation
        fig.update_layout(
            title="Évolution des Prix par Catégorie",
            xaxis_title="Date",
            yaxis_title="Prix (€)",
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {"label": "▶️ Play", "method": "animate", "args": [None]},
                        {"label": "⏸️ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
                    ]
                }
            ],
            sliders=[
                {
                    "steps": [{"args": [[frame.name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], "label": frame.name} for frame in frames],
                    "active": 0,
                    "currentvalue": {"prefix": "Mois: "},
                    "len": len(frames),
                    "x": 0.1,
                    "len": 0.9,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top"
                }
            ],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_animated_demand_forecast(self):
        """Animation des prévisions de demande"""
        st.subheader("🔮 Prévisions Animées de Demande")
        
        # Créer des données de prévision
        dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
        
        fig = go.Figure()
        
        # Prévision avec intervalles de confiance
        forecast = []
        upper_bound = []
        lower_bound = []
        
        for i, date in enumerate(dates):
            base_demand = 0.6 + 0.2 * math.sin(2 * math.pi * i / 30)
            noise = random.uniform(-0.05, 0.05)
            demand = base_demand + noise
            
            forecast.append(demand)
            upper_bound.append(min(1.0, demand + 0.1))
            lower_bound.append(max(0.0, demand - 0.1))
        
        # Tracer la prévision
        fig.add_trace(go.Scatter(
            x=dates,
            y=forecast,
            mode='lines+markers',
            name='Prévision',
            line=dict(color='blue', width=2)
        ))
        
        # Tracer les intervalles de confiance
        fig.add_trace(go.Scatter(
            x=dates,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,255,0.2)',
            name='Intervalle de Confiance'
        ))
        
        # Animation
        frames = []
        for i in range(1, len(dates) + 1):
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=dates[:i], y=forecast[:i], mode='lines+markers', name='Prévision'),
                    go.Scatter(x=dates[:i], y=upper_bound[:i], mode='lines', line=dict(width=0)),
                    go.Scatter(x=dates[:i], y=lower_bound[:i], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,255,0.2)')
                ],
                name=str(i)
            ))
        
        fig.frames = frames
        
        fig.update_layout(
            title="Prévisions de Demande sur 30 jours",
            xaxis_title="Date",
            yaxis_title="Demande",
            height=400,
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {"label": "▶️ Play", "method": "animate", "args": [None]},
                        {"label": "⏸️ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
                    ]
                }
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_heatmaps(self):
        """Cartes de chaleur avancées"""
        st.header("🔥 Cartes de Chaleur Interactives")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_price_heatmap()
        
        with col2:
            self.render_correlation_heatmap()
    
    def render_price_heatmap(self):
        """Carte de chaleur des prix"""
        st.subheader("💰 Carte de Chaleur des Prix")
        
        if 'current_price' in self.products_df.columns:
            # Créer une matrice de prix
            if 'category' in self.products_df.columns and 'retailer' in self.products_df.columns:
                pivot_data = self.products_df.pivot_table(
                    values='current_price',
                    index='category',
                    columns='retailer',
                    aggfunc='mean'
                )
            else:
                # Simuler des données
                categories = ['Fruits', 'Légumes', 'Laitiers', 'Viandes', 'Céréales']
                retailers = ['Carrefour', 'Auchan', 'Leclerc', 'Intermarché']
                pivot_data = pd.DataFrame(
                    np.random.uniform(2, 15, (len(categories), len(retailers))),
                    index=categories,
                    columns=retailers
                )
            
            fig = px.imshow(
                pivot_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Viridis",
                title="Carte de Chaleur des Prix Moyens"
            )
            
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Données de prix non disponibles")
    
    def render_correlation_heatmap(self):
        """Carte de chaleur des corrélations"""
        st.subheader("📊 Matrice de Corrélation")
        
        # Créer des données simulées pour la corrélation
        data = {
            'Prix': np.random.uniform(2, 15, 100),
            'Demande': np.random.uniform(0.3, 1.0, 100),
            'Stock': np.random.uniform(50, 500, 100),
            'Promotion': np.random.uniform(0, 1, 100),
            'Température': np.random.uniform(-5, 35, 100)
        }
        
        df_corr = pd.DataFrame(data)
        correlation_matrix = df_corr.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Matrice de Corrélation des Variables"
        )
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_advanced_charts(self):
        """Graphiques avancés supplémentaires"""
        st.header("📈 Graphiques Avancés")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_sankey_diagram()
        
        with col2:
            self.render_treemap()
    
    def render_sankey_diagram(self):
        """Diagramme de Sankey pour les flux"""
        st.subheader("🔄 Diagramme de Sankey")
        
        # Créer un diagramme de Sankey pour les flux de produits
        sources = ['Fournisseurs', 'Fournisseurs', 'Fournisseurs', 'Entrepôts', 'Entrepôts', 'Magasins', 'Magasins']
        targets = ['Entrepôts', 'Entrepôts', 'Entrepôts', 'Magasins', 'Magasins', 'Clients', 'Clients']
        values = [120, 80, 150, 100, 150, 80, 120]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Fournisseurs", "Entrepôts", "Magasins", "Clients"]
            ),
            link=dict(
                source=[0, 0, 0, 1, 1, 2, 2],  # indices correspond to labels
                target=[1, 1, 1, 2, 2, 3, 3],
                value=values
            )
        )])
        
        fig.update_layout(
            title="Flux de la Chaîne d'Approvisionnement",
            font_size=10,
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_treemap(self):
        """Treemap pour la répartition des produits"""
        st.subheader("🌳 Treemap des Produits")
        
        # Créer des données pour le treemap
        categories = ['Fruits', 'Légumes', 'Laitiers', 'Viandes', 'Céréales']
        subcategories = [
            ['Pommes', 'Bananes', 'Oranges'],
            ['Carottes', 'Tomates', 'Salades'],
            ['Lait', 'Yaourts', 'Fromages'],
            ['Poulet', 'Bœuf', 'Porc'],
            ['Pain', 'Pâtes', 'Riz']
        ]
        values = [
            [30, 25, 20],
            [35, 40, 25],
            [45, 30, 35],
            [50, 60, 40],
            [25, 30, 20]
        ]
        
        # Aplatir les données pour le treemap
        labels = []
        parents = []
        values_flat = []
        
        for i, (cat, subs, vals) in enumerate(zip(categories, subcategories, values)):
            labels.append(cat)
            parents.append("")
            values_flat.append(sum(vals))
            
            for sub, val in zip(subs, vals):
                labels.append(sub)
                parents.append(cat)
                values_flat.append(val)
        
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values_flat,
            branchvalues="total",
            textinfo="label+value+percent parent",
            hovertemplate='<b>%{label}</b><br>Valeur: %{value}<br>%{percentParent:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Répartition des Produits par Catégorie",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_gauge_charts(self):
        """Graphiques de jauge (gauge charts)"""
        st.header("🎯 Tableau de Bord avec Jauges")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.create_gauge("Taux de Service", 92, 0, 100, "green")
        
        with col2:
            self.create_gauge("Taux de Gaspillage", 8, 0, 100, "red")
        
        with col3:
            self.create_gauge("Précision Modèle", 87, 0, 100, "blue")
        
        with col4:
            self.create_gauge("ROI", 156, 0, 200, "orange")
    
    def create_gauge(self, title, value, min_val, max_val, color):
        """Créer un graphique de jauge"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 80 if "Service" in title else 10 if "Gaspillage" in title else 85},
            gauge = {
                'axis': {'range': [None, max_val]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, min_val], 'color': "lightgray"},
                    {'range': [min_val, max_val], 'color': "lightblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    def render_polar_charts(self):
        """Graphiques polaires"""
        st.header("🌐 Graphiques Polaires")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_polar_area()
        
        with col2:
            self.render_polar_bar()
    
    def render_polar_area(self):
        """Graphique en aires polaires"""
        st.subheader("🎨 Graphique en Aires Polaires")
        
        categories = ['Prix', 'Qualité', 'Disponibilité', 'Promotion', 'Service']
        
        fig = go.Figure()
        
        # Ajouter plusieurs retailers
        retailers = ['Carrefour', 'Auchan', 'Leclerc']
        colors = ['blue', 'red', 'green']
        
        for retailer, color in zip(retailers, colors):
            values = [random.uniform(60, 95) for _ in categories]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=retailer,
                line_color=color
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Comparaison des Retailers (Polaire)",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_polar_bar(self):
        """Graphique en barres polaires"""
        st.subheader("📊 Graphique en Barres Polaires")
        
        categories = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        values = [random.uniform(50, 100) for _ in categories]
        
        fig = go.Figure(data=[
            go.Barpolar(
                r=values,
                theta=categories,
                marker_color=values,
                marker_colorscale='Viridis',
                marker_colorbar=dict(title="Niveau de Demande")
            )
        ])
        
        fig.update_layout(
            title="Demande Hebdomadaire (Polaire)",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_all_visualizations(self):
        """Afficher toutes les visualisations avancées"""
        # 3D Visualizations
        self.render_3d_visualizations()
        
        st.markdown("---")
        
        # Animated Charts
        self.render_animated_charts()
        
        st.markdown("---")
        
        # Heatmaps
        self.render_heatmaps()
        
        st.markdown("---")
        
        # Advanced Charts
        self.render_advanced_charts()
        
        st.markdown("---")
        
        # Gauge Charts
        self.render_gauge_charts()
        
        st.markdown("---")
        
        # Polar Charts
        self.render_polar_charts()


def main():
    """Fonction de test"""
    # Générer des données de test
    products_df = pd.DataFrame({
        'current_price': np.random.uniform(2, 15, 100),
        'category': np.random.choice(['fruits_legumes', 'produits_laitiers', 'viandes', 'cereales'], 100),
        'retailer': np.random.choice(['Carrefour', 'Auchan', 'Leclerc', 'Intermarché'], 100),
        'demand': np.random.uniform(0.3, 1.0, 100),
        'stock_level': np.random.randint(50, 500, 100)
    })
    
    weather_df = pd.DataFrame({
        'temperature_2m': np.random.uniform(-5, 35, 100),
        'precipitation': np.random.uniform(0, 10, 100),
        'humidity': np.random.uniform(30, 100, 100)
    })
    
    # Créer les visualisations
    viz = AdvancedVisualizations(products_df, weather_df)
    viz.render_all_visualizations()


if __name__ == "__main__":
    main()
