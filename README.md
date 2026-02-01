# 🌾 Agro Demand Forecasting - Expert Dashboard

## 📋 Vue d'ensemble

Un projet complet de **prévision de la demande agroalimentaire** avec dashboard interactif ultra-avancé. Ce système combine scraping web en temps réel, modèles de machine learning, visualisations 3D/animées et simulation business pour optimiser la production et réduire le gaspillage alimentaire.

### 🎯 Objectifs Business

- **Anticiper la demande** des produits agroalimentaires avec haute précision
- **Réduction du gaspillage** de 15-25% par optimisation de la production
- **Dashboard interactif** en temps réel avec mise à jour hebdomadaire automatique
- **Visualisations avancées** : graphiques 3D, animations, jauges, cartes de chaleur
- **ROI mesurable** à travers simulation business et analyse d'impact

### 🏗️ Architecture du Projet

```
agro_demand_forecasting/
├── 📊 interactive_dashboard.py     # Dashboard principal ultra-interactif
├── 🎨 advanced_visualizations.py   # Visualisations 3D et animations
├── ⏰ realtime_weekly_data.py      # Gestionnaire de données temps réel
├── 🔧 robust_scraper.py            # Scraping robuste avec fallbacks
├── 📈 simple_dashboard.py          # Dashboard simplifié
├── 🧪 test_real_scraping.py        # Tests de scraping
├── src/
│   ├── config/                     # Configuration et paramètres
│   ├── data/                       # Pipeline de données et scraping
│   ├── models/                     # Modèles ML (Prophet, XGBoost, LSTM)
│   ├── business/                   # Simulation business et ROI
│   └── visualization/              # Dashboard Streamlit classique
├── data/                          # Stockage des données
├── notebooks/                     # Analyse exploratoire
├── requirements.txt               # Dépendances Python
└── setup.py                      # Configuration du package
```

## 🚀 Fonctionnalités Principales

### 📊 Dashboard Interactif Expert

**Interface principale : `interactive_dashboard.py`**
- **🎨 Visualisations 3D** : Surfaces de prix, nuages de points 3D
- **🎬 Graphiques animés** : Évolution temporelle avec animations
- **🔥 Cartes de chaleur** : Matrices de corrélation interactives
- **📊 Graphiques avancés** : Diagrammes Sankey, Treemaps
- **🎯 Tableau de bord** : Jauges et indicateurs de performance
- **🌐 Graphiques polaires** : Analyses circulaires et radiales

### ⏰ Données en Temps Réel

**Gestionnaire : `realtime_weekly_data.py`**
- **🔄 Mise à jour automatique** hebdomadaire
- **💾 Cache intelligent** de 7 jours
- **🌡️ Données météo** réelles (Open-Meteo API)
- **🛒 Données produits** (OpenFoodFacts API)
- **📈 Tendances hebdomadaires** avec analyse temporelle
- **🔄 Système de fallback** robuste

### 🔧 Scraping Robuste

**Moteur : `robust_scraper.py`**
- **🌐 Multi-sources** : OpenFoodFacts, Carrefour, APIs externes
- **🔄 Gestion d'erreurs** avancée avec retry automatique
- **📊 Données simulées** réalistes en fallback
- **⚡ Performance** optimisée avec cache
- **🛡️ Anti-détection** avec rotation d'user-agents

### 🤖 Modèles Machine Learning

**Implémentations dans `src/models/`**
- **📈 Prophet** : Prévisions temporelles de base
- **🚀 XGBoost** : Gradient boosting avec features externes
- **🧠 LSTM** : Deep learning pour patterns complexes
- **🎯 Ensemble** : Combinaison optimisée des modèles
- **📊 Métriques** : MAPE, RMSE, MAE avec validation croisée

### 💼 Simulation Business

**Module : `src/business/simulation.py`**
- **📊 Stratégies de production** comparatives
- **💰 Analyse des coûts** : stockage, pénurie, gaspillage
- **📈 Calcul ROI** avec projections financières
- **🎮 Scénarios What-if** : tests de stratégies
- **📋 Recommandations** business actionnables

## 🛠️ Stack Technologique

### Technologies Principales
- **Python 3.10+** : Langage principal
- **Streamlit** : Dashboard interactif
- **Plotly** : Visualisations 3D et animations
- **Pandas/NumPy** : Manipulation de données
- **Requests** : Appels API et scraping

### Machine Learning
- **Prophet** : Prévisions temporelles Facebook
- **XGBoost** : Gradient boosting
- **TensorFlow/Keras** : LSTM et deep learning
- **Scikit-learn** : Utilitaires ML

### Visualisations Avancées
- **Plotly Graph Objects** : Graphiques 3D personnalisés
- **Plotly Subplots** : Multi-graphiques
- **Plotly Figure Factory** : Visualisations complexes
- **Animations** : Transitions temporelles



### Instructions d'Installation

1. **Cloner le repository**
```bash
git clone https://github.com/oumar1958/prevision-de-la-demande-agroalimentaire.git
cd prevision-de-la-demande-agroalimentaire
```

2. **Créer l'environnement virtuel**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 🎮 Utilisation

### Lancement Rapide

1. **Dashboard Expert Interactif**
```bash
streamlit run interactive_dashboard.py
```

2. **Dashboard Simplifié**
```bash
streamlit run simple_dashboard.py
```

3. **Pipeline Complet**
```bash
python main.py --mode pipeline
```

### Fonctionnalités du Dashboard Expert

Le dashboard principal offre **7 sections interactives** :

1. **📊 Tendances Hebdomadaires** : Analyse temporelle avec mise à jour auto
2. **📈 Métriques Temps Réel** : Indicateurs dynamiques et filtres
3. **📊 Graphiques Interactifs** : Tendances, prévisions, comparaisons
4. **🎨 Visualisations Avancées** : 5 onglets de graphiques sophistiqués
5. **🎮 Simulation Business** : Scénarios et analyse ROI
6. **🤖 Insights IA** : Recommandations intelligentes
7. **🔄 Contrôles Interactifs** : Filtres, sliders, sélections


### Visualisations Avancées

#### Graphiques 3D
- **Surfaces de prix** : Visualisation multi-dimensionnelle
- **Nuages 3D** : Distribution prix-stock-demande

#### Animations
- **Évolution temporelle** : Transitions fluides
- **Prévisions animées** : Progression des prédictions

#### Cartes de Chaleur
- **Matrices de corrélation** : Relations entre variables
- **Cartes de prix** : Distribution par catégorie/retailer

#### Graphiques Spécialisés
- **Diagrammes Sankey** : Flux de la chaîne d'approvisionnement
- **Treemaps** : Répartition hiérarchique des produits
- **Jauges** : Indicateurs de performance en temps réel
- **Graphiques polaires** : Analyses radiales


## 🤝 Contributeur
-
**Oumar Abdramane ALLAWAN**
-




