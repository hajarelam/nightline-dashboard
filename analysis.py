import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from utils import find_closest_column
from report_generator import generate_detailed_report
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import traceback
import json

def check_password():
    # Temporairement retourner True pour désactiver l'authentification
    return True
    
    # Code original commenté
    """
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    return True
    """

def load_and_clean_data(sheet_id='1VJ0JaagpbXXKZrgiMRcvtoebo89ZgM2kdjcSEBhrANA'):
    try:
        # Vérifier les secrets silencieusement
        if "gcp" not in st.secrets or "service_account" not in st.secrets["gcp"]:
            st.error("Configuration des secrets manquante")
            return pd.DataFrame()
            
        # Récupérer et vérifier les credentials
        credentials_dict = st.secrets["gcp"]["service_account"]
        
        # Si c'est une chaîne, la convertir en dictionnaire
        if isinstance(credentials_dict, str):
            try:
                credentials_dict = json.loads(credentials_dict)
            except json.JSONDecodeError as e:
                st.error(f"Erreur lors du décodage JSON: {str(e)}")
                return pd.DataFrame()
        
        # Créer les credentials et se connecter
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            credentials_dict,
            ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
        )
        
        client = gspread.authorize(credentials)
        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet('Cleaned')
        
        # Lire les données
        all_values = worksheet.get_all_values()
        headers = all_values[0]
        
        # Créer des en-têtes uniques
        unique_headers = []
        seen = {}
        for h in headers:
            if h in seen:
                seen[h] += 1
                unique_headers.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                unique_headers.append(h)
        
        # Créer le DataFrame avec les en-têtes uniques
        df = pd.DataFrame(all_values[1:], columns=unique_headers)
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        return pd.DataFrame()

def calculate_engagement_kpis(df):
    # Print column names to debug
    print("Available columns:")
    for col in df.columns:
        print(f"- {col}")
    
    # 1. Usage & Engagement KPIs
    total_calls = len(df)
    
    # Find the first-time column using partial matching
    first_time_column = find_closest_column(df, "première fois")
    if not first_time_column:
        print("Warning: Could not find first-time callers column!")
        return {
            'total_calls': total_calls,
            'first_time_percentage': 0,
            'returning_percentage': 0,
            'call_method': pd.Series(),
            'call_timing': pd.Series(),
            'language_distribution': pd.Series(),
            'awareness_sources': pd.Series()
        }
    
    # Find and analyze other columns
    call_timing_col = find_closest_column(df, "Quand a eu lieu")
    language_col = find_closest_column(df, "Tu parles")
    awareness_col = find_closest_column(df, "Comment as-tu entendu")
    call_method_col = find_closest_column(df, "appel s'est déroulé")
    
    # Get statistics
    call_timing = df[call_timing_col].value_counts() if call_timing_col else pd.Series()
    language_dist = df[language_col].value_counts() if language_col else pd.Series()
    awareness_source = df[awareness_col].value_counts() if awareness_col else pd.Series()
    call_method = df[call_method_col].value_counts() if call_method_col else pd.Series()
    
    # Calculate first-time vs returning callers
    if first_time_column:
        # Compter uniquement les réponses valides
        valid_responses = df[df[first_time_column].notna()]
        
        # Compter les réponses directement
        responses = valid_responses[first_time_column].value_counts()
        
        # Calculer les pourcentages
        total_responses = responses.sum()
        
        if total_responses > 0:
            # Corriger les pourcentages pour correspondre au graphique
            returning_pct = (responses.get("Non j'avais déjà appelé", 0) / total_responses) * 100  # 30.6%
            first_time_pct = (responses.get("Oui c'était la première fois", 0) / total_responses) * 100  # 69.4%
            
            # Debug pour vérifier les valeurs exactes
            print("Debug - Réponses brutes:")
            print(responses)
            print(f"Total réponses: {total_responses}")
            print(f"Premiers appels (Oui): {responses.get('Oui c\'était la première fois', 0)} ({first_time_pct:.1f}%)")
            print(f"Appelants réguliers (Non): {responses.get('Non j\'avais déjà appelé', 0)} ({returning_pct:.1f}%)")
        else:
            first_time_pct = 0
            returning_pct = 0
    else:
        first_time_pct = 0
        returning_pct = 0
    
    return {
        'total_calls': total_calls,
        'first_time_percentage': first_time_pct,
        'returning_percentage': returning_pct,
        'call_method': call_method,
        'call_timing': call_timing,
        'language_distribution': language_dist,
        'awareness_sources': awareness_source
    }

def analyze_call_experience(df):
    # Find relevant columns
    comfort_col = find_closest_column(df, "l'aise pour aborder")
    understood_col = find_closest_column(df, "Compris")
    supported_col = find_closest_column(df, "Soutenu")
    lonely_col = find_closest_column(df, "Moins seul")
    
    # Calculate experience metrics with proper handling of "Ne s'applique pas"
    def calculate_percentage(column):
        if column is None:
            return 0
        # Filter out "Ne s'applique pas" responses
        valid_responses = df[~df[column].str.contains("ne s'applique pas", na=False, case=False)]
        if len(valid_responses) == 0:
            return 0
        # Count positive responses (Oui)
        positive = valid_responses[column].str.contains('Oui', na=False, case=False).sum()
        return (positive / len(valid_responses)) * 100

    experience_metrics = {
        'comfort': calculate_percentage(comfort_col),
        'understood': calculate_percentage(understood_col),
        'supported': calculate_percentage(supported_col),
        'less_lonely': calculate_percentage(lonely_col)
    }
    
    # Analyze future intentions
    resources_col = find_closest_column(df, "Consulter des ressources")
    talk_family_col = find_closest_column(df, "Parler avec des proches")
    professional_col = find_closest_column(df, "professionnel de santé")
    
    future_intentions = {
        'seek_resources': calculate_percentage(resources_col),
        'talk_to_family': calculate_percentage(talk_family_col),
        'seek_professional': calculate_percentage(professional_col)
    }
    
    return {
        'experience_metrics': experience_metrics,
        'future_intentions': future_intentions
    }

def analyze_wellbeing_impact(df):
    # Find relevant columns using partial matching
    before_col = find_closest_column(df, "AVANT d'appeler")
    after_col = find_closest_column(df, "APRÈS après l'appel")
    
    # Find experience columns
    comfort_col = find_closest_column(df, "l'aise pour aborder")
    understood_col = find_closest_column(df, "Compris")
    supported_col = find_closest_column(df, "Soutenu")
    lonely_col = find_closest_column(df, "Moins seul")
    
    # Analyze before/after states
    before_after = pd.DataFrame({
        'before': df[before_col],
        'after': df[after_col]
    })
    
    # Calculate detailed metrics
    improved = (before_after['after'] > before_after['before']).sum()
    worsened = (before_after['after'] < before_after['before']).sum()
    same = (before_after['after'] == before_after['before']).sum()
    
    improvement_percentage = (improved / len(df)) * 100
    worsening_percentage = (worsened / len(df)) * 100
    
    # Analyze feeling states
    before_states = df[before_col].value_counts()
    after_states = df[after_col].value_counts()
    
    # Analyze call experience with case-insensitive matching
    def get_positive_percentage(series):
        positive_responses = series.str.lower().isin(['oui', 'yes'])
        return (positive_responses.sum() / len(series)) * 100
    
    experience_metrics = {
        'comfort': get_positive_percentage(df[comfort_col]) if comfort_col else 0,
        'understood': get_positive_percentage(df[understood_col]) if understood_col else 0,
        'supported': get_positive_percentage(df[supported_col]) if supported_col else 0,
        'less_lonely': get_positive_percentage(df[lonely_col]) if lonely_col else 0
    }
    
    # Analyze future intentions with case-insensitive matching
    resources_col = find_closest_column(df, "Consulter des ressources")
    talk_family_col = find_closest_column(df, "Parler avec des proches")
    talk_professional_col = find_closest_column(df, "professionnel de santé")
    
    future_intentions = {
        'seek_resources': get_positive_percentage(df[resources_col]) if resources_col else 0,
        'talk_to_family': get_positive_percentage(df[talk_family_col]) if talk_family_col else 0,
        'seek_professional': get_positive_percentage(df[talk_professional_col]) if talk_professional_col else 0
    }
    
    return {
        'before_after': before_after,
        'improvement_percentage': improvement_percentage,
        'worsening_percentage': worsening_percentage,
        'before_states': before_states,
        'after_states': after_states,
        'experience_metrics': experience_metrics,
        'future_intentions': future_intentions
    }

def analyze_reasons_for_calling(df):
    # Analyze reasons for calling
    reasons = df["Qu'est-ce que tu cherchais en appelant la ligne d'écoute Nightline ?"].str.split(',').explode().str.strip()
    reasons_count = reasons.value_counts()
    
    # Calculate percentage based on number of respondents
    total_respondents = len(df)
    reasons_percentage = (reasons_count / total_respondents) * 100
    
    # Take only top 8 reasons
    top_8_reasons = reasons_percentage.head(8)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_8_reasons.values,
        y=top_8_reasons.index,
        orientation='h',
        marker_color='#4ecdc4',
        text=[f'{x:.1f}%' for x in top_8_reasons.values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Top 8 des raisons d'appel",
        xaxis_title="Pourcentage des appelants",
        yaxis_title=None,  # Remove y-axis title
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        yaxis=dict(
            autorange="reversed",
            gridcolor='rgba(128,128,128,0.1)',  # Very subtle grid
            tickfont=dict(size=12)  # Adjust font size
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.1)',  # Very subtle grid
            zeroline=False
        ),
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="white"
        ),
        title_font_size=20
    )

    # Add percentage labels on the bars
    fig.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        hovertemplate='%{y}<br>%{x:.1f}%<extra></extra>'
    )
    
    return fig, top_8_reasons

def analyze_demographics(df):
    # Afficher toutes les colonnes disponibles pour le debug
    print("Colonnes disponibles pour l'analyse démographique:")
    print(df.columns.tolist())
    
    # Trouver les colonnes avec find_closest_column
    age_col = find_closest_column(df, "âge")
    gender_col = find_closest_column(df, "genre")
    academic_col = find_closest_column(df, "année d'étude")
    institution_col = find_closest_column(df, "établissement")
    
    # Vérifier si les colonnes existent
    if not all([age_col, gender_col, academic_col, institution_col]):
        st.warning("Certaines colonnes démographiques sont manquantes")
        return {
            'age_distribution': pd.Series(),
            'gender_distribution': pd.Series(),
            'academic_level': pd.Series(),
            'institution_type': pd.Series(),
            'age_gender_distribution': pd.DataFrame(),
            'academic_by_institution': pd.DataFrame()
        }
    
    # Calculer les distributions
    age_dist = df[age_col].value_counts() if age_col else pd.Series()
    gender_dist = df[gender_col].value_counts(normalize=True) * 100 if gender_col else pd.Series()
    academic_level = df[academic_col].value_counts() if academic_col else pd.Series()
    institution_type = df[institution_col].value_counts() if institution_col else pd.Series()
    
    # Crosstabs
    age_gender_dist = pd.crosstab(df[age_col], df[gender_col]) if all([age_col, gender_col]) else pd.DataFrame()
    academic_by_institution = pd.crosstab(df[academic_col], df[institution_col]) if all([academic_col, institution_col]) else pd.DataFrame()
    
    return {
        'age_distribution': age_dist,
        'gender_distribution': gender_dist,
        'academic_level': academic_level,
        'institution_type': institution_type,
        'age_gender_distribution': age_gender_dist,
        'academic_by_institution': academic_by_institution
    }

def analyze_nl_awareness_sources(df):
    # Trouver la colonne des sources
    source_col = find_closest_column(df, "Comment as-tu entendu")
    
    if source_col:
        # Grouper les sources en catégories plus larges
        source_mapping = {
            'Réseaux sociaux': ['Instagram', 'Facebook', 'Twitter', 'LinkedIn', 'TikTok', 'Snapchat'],
            'Site web': ['Site internet', 'Google', 'Recherche web'],
            'Institution': ['École', 'Université', 'Administration', 'CROUS', 'Service de santé universitaire'],
            'Bouche à oreille': ['Ami·e·s', 'Famille', 'Camarades'],
            'Professionnels': ['Psychologue', 'Médecin', 'Thérapeute', 'Infirmier·e'],
            'Communication physique': ['Affiches', 'Flyers', 'Événement', 'Stand'],
            'Autres': ['Autre']
        }
        
        # Nettoyer et catégoriser les données
        sources = df[source_col].str.split(',').explode().str.strip()
        
        # Mapper les sources aux catégories
        categorized_sources = []
        for source in sources:
            if pd.notna(source):
                found = False
                for category, keywords in source_mapping.items():
                    if any(keyword.lower() in source.lower() for keyword in keywords):
                        categorized_sources.append(category)
                        found = True
                        break
                if not found:
                    categorized_sources.append('Autres')
        
        # Créer le DataFrame des sources catégorisées
        source_counts = pd.Series(categorized_sources).value_counts()
        source_percentages = (source_counts / len(df)) * 100
        
        # Créer la visualisation
        fig = go.Figure()
        
        # Ajouter le graphique en barres
        fig.add_trace(go.Bar(
            x=source_percentages.values,
            y=source_percentages.index,
            orientation='h',
            marker_color='#4ecdc4',
            text=[f'{x:.1f}%' for x in source_percentages.values],
            textposition='auto',
        ))
        
        # Mise en page
        fig.update_layout(
            title="Comment les appelants ont découvert Nightline",
            xaxis_title="Pourcentage des appelants",
            yaxis_title="Source",
            template='plotly_dark',
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#1e1e1e',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        
        return fig, source_percentages
    
    return None, None

def create_satisfaction_evolution_chart(df):
    # Trouver la colonne de satisfaction
    satisfaction_col = find_closest_column(df, "satisfaction globale")
    date_col = find_closest_column(df, "Date")
    
    if satisfaction_col and date_col:
        # Convertir les dates en format datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Grouper par semaine au lieu de mois pour plus de détails
        weekly_data = df.groupby(df[date_col].dt.strftime('%Y-%m-%d')).agg({
            satisfaction_col: lambda x: {
                'Positive': (x == 'Très satisfait·e').mean() * 100,
                'Neutre': (x == 'Neutre').mean() * 100,
                'Negative': (x == 'Pas satisfait·e').mean() * 100
            }
        }).apply(pd.Series)
        
        # Créer le graphique
        fig = go.Figure()
        
        # Ajouter les lignes avec un style plus clair
        fig.add_trace(go.Scatter(
            x=weekly_data.index,
            y=weekly_data['Positive'],
            name='Satisfait·e',
            line=dict(color='#00C853', width=2),
            mode='lines+markers',
            marker=dict(size=6, symbol='circle')
        ))
        
        fig.add_trace(go.Scatter(
            x=weekly_data.index,
            y=weekly_data['Neutre'],
            name='Neutre',
            line=dict(color='#FFB300', width=2),
            mode='lines+markers',
            marker=dict(size=6, symbol='circle')
        ))
        
        fig.add_trace(go.Scatter(
            x=weekly_data.index,
            y=weekly_data['Negative'],
            name='Pas satisfait·e',
            line=dict(color='#FF1744', width=2),
            mode='lines+markers',
            marker=dict(size=6, symbol='circle')
        ))
        
        # Mise en page améliorée
        fig.update_layout(
            title={
                'text': 'Évolution de la satisfaction',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18)
            },
            xaxis_title="Date",
            yaxis_title="Pourcentage (%)",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            width=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            yaxis=dict(
                gridcolor='rgba(128,128,128,0.2)',
                range=[0, 100],
                ticksuffix='%',
                tickfont=dict(size=10),
                zeroline=False
            ),
            xaxis=dict(
                gridcolor='rgba(128,128,128,0.2)',
                tickangle=45,
                tickfont=dict(size=10),
                zeroline=False,
                nticks=10  # Limiter le nombre de dates affichées
            ),
            hovermode='x unified'
        )
        
        # Améliorer le hover
        fig.update_traces(
            hovertemplate='<b>%{y:.1f}%</b><extra>%{fullData.name}</extra>'
        )
        
        return fig
    
    return None

def create_streamlit_dashboard():
    # Set page config
    st.set_page_config(
        page_title="Nightline Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )

    # Add custom CSS
    st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 28px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Unique title at the top
    st.title("📊 Nightline Analytics Dashboard")
    
    # Load data avec le bon chemin
    sheet_id = '1VJ0JaagpbXXKZrgiMRcvtoebo89ZgM2kdjcSEBhrANA'
    df = load_and_clean_data(sheet_id)
    
    # Trouver la colonne de date et la convertir en datetime
    date_col = find_closest_column(df, "Date")
    if date_col:
        # Convertir la colonne date en datetime
        df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
        
        # Créer la colonne month
        df['month'] = df[date_col].dt.to_period('M')
    
    # Ajouter les filtres dans la sidebar avant la navigation
    st.sidebar.title("Filtres")
    
    # Filtre de type d'appel
    call_method_col = find_closest_column(df, "appel s'est déroulé")
    if call_method_col:
        st.sidebar.subheader("Type d'appel")
        call_methods = ['Tous'] + list(df[call_method_col].unique())
        selected_method = st.sidebar.selectbox(
            "Sélectionner le type d'appel",
            options=call_methods,
            index=0  # 'Tous' par défaut
        )
        
        # Appliquer le filtre de type d'appel
        if selected_method != 'Tous':
            df = df[df[call_method_col] == selected_method]
    
    # Filtre de date
    if date_col:
        st.sidebar.subheader("Période")
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        date_range = st.sidebar.date_input(
            "Sélectionner la période",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Appliquer le filtre de date
        if len(date_range) == 2:
            df = df[(df[date_col].dt.date >= date_range[0]) & 
                   (df[date_col].dt.date <= date_range[1])]
    
    # Filtre de satisfaction
    satisfaction_col = find_closest_column(df, "satisfaction globale")
    if satisfaction_col:
        st.sidebar.subheader("Niveau de satisfaction")
        satisfaction_options = ['Tous'] + list(df[satisfaction_col].unique())
        selected_satisfaction = st.sidebar.multiselect(
            "Sélectionner le niveau de satisfaction",
            options=satisfaction_options,
            default=['Tous']
        )
        
        if 'Tous' not in selected_satisfaction:
            df = df[df[satisfaction_col].isin(selected_satisfaction)]
    
    # Séparateur avant la navigation
    st.sidebar.markdown("---")
    
    # Navigation existante
    page = st.sidebar.selectbox(
        "Navigation",
        ["Vue d'ensemble", "Expérience Appelants", "Démographie", "Raisons des Appels", "Sources et Communication"]
    )
    
    # Calculate KPIs
    engagement_kpis = calculate_engagement_kpis(df)
    experience_kpis = analyze_call_experience(df)
    demographic_kpis = analyze_demographics(df) if 'analyze_demographics' in globals() else None
    
    if page == "Vue d'ensemble":
        # Display metrics directly
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nombre total d'appels", engagement_kpis['total_calls'])
        
        with col2:
            st.metric("Premiers appels", f"{engagement_kpis['first_time_percentage']:.1f}%")
        
        with col3:
            st.metric("Appelants réguliers", f"{engagement_kpis['returning_percentage']:.1f}%")
        
        with col4:
            # Calculer la satisfaction moyenne
            comfort = experience_kpis['experience_metrics'].get('comfort', 0)
            understood = experience_kpis['experience_metrics'].get('understood', 0)
            supported = experience_kpis['experience_metrics'].get('supported', 0)
            satisfaction = (comfort + understood + supported) / 3
            
            st.metric(
                "Satisfaction globale", 
                f"{satisfaction:.1f}%",
                help="Moyenne du pourcentage d'appelants qui se sont sentis à l'aise, compris et soutenus"
            )
        
        # Call Method Distribution
        fig = px.pie(
            values=engagement_kpis['call_method'].values,
            names=engagement_kpis['call_method'].index,
            title="Distribution des Méthodes d'Appel"
        )
        st.plotly_chart(fig)
        
        # Satisfaction by Method
        fig = go.Figure()
        satisfaction_data = experience_kpis['experience_metrics']
        for col in satisfaction_data.keys():
            fig.add_trace(go.Bar(
                name=col,
                x=[col],
                y=[satisfaction_data[col]],
            ))
        fig.update_layout(
            barmode='stack',
            title="Satisfaction par Méthode d'Appel",
            xaxis_title="Méthode d'Appel",
            yaxis_title="Pourcentage"
        )
        st.plotly_chart(fig)
        
        # Add time series analysis
        if date_col:
            st.subheader("Evolution temporelle")
            df['month'] = df[date_col].dt.to_period('M')
            monthly_calls = df.groupby('month').size()
            
            fig = px.line(
                x=monthly_calls.index.astype(str),
                y=monthly_calls.values,
                title="Evolution du nombre d'appels par mois",
                labels={'x': 'Mois', 'y': 'Nombre d\'appels'}
            )
            st.plotly_chart(fig)
        
        # Add interactive heatmap for call timing
        st.subheader("Distribution horaire des appels")
        df['hour'] = df[date_col].dt.hour
        df['day_of_week'] = df[date_col].dt.day_name()
        
        call_heatmap = pd.crosstab(df['day_of_week'], df['hour'])
        fig = px.imshow(
            call_heatmap,
            title="Distribution des appels par jour et heure",
            labels=dict(x="Heure", y="Jour", color="Nombre d'appels")
        )
        st.plotly_chart(fig)
        
        # Remove "Évolution de la satisfaction" subheader
        satisfaction_fig = create_satisfaction_evolution_chart(df)
        if satisfaction_fig:
            st.plotly_chart(satisfaction_fig, use_container_width=True)
            
            # Keep the explanatory legend
            st.markdown("""
            **Comment lire ce graphique :**
            - La ligne verte montre le pourcentage d'appelants satisfaits
            - La ligne jaune montre le pourcentage d'appelants neutres
            - La ligne rouge montre le pourcentage d'appelants insatisfaits
            - Chaque point représente la moyenne mensuelle
            """)
        
    elif page == "Expérience Appelants":
        st.header("Expérience des Appelants")
        
        # Experience Metrics
        exp_metrics = experience_kpis['experience_metrics']
        fig = px.bar(
            x=['À l\'aise', 'Compris·e', 'Soutenu·e', 'Moins seul·e'],
            y=[exp_metrics['comfort'], exp_metrics['understood'], 
               exp_metrics['supported'], exp_metrics['less_lonely']],
            title="Métriques d'Expérience",
            labels={'x': 'Métrique', 'y': 'Pourcentage de réponses positives'}
        )
        st.plotly_chart(fig)
        
        # Future Intentions
        intentions = experience_kpis['future_intentions']
        fig = px.bar(
            x=['Consulter des ressources', 'Parler aux proches', 'Voir un professionnel'],
            y=[intentions['seek_resources'], intentions['talk_to_family'], 
               intentions['seek_professional']],
            title="Intentions Futures",
            labels={'x': 'Action', 'y': 'Pourcentage'}
        )
        st.plotly_chart(fig)
        
        # Add satisfaction trend over time
        st.subheader("Evolution de la satisfaction")
        sentiment_col = find_closest_column(df, "sentiment général")
        if sentiment_col and date_col:
            # Convert to datetime and group by month
            df['month'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m')
            satisfaction_trend = df.groupby('month')[sentiment_col].value_counts(normalize=True).unstack()
            
            # Reorder columns and combine positive/negative categories
            satisfaction_trend['Positive'] = satisfaction_trend[['Excellent', 'Bon']].sum(axis=1)
            satisfaction_trend['Neutre'] = satisfaction_trend['Moyen']
            satisfaction_trend['Negative'] = satisfaction_trend[['Mauvais', 'Très mauvais']].sum(axis=1)
            
            # Create a more professional line chart with dark theme
            fig = go.Figure()
            
            # Professional color palette for dark theme
            colors = {
                'Positive': '#00CC96',    # Bright teal
                'Neutre': '#FFB266',      # Soft orange
                'Negative': '#EF553B'      # Coral red
            }
            
            # Add traces with professional styling
            for category in ['Positive', 'Neutre', 'Negative']:
                fig.add_trace(go.Scatter(
                    x=satisfaction_trend.index,
                    y=satisfaction_trend[category] * 100,
                    name=category,
                    mode='lines+markers',
                    line=dict(
                        width=2,
                        color=colors[category],
                        shape='linear'
                    ),
                    marker=dict(
                        size=6,
                        color=colors[category],
                        line=dict(width=1, color='#1e1e1e')
                    ),
                    hovertemplate="<b>" + category + "</b><br>Date: %{x}<br>Taux: %{y:.1f}%<extra></extra>"
                ))
            
            fig.update_layout(
                template='plotly_dark',
                title={
                    'text': "Evolution de la satisfaction au fil du temps",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=16, color='#ffffff')
                },
                xaxis_title="Mois",
                yaxis_title="Pourcentage",
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor='rgba(30,30,30,0.8)',
                    bordercolor='rgba(255,255,255,0.3)',
                    borderwidth=1,
                    font=dict(size=10, color='#ffffff')
                ),
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#1e1e1e',
                yaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    tickformat='.0%',
                    range=[0, 100],
                    tickmode='linear',
                    tick0=0,
                    dtick=20,
                    tickfont=dict(size=10, color='#ffffff'),
                    title_font=dict(size=12, color='#ffffff'),
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                xaxis=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    tickangle=45,
                    tickmode='auto',
                    nticks=8,  # Reduced number of ticks
                    tickfont=dict(size=10, color='#ffffff'),
                    title_font=dict(size=12, color='#ffffff'),
                    zerolinecolor='rgba(255,255,255,0.2)'
                ),
                margin=dict(t=50, l=50, r=20, b=50),
                height=400,  # Reduced height
                width=800    # Fixed width
            )
            
            # Add subtle grid lines
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            
            st.plotly_chart(fig, use_container_width=False)  # Disabled container width to maintain fixed size
            
            # Keep the existing summary statistics
            st.subheader("Résumé de la satisfaction")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                positive = satisfaction_trend['Positive'].mean() * 100
                st.metric("Satisfaction positive", f"{positive:.1f}%",
                         help="Pourcentage moyen de réponses 'Excellent' ou 'Bon'")
            
            with col2:
                neutral = satisfaction_trend['Neutre'].mean() * 100
                st.metric("Satisfaction neutre", f"{neutral:.1f}%",
                         help="Pourcentage moyen de réponses 'Moyen'")
            
            with col3:
                negative = satisfaction_trend['Negative'].mean() * 100
                st.metric("Satisfaction négative", f"{negative:.1f}%",
                         help="Pourcentage moyen de réponses 'Mauvais' ou 'Très mauvais'")
        
    elif page == "Démographie":
        st.header("Analyse Démographique")
        
        # Age Distribution
        fig = px.bar(
            x=demographic_kpis['age_distribution'].index,
            y=demographic_kpis['age_distribution'].values,
            title="Distribution des Âges"
        )
        st.plotly_chart(fig)
        
        # Gender Distribution
        fig = px.pie(
            values=demographic_kpis['gender_distribution'].values,
            names=demographic_kpis['gender_distribution'].index,
            title="Distribution des Genres"
        )
        st.plotly_chart(fig)
        
        # Age-Gender Heatmap
        fig = px.imshow(
            demographic_kpis['age_gender_distribution'],
            title="Distribution Age-Genre",
            labels=dict(x="Genre", y="Âge", color="Nombre d'appelants")
        )
        st.plotly_chart(fig)
        
        # Add map visualization
        st.subheader("Distribution géographique")
        region_col = "Dans quelle région es-tu étudiant ?"
        if region_col in df.columns:
            region_counts = df[region_col].dropna().value_counts()
            if not region_counts.empty:
                try:
                    fig = px.choropleth(
                        locations=region_counts.index,
                        locationmode="country names",
                        color=region_counts.values,
                        scope="europe",
                        title="Distribution des appelants par région"
                    )
                    st.plotly_chart(fig)
                except Exception as e:
                    st.warning(f"Impossible de générer la carte: {str(e)}")
                    # Fallback to bar chart
                    fig = px.bar(
                        x=region_counts.index,
                        y=region_counts.values,
                        title="Distribution par région"
                    )
                    st.plotly_chart(fig)
        
        # Add sunburst chart for academic distribution
        st.subheader("Répartition académique")
        
        # Clean the data first
        establishment_col = "Dans quel type d'établissement es-tu étudiant·e ?"
        year_col = "En quelle année d'étude es-tu inscrit·e cette année ?"
        
        # Remove rows with missing values
        academic_data = df[[establishment_col, year_col]].dropna()
        
        # Clean string values
        academic_data[establishment_col] = academic_data[establishment_col].str.strip()
        academic_data[year_col] = academic_data[year_col].str.strip()
        
        if not academic_data.empty:
            fig = px.sunburst(
                academic_data,
                path=[establishment_col, year_col],
                title="Répartition par type d'établissement et niveau d'études"
            )
            st.plotly_chart(fig)
        else:
            st.warning("Pas assez de données pour générer le graphique de répartition académique")
        
    elif page == "Raisons des Appels":
        st.header("Raisons des Appels")
        
        # Find the column for reasons
        reasons_col = "Qu'est-ce que tu cherchais en appelant la ligne d'écoute Nightline ?"
        if reasons_col in df.columns:
            # Clean and process the reasons
            reasons = df[reasons_col].str.split(',').explode().str.strip()
            reasons_count = reasons.value_counts()
            
            # Calculate percentage based on number of respondents
            total_respondents = len(df)
            reasons_percentage = (reasons_count / total_respondents) * 100
            
            # Take only top 8 reasons
            top_8_reasons = reasons_percentage.head(8)
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=top_8_reasons.values,
                y=top_8_reasons.index,
                orientation='h',
                marker_color='#4ecdc4',
                text=[f'{x:.1f}%' for x in top_8_reasons.values],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Top 8 des raisons d'appel",
                xaxis_title="Pourcentage des appelants",
                yaxis_title=None,  # Remove y-axis title
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                height=600,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                yaxis=dict(
                    autorange="reversed",
                    gridcolor='rgba(128,128,128,0.1)',  # Very subtle grid
                    tickfont=dict(size=12)  # Adjust font size
                ),
                xaxis=dict(
                    gridcolor='rgba(128,128,128,0.1)',  # Very subtle grid
                    zeroline=False
                ),
                font=dict(
                    family="Arial, sans-serif",
                    size=14,
                    color="white"
                ),
                title_font_size=20
            )

            # Add percentage labels on the bars
            fig.update_traces(
                texttemplate='%{text}',
                textposition='outside',
                hovertemplate='%{y}<br>%{x:.1f}%<extra></extra>'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Style the metrics container
            st.markdown("""
                <style>
                [data-testid="stMetricValue"] {
                    font-size: 24px;
                }
                [data-testid="stMetricLabel"] {
                    font-size: 16px;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Add metrics with better formatting
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Raisons principales affichées",
                    f"Top {len(top_8_reasons)}",
                    help="Nombre de raisons les plus fréquemment mentionnées affichées dans le graphique"
                )
            
            with col2:
                avg_reasons_per_caller = len(reasons) / total_respondents
                st.metric(
                    "Moyenne de raisons par appelant",
                    f"{avg_reasons_per_caller:.1f}",
                    help="En moyenne, chaque appelant mentionne ce nombre de raisons"
                )
        else:
            st.warning("Données sur les raisons d'appel non disponibles")
        
    elif page == "Sources et Communication":
        st.header("Sources d'Information et Communication")
        
        # Afficher le graphique des sources
        source_fig, source_data = analyze_nl_awareness_sources(df)
        if source_fig:
            st.plotly_chart(source_fig)
        
        # Option pour générer le rapport détaillé
        if st.button("Générer un rapport détaillé"):
            # Calculer toutes les métriques nécessaires
            wellbeing_analysis = analyze_wellbeing_impact(df)
            call_exp = analyze_call_experience(df)
            
            report = generate_detailed_report(
                df,
                wellbeing_analysis,
                call_exp,  # Utiliser les nouvelles métriques
                source_data
            )
            
            # Permettre le téléchargement du rapport
            st.download_button(
                label="Télécharger le rapport (HTML)",
                data=report,
                file_name=f"rapport_nightline_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html"
            )

    # Add download button in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Télécharger les données")
    
    # Prepare filtered data for download
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(df)
    st.sidebar.download_button(
        label="Télécharger les données filtrées (CSV)",
        data=csv,
        file_name='nightline_data_filtered.csv',
        mime='text/csv',
    )

def main():
    if check_password():
        # Your existing app code here
        create_streamlit_dashboard()

if __name__ == "__main__":
    main() 
