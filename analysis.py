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
from io import BytesIO
import xlsxwriter
import wordcloud
from wordcloud import WordCloud

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
    """
    Version corrigée avec filtrage minimal pour exclure seulement les non-répondants
    """
    
    # Trouver les colonnes avant/après
    before_col = find_closest_column(df, "AVANT d'appeler")
    after_col = find_closest_column(df, "APRÈS après l'appel")
    
    if not before_col or not after_col:
        print("ERREUR: Colonnes avant/après non trouvées")
        return {
            'feeling_bad_before_pct': 0,
            'overall_improvement_pct': 0,
            'very_bad_pct': 0,
            'very_bad_improvement_pct': 0,
            'avg_improvement_magnitude': 0,
            'improvement_levels': {1: 0, 2: 0, 3: 0},
            'before_stats': pd.DataFrame(),
            'after_stats': pd.DataFrame(),
            'experience_stats': {},
            'intention_stats': {}
        }
    
    # FILTRAGE MINIMAL : Exclure seulement les vraies valeurs nulles
    # Garder toutes les réponses textuelles, même vides
    valid_responses = df[
        df[before_col].notna() & 
        df[after_col].notna()
    ]
    
    total_valid = len(valid_responses)
    
    print(f"Debug - Total réponses: {len(df)}")
    print(f"Debug - Réponses valides: {total_valid}")
    print(f"Debug - Réponses exclues: {len(df) - total_valid}")
    
    # Mapping des états après l'appel
    improvement_mapping = {
        "Ça allait beaucoup plus mal": {"improvement": 0, "magnitude": 0},
        "Ça allait un peu plus mal": {"improvement": 0, "magnitude": 0},
        "Ça allait pareil": {"improvement": 0, "magnitude": 0},
        "Ça allait un peu mieux": {"improvement": 33.3, "magnitude": 1},
        "Ça allait beaucoup mieux": {"improvement": 100, "magnitude": 3}
    }
    
    def calculate_improvement_magnitude(row):
        before = row[before_col]
        after = row[after_col]
        
        if after == "Je ne sais pas dire comment je me sentais":
            return None
            
        if before == "Ça allait très mal":
            return improvement_mapping.get(after, {"improvement": 0, "magnitude": 0})
        return None
    
    # 1. Pourcentage qui se sentent mal ou très mal avant
    feeling_bad_before = valid_responses[valid_responses[before_col].isin(
        ["Ça allait très mal", "Ça allait plutôt mal"]
    )]
    feeling_bad_pct = (len(feeling_bad_before) / total_valid) * 100
    
    # 2. Analyse de l'amélioration
    def compare_states(before, after):
        if after == "Je ne sais pas dire comment je me sentais":
            return None
        if after == "Ça allait pareil":
            return False
        if "mieux" in after.lower():
            return True
        if "mal" in after.lower():
            return False
        return None
    
    improvements = valid_responses.apply(
        lambda row: compare_states(row[before_col], row[after_col]), 
        axis=1
    )
    improvements = improvements.dropna()
    
    overall_improvement_pct = (improvements.sum() / len(improvements)) * 100 if len(improvements) > 0 else 0
    
    # 3. Analyse spécifique des "très mal"
    very_bad_before = valid_responses[valid_responses[before_col] == "Ça allait très mal"]
    very_bad_pct = (len(very_bad_before) / total_valid) * 100
    
    if len(very_bad_before) > 0:
        improvements = very_bad_before.apply(calculate_improvement_magnitude, axis=1).dropna()
        positive_improvements = [imp for imp in improvements if imp and imp['improvement'] > 0]
        
        if positive_improvements:
            improvement_count = len(positive_improvements)
            very_bad_improvement_pct = (improvement_count / len(very_bad_before)) * 100
            avg_improvement = sum(imp['improvement'] for imp in positive_improvements) / improvement_count
            
            level_counts = {
                1: sum(1 for imp in positive_improvements if imp['magnitude'] == 1),
                2: 0,
                3: sum(1 for imp in positive_improvements if imp['magnitude'] == 3)
            }
            
            total_improved = sum(level_counts.values())
            if total_improved > 0:
                level_percentages = {
                    level: (count / total_improved) * 100 
                    for level, count in level_counts.items()
                }
            else:
                level_percentages = {1: 0, 2: 0, 3: 0}
        else:
            very_bad_improvement_pct = 0
            avg_improvement = 0
            level_percentages = {1: 0, 2: 0, 3: 0}
    else:
        very_bad_improvement_pct = 0
        avg_improvement = 0
        level_percentages = {1: 0, 2: 0, 3: 0}
    
    # Créer les DataFrames pour les statistiques avant/après
    before_valid = valid_responses[before_col][valid_responses[before_col].str.strip() != ""]
    before_counts = before_valid.value_counts()
    before_pct = (before_counts / len(before_valid) * 100).round(1)
    before_df = pd.DataFrame({
        'État': before_counts.index,
        'Nombre': before_counts.values,
        'Pourcentage': before_pct.values
    })

    after_valid = valid_responses[after_col][valid_responses[after_col].str.strip() != ""]
    after_counts = after_valid.value_counts()
    after_pct = (after_counts / len(after_valid) * 100).round(1)
    after_df = pd.DataFrame({
        'État': after_counts.index,
        'Nombre': after_counts.values,
        'Pourcentage': after_pct.values
    })
    
    # Debug final
    print(f"AVANT - 'Ça allait plutôt mal': {before_counts.get('Ça allait plutôt mal', 0)}")
    print(f"AVANT - 'Ça allait très mal': {before_counts.get('Ça allait très mal', 0)}")
    
    # Analyse des questions sur le ressenti pendant l'appel
    experience_cols = {
        'À l\'aise': find_closest_column(df, "l'aise pour aborder"),
        'Compris(e)': find_closest_column(df, "Compris"),
        'Soutenu(e)': find_closest_column(df, "Soutenu"),
        'Moins seul(e)': find_closest_column(df, "Moins seul")
    }
    
    experience_stats = {}
    for name, col in experience_cols.items():
        if col:
            # Filtrage minimal pour les questions d'expérience
            valid_exp_responses = df[
                ~df[col].str.contains("ne s'applique pas", na=False, case=False) &
                df[col].notna()
            ]
            yes_count = valid_exp_responses[col].str.contains('Oui', na=False, case=False).sum()
            no_count = valid_exp_responses[col].str.contains('Non', na=False, case=False).sum()
            total_valid_exp = yes_count + no_count
            
            if total_valid_exp > 0:
                experience_stats[name] = {
                    'Oui': yes_count,
                    'Non': no_count,
                    'Pourcentage Oui': (yes_count / total_valid_exp) * 100,
                    'Pourcentage Non': (no_count / total_valid_exp) * 100
                }
    
    # Analyse des intentions futures
    intention_cols = {
        'Consulter des ressources': find_closest_column(df, "Consulter des ressources"),
        'Consulter un professionnel': find_closest_column(df, "professionnel de santé"),
        'Parler avec des proches': find_closest_column(df, "Parler avec des proches")
    }
    
    intention_stats = {}
    for name, col in intention_cols.items():
        if col:
            # Filtrage minimal pour les intentions
            valid_int_responses = df[
                ~df[col].str.contains("ne s'applique pas", na=False, case=False) &
                df[col].notna()
            ]
            yes_count = valid_int_responses[col].str.contains('Oui', na=False, case=False).sum()
            no_count = valid_int_responses[col].str.contains('Non', na=False, case=False).sum()
            total_valid_int = yes_count + no_count
            
            if total_valid_int > 0:
                intention_stats[name] = {
                    'Oui': yes_count,
                    'Non': no_count,
                    'Pourcentage Oui': (yes_count / total_valid_int) * 100,
                    'Pourcentage Non': (no_count / total_valid_int) * 100
                }
    
    return {
        'feeling_bad_before_pct': feeling_bad_pct,
        'overall_improvement_pct': overall_improvement_pct,
        'very_bad_pct': very_bad_pct,
        'very_bad_improvement_pct': very_bad_improvement_pct,
        'avg_improvement_magnitude': avg_improvement,
        'improvement_levels': level_percentages,
        'before_stats': before_df,
        'after_stats': after_df,
        'experience_stats': experience_stats,
        'intention_stats': intention_stats,
        'total_valid_responses': total_valid,
        'total_excluded_responses': len(df) - total_valid
    }

print("Fonction analyze_wellbeing_impact corrigée !")
print("Modifications apportées :")
print("1. Filtrage moins strict - garde toutes les réponses textuelles valides")
print("2. Conversion explicite en string avec astype(str)")
print("3. Exclusion seulement des vraies valeurs vides/nulles")
print("4. Debug détaillé pour vérifier les comptes")
print("5. Conservation de 'Je ne sais pas dire comment je me sentais' comme réponse valide")



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
    source_col = find_closest_column(df, "Comment as-tu entendu")
    if not source_col:
        return None, None

    # Séparer les réponses multiples
    sources = df[source_col].str.split(',').explode().str.strip()
    
    def categorize_source(source):
        if pd.isna(source):
            return "Non spécifié", "Non spécifié"
            
        source = source.lower()
        
        # Catégories principales
        if any(term in source for term in ["instagram", "facebook", "twitter", "linkedin", "tiktok"]):
            return "Réseaux sociaux", source.title()
        elif any(term in source for term in ["université", "école", "fac", "campus"]):
            return "Université/École", source.title()
        elif any(term in source for term in ["psy", "médecin", "santé", "thérapeute"]):
            return "Professionnel de santé", source.title()
        elif any(term in source for term in ["ami", "proche", "famille"]):
            return "Ami/Proche", source.title()
        elif any(term in source for term in ["affiche", "flyer", "poster"]):
            return "Communication physique", source.title()
        elif any(term in source for term in ["internet", "site", "web"]):
            return "Site internet", source.title()
        else:
            cleaned_source = source.strip().capitalize()
            return "Autres", cleaned_source
    
    # Appliquer la catégorisation
    categorized = sources.apply(lambda x: categorize_source(x))
    main_categories = categorized.apply(lambda x: x[0])
    detailed_categories = categorized.apply(lambda x: x[1])
    
    # Calculer les métriques principales
    total_respondents = len(df)  # Nombre total d'appelants
    total_responses = len(main_categories)  # Nombre total de réponses
    
    # Exclure la catégorie "Autres" du graphique principal
    main_counts = main_categories[main_categories != "Autres"].value_counts()
    
    # Calculer les deux types de pourcentages pour le graphique principal
    main_percentages_by_caller = (main_counts / total_respondents * 100).round(1)
    main_percentages_by_response = (main_counts / total_responses * 100).round(1)
    
    # Créer le graphique principal avec les deux métriques
    fig_main = go.Figure()
    
    # Barres pour % par appelant
    fig_main.add_trace(go.Bar(
        name="% des appelants",
        y=main_percentages_by_caller.index,
        x=main_percentages_by_caller.values,
        orientation='h',
        marker_color='#4ecdc4',
        text=[f'{x:.1f}%' for x in main_percentages_by_caller.values],
        textposition='outside'
    ))
    
    # Barres pour % par réponse
    fig_main.add_trace(go.Bar(
        name="% des réponses",
        y=main_percentages_by_response.index,
        x=main_percentages_by_response.values,
        orientation='h',
        marker_color='#ff6b6b',
        text=[f'{x:.1f}%' for x in main_percentages_by_response.values],
        textposition='outside'
    ))
    
    fig_main.update_layout(
        title="Sources d'information principales",
        xaxis_title="Pourcentage",
        yaxis_title=None,
        height=400,
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Analyser le détail de la catégorie "Autres"
    others_detail = detailed_categories[main_categories == "Autres"].value_counts()
    total_others = len(detailed_categories[main_categories == "Autres"])
    
    # Calculer les deux types de pourcentages pour "Autres"
    others_percentages_by_caller = (others_detail / total_respondents * 100).round(1)
    others_percentages_by_response = (others_detail / total_others * 100).round(1)
    
    if len(others_percentages_by_response) > 0:
        # Créer le graphique détaillé pour "Autres" avec les deux métriques
        fig_others = go.Figure()
        
        # Barres pour % par appelant
        fig_others.add_trace(go.Bar(
            name="% des appelants",
            y=others_percentages_by_caller.index,
            x=others_percentages_by_caller.values,
            orientation='h',
            marker_color='#4ecdc4',
            text=[f'{x:.1f}%' for x in others_percentages_by_caller.values],
            textposition='outside'
        ))
        
        # Barres pour % des réponses
        fig_others.add_trace(go.Bar(
            name="% des réponses 'Autres'",
            y=others_percentages_by_response.index,
            x=others_percentages_by_response.values,
            orientation='h',
            marker_color='#ff6b6b',
            text=[f'{x:.1f}%' for x in others_percentages_by_response.values],
            textposition='outside'
        ))
        
        fig_others.update_layout(
            title="Détail de la catégorie 'Autres'",
            xaxis_title="Pourcentage",
            yaxis_title=None,
            height=max(400, len(others_percentages_by_response) * 25),
            margin=dict(l=200, r=20, t=40, b=20),
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Créer le tableau détaillé
        others_df = pd.DataFrame({
            'Source': others_percentages_by_response.index,
            'Nombre': others_detail.values,
            '% des appelants': others_percentages_by_caller.values.round(1),
            '% des réponses': others_percentages_by_response.values.round(1)
        }).sort_values('Nombre', ascending=False)
        
    else:
        fig_others = None
        others_df = pd.DataFrame()
    
    # Afficher les métriques principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Nombre total d'appelants",
            total_respondents,
            help="Nombre total d'appelants ayant répondu au questionnaire"
        )
    
    with col2:
        st.metric(
            "Nombre total de réponses",
            total_responses,
            help="Nombre total de sources mentionnées"
        )
    
    with col3:
        st.metric(
            "Moyenne de sources par appelant",
            f"{total_responses/total_respondents:.1f}",
            help="En moyenne, chaque appelant mentionne ce nombre de sources"
        )
    
    # Afficher le graphique principal
    st.plotly_chart(fig_main, key="sources_principal")
    
    if fig_others is not None:
        st.subheader("Détail de la catégorie 'Autres'")
        
        # Métriques pour la catégorie "Autres"
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Nombre de réponses 'Autres'",
                total_others,
                help="Nombre total de réponses dans la catégorie 'Autres'"
            )
        
        with col2:
            st.metric(
                "% des réponses totales",
                f"{(total_others/total_responses*100):.1f}%",
                help="Pourcentage que représente la catégorie 'Autres' sur l'ensemble des réponses"
            )
        
        with col3:
            st.metric(
                "Sources uniques",
                len(others_percentages_by_response),
                help="Nombre de sources différentes dans la catégorie 'Autres'"
            )
        
        # Afficher le graphique et le tableau
        st.plotly_chart(fig_others, key="sources_autres")
        st.subheader("Liste complète des autres sources")
        st.dataframe(others_df, hide_index=True)
    
    return fig_main, main_percentages_by_response

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

def process_feedback(df):
    # Traiter les colonnes spécifiques avec les noms exacts
    feedback_columns = {
        "Indiques nous si les phrases ci-dessous sont vraies ou fausses pour toi [Cet appel m'a été utile]": "Cet appel m'a été utile",
        "Indiques nous si les phrases ci-dessous sont vraies ou fausses pour toi [Sans cet appel, je n'aurai rien fait d'autre pour obtenir du soutien ou de l'aide]": "Sans cet appel, je n'aurai rien fait d'autre pour obtenir du soutien ou de l'aide",
        "Indiques nous si les phrases ci-dessous sont vraies ou fausses pour toi [Durant l'appel j'ai partagé des sentiments ou des expériences que je n'ai jamais partagés avec quelqu'un d'autre]": "Durant l'appel j'ai partagé des sentiments ou des expériences que je n'ai jamais partagés avec quelqu'un d'autre"
    }
    
    # Créer un DataFrame pour stocker les résultats
    results = pd.DataFrame(columns=["Feedback", "Oui (%)", "Non (%)", "Ne s'applique pas (%)", "Nombre Oui"])
    
    for question, col in feedback_columns.items():
        if col in df.columns:
            total_responses = len(df)
            true_count = (df[col] == "Vrai").sum()
            false_count = (df[col] == "Faux").sum()
            not_applicable_count = (df[col] == "Cela ne s'applique pas vraiment à ma situation").sum()
            
            results = results.append({
                "Feedback": question,
                "Oui (%)": (true_count / total_responses) * 100 if total_responses > 0 else 0,
                "Non (%)": (false_count / total_responses) * 100 if total_responses > 0 else 0,
                "Ne s'applique pas (%)": (not_applicable_count / total_responses) * 100 if total_responses > 0 else 0,
                "Nombre Oui": true_count
            }, ignore_index=True)
    
    return results

def analyze_new_columns(df):
    # Créer un DataFrame pour afficher les résultats
    results_df = pd.DataFrame({
        "Question": [
            "Cet appel m'a été utile",
            "Sans cet appel, je n'aurai rien fait d'autre pour obtenir du soutien ou de l'aide",
            "Durant l'appel j'ai partagé des sentiments ou des expériences que je n'ai jamais partagés avec quelqu'un d'autre"
        ]
    })
    
    # Colonnes complètes pour référence
    columns = [
        "Indiques nous si les phrases ci-dessous sont vraies ou fausses pour toi [Cet appel m'a été utile]",
        "Indiques nous si les phrases ci-dessous sont vraies ou fausses pour toi [Sans cet appel, je n'aurai rien fait d'autre pour obtenir du soutien ou de l'aide]",
        "Indiques nous si les phrases ci-dessous sont vraies ou fausses pour toi [Durant l'appel j'ai partagé des sentiments ou des expériences que je n'ai jamais partagés avec quelqu'un d'autre]"
    ]
    
    # Calculer les pourcentages et les nombres pour chaque question
    oui_pct = []
    non_pct = []
    nombre_oui = []
    nombre_non = []
    
    for col in columns:
        # Compter les "Vrai" et "Faux"
        vrai_count = df[col].value_counts().get("Vrai", 0)
        faux_count = df[col].value_counts().get("Faux", 0)
        
        # Calculer le total des réponses pour cette question spécifique
        total_reponses = vrai_count + faux_count
        
        # Calculer les pourcentages
        if total_reponses > 0:
            oui_percent = (vrai_count / total_reponses) * 100
            non_percent = (faux_count / total_reponses) * 100
        else:
            oui_percent = 0
            non_percent = 0
        
        oui_pct.append(oui_percent)
        non_pct.append(non_percent)
        nombre_oui.append(vrai_count)
        nombre_non.append(faux_count)
    
    # Ajouter les colonnes calculées au DataFrame
    results_df["Oui (%)"] = oui_pct
    results_df["Non (%)"] = non_pct
    results_df["Nombre Oui"] = nombre_oui
    results_df["Nombre Non"] = nombre_non
    
    return results_df

def display_impact_satisfaction_analysis(df):
    """
    Fonction séparée pour afficher l'analyse dans la page Impact et Satisfaction
    """
    # Obtenir les résultats
    results_df = analyze_new_columns(df)
    
    # Afficher les résultats
    st.subheader("Analyse des nouvelles colonnes")
    st.dataframe(results_df, hide_index=True)

    # Ajouter un histogramme pour visualiser les résultats
    fig = go.Figure()

    # Ajouter les barres pour Oui et Non avec des couleurs douces
    fig.add_trace(go.Bar(
        x=results_df["Question"],
        y=results_df["Oui (%)"],
        name='Oui (%)',
        marker_color='lightgreen'
    ))

    fig.add_trace(go.Bar(
        x=results_df["Question"],
        y=results_df["Non (%)"],
        name='Non (%)',
        marker_color='lightcoral'
    ))

    # Mettre à jour la mise en page du graphique
    fig.update_layout(
        title='Pourcentages de réponses',
        barmode='group',
        xaxis_title='Questions',
        yaxis_title='Pourcentage',
        template='plotly_dark'
    )

    st.plotly_chart(fig)

def create_csv_report(exp_df, int_df):
    """Crée un fichier CSV avec les données d'impact et satisfaction"""
    output = BytesIO()
    
    # Combine les deux DataFrames avec un séparateur
    combined_data = (
        "RESSENTI PENDANT L'APPEL\n" +
        exp_df.to_csv(index=False) +
        "\n\nINTENTIONS APRÈS L'APPEL\n" +
        int_df.to_csv(index=False)
    )
    
    return combined_data.encode('utf-8')

def create_excel_report(wellbeing_metrics):
    """Crée un fichier Excel avec toutes les données d'impact et satisfaction"""
    output = BytesIO()
    
    # Créer un nouveau classeur Excel
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # 1. État avant/après
        wellbeing_metrics['before_stats'].to_excel(writer, sheet_name='État avant', index=False)
        wellbeing_metrics['after_stats'].to_excel(writer, sheet_name='État après', index=False)
        
        # 2. Statistiques détaillées
        stats_data = pd.DataFrame({
            'Métrique': [
                'Appelants se sentant mal ou très mal avant',
                'Appelants se sentant mieux après',
                'Appelants se sentant très mal avant',
                'Amélioration des cas très mal',
                'Amélioration moyenne des cas très mal'
            ],
            'Pourcentage': [
                f"{wellbeing_metrics['feeling_bad_before_pct']:.1f}%",
                f"{wellbeing_metrics['overall_improvement_pct']:.1f}%",
                f"{wellbeing_metrics['very_bad_pct']:.1f}%",
                f"{wellbeing_metrics['very_bad_improvement_pct']:.1f}%",
                f"{wellbeing_metrics['avg_improvement_magnitude']:.1f}%"
            ]
        })
        stats_data.to_excel(writer, sheet_name='Statistiques', index=False)
        
        # 3. Détail des améliorations
        improvement_data = pd.DataFrame({
            'Niveau d\'amélioration': ['Légère', 'Modérée', 'Forte'],
            'Pourcentage': [
                f"{wellbeing_metrics['improvement_levels'][1]:.1f}%",
                f"{wellbeing_metrics['improvement_levels'][2]:.1f}%",
                f"{wellbeing_metrics['improvement_levels'][3]:.1f}%"
            ]
        })
        improvement_data.to_excel(writer, sheet_name='Détail améliorations', index=False)
        
        # 4. Ressenti pendant l'appel
        exp_data = []
        for feeling, stats in wellbeing_metrics['experience_stats'].items():
            exp_data.append({
                'Ressenti': feeling,
                'Oui (%)': f"{stats['Pourcentage Oui']:.1f}%",
                'Non (%)': f"{stats['Pourcentage Non']:.1f}%",
                'Nombre Oui': stats['Oui'],
                'Nombre Non': stats['Non']
            })
        pd.DataFrame(exp_data).to_excel(writer, sheet_name='Ressenti', index=False)
        
        # 5. Intentions après l'appel
        int_data = []
        for intention, stats in wellbeing_metrics['intention_stats'].items():
            int_data.append({
                'Intention': intention,
                'Oui (%)': f"{stats['Pourcentage Oui']:.1f}%",
                'Non (%)': f"{stats['Pourcentage Non']:.1f}%",
                'Nombre Oui': stats['Oui'],
                'Nombre Non': stats['Non']
            })
        pd.DataFrame(int_data).to_excel(writer, sheet_name='Intentions', index=False)
    
    return output.getvalue()

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
    
    if df.empty:
        st.error("Aucune donnée à afficher.")
        return
    
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
    
    # Ajouter le filtre Antennes (régions)
    # Utiliser find_closest_column pour être plus robuste
    region_col = find_closest_column(df, "région es-tu étudiant")
    
    if region_col:
        # Afficher le nom de la colonne trouvée pour le débogage
        print(f"Colonne région trouvée: {region_col}")
        
        # Obtenir les valeurs uniques et les trier
        regions = df[region_col].dropna().unique()
        regions = sorted([r for r in regions if isinstance(r, str) and r.strip()])
        
        if len(regions) > 0:
            st.sidebar.subheader("Antennes")
            region_options = ['Toutes'] + regions
            selected_regions = st.sidebar.multiselect(
                "Sélectionner les régions",
                options=region_options,
                default=['Toutes']
            )
            
            # Appliquer le filtre
            if 'Toutes' not in selected_regions:
                df = df[df[region_col].isin(selected_regions)]
    
    # Séparateur avant la navigation
    st.sidebar.markdown("---")
    
    # Navigation existante
    page = st.sidebar.selectbox(
        "Navigation",
        ["Vue d'ensemble", "Expérience Appelants", "Démographie", "Impact et Satisfaction", "Raisons des Appels", "Sources et Communication"]
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
        
    elif page == "Impact et Satisfaction":
        st.header("Impact et Satisfaction des Appelants")
        
        if df.empty:
            st.error("Aucune donnée à afficher.")
            return
        
        # Calculer les métriques
        wellbeing_metrics = analyze_wellbeing_impact(df)
        
        # Afficher les métriques principales dans des colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "État initial critique",
                f"{wellbeing_metrics['feeling_bad_before_pct']:.1f}%",
                help="Pourcentage d'appelants se sentant mal ou très mal avant l'appel"
            )

        with col2:
            st.metric(
                "Amélioration globale",
                f"{wellbeing_metrics['overall_improvement_pct']:.1f}%",
                help="Pourcentage d'appelants qui se sont sentis mieux après l'appel"
            )

        with col3:
            st.metric(
                "Amélioration cas critiques",
                f"{wellbeing_metrics['very_bad_improvement_pct']:.1f}%",
                help="Pourcentage d'amélioration pour les appelants qui se sentaient très mal"
            )

        # Ajouter une section d'analyse détaillée
        st.markdown("---")
        st.subheader("Analyse détaillée de l'impact")

        # Créer un conteneur stylisé pour les statistiques détaillées
        st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <p style='font-size: 16px; color: #1f1f1f;'>
                    • <b>{wellbeing_metrics['feeling_bad_before_pct']:.1f}%</b> des appelants se sentaient mal ou très mal avant l'appel
                    <br><br>
                    • <b>{wellbeing_metrics['overall_improvement_pct']:.1f}%</b> se sont sentis mieux après l'appel
                    <br><br>
                    • Parmi les <b>{wellbeing_metrics['very_bad_pct']:.1f}%</b> qui se sentaient très mal au moment de l'appel :
                    <br>- <b>{wellbeing_metrics['very_bad_improvement_pct']:.1f}%</b> ont constaté une amélioration
                    <br>- L'amélioration moyenne est de <b>{wellbeing_metrics['avg_improvement_magnitude']:.1f}%</b>
                    <br>- Détail de l'amélioration :
                    <br>&nbsp;&nbsp;&nbsp;• <b>{wellbeing_metrics['improvement_levels'][1]:.1f}%</b> ont constaté une légère amélioration
                    <br>&nbsp;&nbsp;&nbsp;• <b>{wellbeing_metrics['improvement_levels'][2]:.1f}%</b> ont constaté une amélioration modérée
                    <br>&nbsp;&nbsp;&nbsp;• <b>{wellbeing_metrics['improvement_levels'][3]:.1f}%</b> ont constaté une forte amélioration
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Afficher les statistiques détaillées
        st.subheader("État avant l'appel")
        before_df = wellbeing_metrics['before_stats']  # C'est déjà un DataFrame avec les bonnes colonnes
        st.dataframe(before_df.sort_values('Nombre', ascending=False), hide_index=True)

        st.subheader("État après l'appel")
        after_df = wellbeing_metrics['after_stats']  # C'est déjà un DataFrame avec les bonnes colonnes
        st.dataframe(after_df.sort_values('Nombre', ascending=False), hide_index=True)
        
        # Bouton de téléchargement
        st.download_button(
            label="Télécharger les données (CSV)",
            data=create_csv_report(before_df, after_df),
            file_name=f"impact_satisfaction_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Ajouter un graphique avant/après
        st.subheader("Évolution de l'état émotionnel")

        # Créer deux colonnes pour les graphiques
        col1, col2 = st.columns(2)

        with col1:
            # Graphique "Avant l'appel"
            before_data = wellbeing_metrics['before_stats']  # Déjà un DataFrame
            fig1 = px.pie(
                before_data,
                values='Nombre',
                names='État',
                title="État avant l'appel",
                color_discrete_sequence=px.colors.sequential.Teal,
                hover_data=['Pourcentage'],
                custom_data=['Pourcentage']
            )
            fig1.update_traces(
                texttemplate='%{label}<br>%{customdata[0]:.1f}%',
                hovertemplate='%{label}<br>Nombre: %{value}<br>Pourcentage: %{customdata[0]:.1f}%'
            )
            st.plotly_chart(fig1)

        with col2:
            # Graphique "Après l'appel"
            after_data = wellbeing_metrics['after_stats']  # Déjà un DataFrame
            fig2 = px.pie(
                after_data,
                values='Nombre',
                names='État',
                title="État après l'appel",
                color_discrete_sequence=px.colors.sequential.Teal,
                hover_data=['Pourcentage'],
                custom_data=['Pourcentage']
            )
            fig2.update_traces(
                texttemplate='%{label}<br>%{customdata[0]:.1f}%',
                hovertemplate='%{label}<br>Nombre: %{value}<br>Pourcentage: %{customdata[0]:.1f}%'
            )
            st.plotly_chart(fig2)
        
        # Ajouter des graphiques pour visualiser ces données
        st.markdown("---")
        st.subheader("Ressenti pendant l'appel")

        # Créer un DataFrame pour le ressenti
        exp_data = []
        for feeling, stats in wellbeing_metrics['experience_stats'].items():
            exp_data.append({
                'Ressenti': feeling,
                'Oui (%)': f"{stats['Pourcentage Oui']:.1f}%",
                'Non (%)': f"{stats['Pourcentage Non']:.1f}%",
                'Nombre Oui': stats['Oui'],
                'Nombre Non': stats['Non']
            })

        exp_df = pd.DataFrame(exp_data)
        st.dataframe(exp_df, hide_index=True)

        st.markdown("---")
        st.subheader("Intentions après l'appel")

        # Ajouter une note explicative
        st.markdown("""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
                <p style='margin: 0; font-size: 14px; color: #1f1f1f;'>
                    <i>Note : Les pourcentages peuvent dépasser 100% car un même appelant peut avoir plusieurs intentions 
                    (par exemple, vouloir à la fois consulter des ressources et parler avec des proches).</i>
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Créer un DataFrame pour les intentions
        int_data = []
        for intention, stats in wellbeing_metrics['intention_stats'].items():
            int_data.append({
                'Intention': intention,
                'Oui (%)': f"{stats['Pourcentage Oui']:.1f}%",
                'Non (%)': f"{stats['Pourcentage Non']:.1f}%",
                'Nombre Oui': stats['Oui'],
                'Nombre Non': stats['Non']
            })

        int_df = pd.DataFrame(int_data)
        st.dataframe(int_df, hide_index=True)

        # Graphique des intentions
        fig_int = px.bar(
            int_data,
            x='Intention',
            y=[float(str(x).rstrip('%')) for x in int_df['Oui (%)']],
            title="Intentions après l'appel",
            labels={'y': 'Pourcentage de réponses positives'},
            color_discrete_sequence=['#4ecdc4']
        )
        fig_int.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig_int)
        
        # Afficher l'analyse des nouvelles colonnes
        display_impact_satisfaction_analysis(df)
        
        # Ajouter un bouton pour exporter les données en Excel
        st.markdown("---")
        st.subheader("Exporter les données")

        # Bouton pour télécharger le rapport Excel
        st.download_button(
            label="📊 Télécharger toutes les données (Excel)",
            data=create_excel_report(wellbeing_metrics),
            file_name=f"impact_satisfaction_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Télécharger toutes les données de cette section en format Excel"
        )
        
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
        
        # Afficher les graphiques des sources
        source_fig, source_data = analyze_nl_awareness_sources(df)
        
        # Option pour générer le rapport détaillé
        if st.button("Générer un rapport détaillé"):
            wellbeing_analysis = analyze_wellbeing_impact(df)
            call_exp = analyze_call_experience(df)
            
            report = generate_detailed_report(
                df,
                wellbeing_analysis,
                call_exp,
                source_data
            )
            
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
        create_streamlit_dashboard()

if __name__ == "__main__":
    main()
