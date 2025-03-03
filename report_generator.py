import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from utils import find_closest_column

def generate_detailed_report(df, wellbeing_analysis, call_exp, source_data):
    """
    Génère un rapport détaillé au format HTML
    """
    html_content = f"""
    <html>
    <head>
        <title>Rapport Détaillé Nightline</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #4ecdc4; }}
            .metric {{ margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 8px; }}
            .kpi {{ font-size: 24px; color: #333; margin: 10px 0; }}
            .description {{ color: #666; }}
        </style>
    </head>
    <body>
        <h1>Rapport Détaillé Nightline</h1>
        
        <div class="metric">
            <h2>Impact sur le Bien-être</h2>
            <div class="kpi">Amélioration: {wellbeing_analysis['improvement_percentage']:.1f}%</div>
            <div class="kpi">Satisfaction: {call_exp['experience_metrics']['comfort']:.1f}%</div>
        </div>

        <div class="metric">
            <h2>Expérience des Appelants</h2>
            <div class="kpi">Se sont sentis compris: {call_exp['experience_metrics']['understood']:.1f}%</div>
            <div class="kpi">Se sont sentis soutenus: {call_exp['experience_metrics']['supported']:.1f}%</div>
            <div class="kpi">Se sont sentis moins seuls: {call_exp['experience_metrics']['less_lonely']:.1f}%</div>
        </div>

        <div class="metric">
            <h2>Intentions Futures</h2>
            <div class="kpi">Consulter des ressources: {call_exp['future_intentions']['seek_resources']:.1f}%</div>
            <div class="kpi">Parler avec des proches: {call_exp['future_intentions']['talk_to_family']:.1f}%</div>
            <div class="kpi">Voir un professionnel: {call_exp['future_intentions']['seek_professional']:.1f}%</div>
        </div>

        <div class="metric">
            <h2>Sources d'Information</h2>
            {''.join([f"<div class='kpi'>{source}: {percentage:.1f}%</div>" for source, percentage in source_data.items()])}
        </div>

        <div class="metric">
            <h2>États Avant/Après</h2>
            <div class="kpi">Amélioration: {wellbeing_analysis['improvement_percentage']:.1f}%</div>
            <div class="kpi">Détérioration: {wellbeing_analysis['worsening_percentage']:.1f}%</div>
        </div>
    </body>
    </html>
    """
    return html_content

def generate_source_list(source_analysis):
    """Génère la liste HTML des sources triées par importance"""
    source_items = []
    for source, percentage in source_analysis.items():
        source_items.append(
            f'<li>{source}: <span class="highlight">{percentage:.1f}%</span></li>'
        )
    return '\n'.join(source_items) 
