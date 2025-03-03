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
            h1 {{ color: #4ecdc4; }}
            .metric {{ margin: 20px 0; padding: 10px; background: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>Rapport Détaillé Nightline</h1>
        <div class="metric">
            <h2>Impact sur le Bien-être</h2>
            <p>Amélioration: {wellbeing_analysis['improvement_percentage']:.1f}%</p>
            <p>Satisfaction: {call_exp['experience_metrics']['comfort']:.1f}%</p>
        </div>
    </body>
    </html>
    """
    return html_content
