def find_closest_column(df, search_term):
    """
    Trouve la colonne qui correspond le mieux au terme recherché
    """
    search_term = search_term.lower()
    for col in df.columns:
        if search_term in col.lower():
            return col
    return None
