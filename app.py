
import streamlit as st
from pytrends.request import TrendReq
from serpapi import GoogleSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import openai
import pandas as pd

# --- API KEYS ---
SERPAPI_KEY = "iA3WvNt5WVtKYqWbXmQpmWeG"
openai.api_key = "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"

# --- UI ---
st.title("🔎 Google Trends & SERP Content Finder")
queries = [st.text_input(f"Suchbegriff {i+1}") for i in range(5)]
time_filter = st.selectbox("Zeitraum auswählen", ["Letzte 4 Stunden", "Letzter Tag"])

# --- Google Trends ---
pytrends = TrendReq(hl='de', tz=360)
related_queries = {}

for query in queries:
    if query:
        timeframe = 'now 4-H' if time_filter == "Letzte 4 Stunden" else 'now 1-d'
        pytrends.build_payload([query], timeframe=timeframe)
        data = pytrends.related_queries().get(query, {}).get('top')
        if data is not None:
            related_queries[query] = data
            st.markdown(f"### 📈 Trends für '{query}'")
            st.dataframe(data)

# --- SERP Screenshot via SerpAPI ---
def get_serp_screenshot(query):
    params = {
        "engine": "google",
        "q": query,
        "hl": "de",
        "gl": "de",
        "tbs": "qdr:h" if time_filter == "Letzte 4 Stunden" else "qdr:d",
        "api_key": SERPAPI_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("screenshot", None), results

serp_data = {}
for query in queries:
    if query:
        screenshot_url, serp_result = get_serp_screenshot(query)
        if screenshot_url:
            st.image(screenshot_url, caption=f"SERP für '{query}'")
        serp_data[query] = serp_result

# --- Keyword Clustering ---
def cluster_keywords(keywords, n_clusters=3):
    vect = TfidfVectorizer()
    X = vect.fit_transform(keywords)
    kmeans = KMeans(n_clusters=min(n_clusters, len(keywords)))
    kmeans.fit(X)
    clusters = {i: [] for i in range(kmeans.n_clusters)}
    for keyword, label in zip(keywords, kmeans.labels_):
        clusters[label].append(keyword)
    return clusters

# --- KI Empfehlungen ---
def generate_recommendation(query, clusters, headlines):
    prompt = f"""
    Du bist ein erfahrener SEO-Experte. Für das Thema '{query}' wurden folgende Keyword-Cluster erkannt: {clusters}.
    Diese Headlines wurden auf der SERP gefunden: {headlines}.
    Gib bitte eine Empfehlung, worüber heute ein Artikel geschrieben werden sollte. Füge einen möglichen Titel, relevante Keywords und eine Bildidee hinzu. Wenn es Newsbox-Angles gibt, nutze diese. Sonst schlage einen originellen Angle vor.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Empfehlungen anzeigen ---
for query in queries:
    if query and query in related_queries:
        top_keywords = related_queries[query]['query'].dropna().tolist()
        if len(top_keywords) > 1:
            clusters = cluster_keywords(top_keywords)
            headlines = ["Headline 1", "Headline 2"]  # TODO: Headlines aus serp_data[query] extrahieren
            recommendation = generate_recommendation(query, clusters, headlines)
            st.markdown(f"### 🔮 KI-Empfehlung für '{query}'")
            st.write(recommendation)
