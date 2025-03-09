import os
import time
import threading
import joblib
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import matplotlib.colors as mcolors
import numpy as np
import requests
import json
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure folders
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load models
models = {
    "sardine": joblib.load("MODEL/species_prediction_ESP1_optimized.pkl"),
    "rails": joblib.load("MODEL/species_prediction_ESP2_optimized.pkl"),
}

# Load shapefiles
morocco_shapefile = gpd.read_file("VF/T_cote_Maroc_MAJ_MAI_06_2021.shp")
grid_shapefile = gpd.read_file("GRILLE/grill_adam.shp")

# DeepSeek API configuration
DEEPSEEK_API_KEY = "sk-e1c90c8938854adfb6a07e5618d6e093"  # Replace with your API key
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

# Dictionary to store temporary predictions
predictions_cache = {}

def delete_files_after_delay(file_paths, delay=120):
    """Delete files after a specified delay in seconds (default: 2 minutes)."""
    time.sleep(delay)
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

def get_regional_analysis(predictions, species):
    """Generate regional analysis using DeepSeek API."""
    cities_data = {
        "Agadir": {"lat": 30.427755, "lon": -9.598107, "facteurs": ["upwelling", "nutriments"]},
        "Tanger": {"lat": 35.7595, "lon": -5.8340, "facteurs": ["courants", "température"]},
        "Dakhla": {"lat": 23.6841, "lon": -15.9575, "facteurs": ["salinité", "profondeur"]}
    }

    prompt = f"""
    Analysez cette distribution de présence de {species} selon ces données clés:
    - Concentration moyenne: {np.mean(predictions):.2f}
    - Maximum: {np.max(predictions):.2f} | Minimum: {np.min(predictions):.2f}
    
    Localisations stratégiques:
    {json.dumps(cities_data, indent=4)}

    Fournissez une analyse en français avec:
    1. Classement des zones par densité (Agadir vs Tanger vs Dakhla)
    2. Facteurs environnementaux dominants pour chaque zone
    3. Recommandations de gestion halieutique
    4. Périodes de concentration maximale

    Basez-vous sur ces patterns cartographiques:
    - Zones rouges: forte concentration
    - Zones jaunes: présence modérée
    - Zones non colorées: faible présence
    """

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 500
    }

    try:
        response = requests.post(DEEPSEEK_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Erreur d'analyse: {str(e)}"

def process_and_plot(csv_path, species):
    """Process data and generate the map for the selected species."""
    if species not in models:
        raise ValueError("Espèce invalide. Choisissez 'sardine' ou 'rails'.")

    model = models[species]
    prediction_column = "Pres_Esp1_pred" if species == "sardine" else "Pres_Esp2_pred"

    # Load CSV data
    df = pd.read_csv(csv_path, delimiter=";")
    df.columns = df.columns.str.strip()

    # Convert numeric columns
    for col in ["LAT_DD", "LONG_DD", "Salinite", "Temp", "DO", "pH"]:
        df[col] = df[col].str.replace(",", ".").astype(float)

    # Predict using the model
    features = ["LAT_DD", "LONG_DD", "Salinite", "Temp", "DO", "pH"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    df[prediction_column] = model.predict(X_scaled)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, geometry=[Point(lon, lat) for lon, lat in zip(df["LONG_DD"], df["LAT_DD"])], crs="EPSG:4326"
    )

    # Create a colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("YellowToRed", ["yellow", "red"])

    # Plot the map
    fig, ax = plt.subplots(figsize=(14, 16))
    morocco_shapefile.plot(ax=ax, color="white", edgecolor="black", linewidth=0.5, alpha=0.7)

    for _, grid_row in grid_shapefile.iterrows():
        points_in_square = gdf[gdf.geometry.within(grid_row.geometry)]
        if not points_in_square.empty:
            mean_pred = points_in_square[prediction_column].mean()
            color = cmap(mean_pred)
            grid_shapefile[grid_shapefile["LABEL"] == grid_row["LABEL"]].plot(
                ax=ax, 
                edgecolor="none", 
                facecolor=color, 
                alpha=0.7
            )

    # Add OpenStreetMap basemap
    ctx.add_basemap(
        ax=ax,
        crs=gdf.crs.to_string(),
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha=0.6
    )

    ax.set_title(f"Carte de présence de {species.capitalize()} (Prédictions)", fontsize=18, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.025, pad=0.02, label=f"Présence {species.capitalize()}")

    output_path = os.path.join(STATIC_FOLDER, f"output_map_{species}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Store predictions in cache
    predictions_cache[species] = df[prediction_column].values

    # Schedule file deletion after 2 minutes
    threading.Thread(target=delete_files_after_delay, args=([csv_path, output_path], 120)).start()

    return output_path

@app.route("/upload", methods=["POST"])
def upload_file():
    """Route to upload a file and generate the map."""
    if "file" not in request.files or "species" not in request.form:
        return jsonify({"error": "Fichier ou espèce non spécifiée"}), 400

    file = request.files["file"]
    species = request.form["species"].lower()

    if file.filename == "":
        return jsonify({"error": "Fichier invalide"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:
        # Generate the map
        image_path = process_and_plot(file_path, species)
        return jsonify({"image_path": image_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze_data():
    """Route to perform DeepSeek analysis."""
    if "species" not in request.form:
        return jsonify({"error": "Espèce non spécifiée"}), 400

    species = request.form["species"].lower()

    if species not in predictions_cache:
        return jsonify({"error": "Aucune donnée disponible pour cette espèce"}), 404

    try:
        # Retrieve predictions from cache
        predictions = predictions_cache[species]
        
        # Generate regional analysis
        analysis = get_regional_analysis(predictions, species)
        return jsonify({"analysis": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download/<species>")
def download_map(species):
    """Route to download the generated map."""
    file_path = f"static/output_map_{species}.png"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "Fichier non trouvé"}), 404

if __name__ == "__main__":
    app.run(debug=True)
