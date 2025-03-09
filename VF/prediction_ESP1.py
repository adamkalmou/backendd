import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model (fix the path to the correct model file)
model = joblib.load('C:/Users/ADAM/Desktop/Nouveau dossier (9)/projet pfe vf/VF/species_prediction_model_ESP1.pkl')

# 1. Load the shapefiles
morocco_shapefile = gpd.read_file("C:/Users/ADAM/Desktop/Nouveau dossier (9)/projet pfe vf/VF/T_cote_Maroc_MAJ_MAI_06_2021.shp")  # Morocco map
grid_shapefile = gpd.read_file("C:/Users/ADAM/Desktop/Nouveau dossier (9)/projet pfe vf/VF/Grill_22122024_VF.shp")  # Grid squares with labels

# 2. Load the test CSV data (fix separator and clean column names)
csv_file = "C:/Users/ADAM/Desktop/Nouveau dossier (9)/projet pfe vf/VF/a.cop.csv"  # New data for prediction
df = pd.read_csv(csv_file, delimiter=';')  # Ensure this matches your file's separator

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Check for required columns
required_columns = ['LAT_D', 'LONG_D', 'Salinite', 'Temp', 'DO', 'pH']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    print("All required columns are present.")

# Replace commas with dots for numeric columns and convert to float
df['LAT_D'] = df['LAT_D'].str.replace(',', '.').astype(float)
df['LONG_D'] = df['LONG_D'].str.replace(',', '.').astype(float)
df['Salinite'] = df['Salinite'].str.replace(',', '.').astype(float)
df['Temp'] = df['Temp'].str.replace(',', '.').astype(float)
df['DO'] = df['DO'].str.replace(',', '.').astype(float)
df['pH'] = df['pH'].str.replace(',', '.').astype(float)

# 3. Prepare the feature columns for prediction
features = ['LAT_D', 'LONG_D', 'Salinite', 'Temp', 'DO', 'pH']
X_new = df[features]

# Standardize the new data (use the same scaler as before)
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new)

# 4. Make predictions
predictions = model.predict(X_new_scaled)

# Add predictions to the dataframe
df['Pres_Esp1_pred'] = predictions

# Convert the dataframe to a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=[Point(lon, lat) for lon, lat in zip(df['LONG_D'], df['LAT_D'])], crs="EPSG:4326")

# 5. Plot the map of Morocco
fig, ax = plt.subplots(figsize=(10, 12))
morocco_shapefile.plot(ax=ax, color='lightgray', edgecolor='black')

# 6. Iterate over each grid square to color based on the predicted Pres_Esp1 values
for _, grid_row in grid_shapefile.iterrows():
    label = grid_row['LABEL']
    
    # Find the points in the dataframe that are inside this grid square
    points_in_square = gdf[gdf.geometry.within(grid_row.geometry)]
    
    # Check if there is any data (i.e., points in this grid square)
    if points_in_square.empty:
        # No data for this grid square, color it white
        grid_shapefile[grid_shapefile['LABEL'] == label].plot(ax=ax, edgecolor='none', facecolor='#C4C8C5', alpha=0.5)
    else:
        # Get the predicted Pres_Esp1 value for the points inside this square
        pres_esp1_values = points_in_square['Pres_Esp1_pred']
        
        # If any point has Pres_Esp1 == 1, color the grid square red
        if (pres_esp1_values == 1).any():
            grid_shapefile[grid_shapefile['LABEL'] == label].plot(ax=ax, edgecolor='none', facecolor='red', alpha=0.5)  # Red if Pres_Esp1 == 1
        else:
            grid_shapefile[grid_shapefile['LABEL'] == label].plot(ax=ax, edgecolor='none', facecolor='yellow', alpha=0.5)  # Yellow if Pres_Esp1 == 0

# 7. Customize the plot
ax.set_title("Grid Square Coloring Based on Predicted Species Presence (Pres_Esp1)", fontsize=16)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# 8. Hide the grid itself (no grid will be shown to the user)
ax.grid(False)

# 9. Show the plot
plt.show()
