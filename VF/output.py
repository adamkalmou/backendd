import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

# 1. Load the shapefiles
morocco_shapefile = gpd.read_file("C:/Users/ADAM/Desktop/Nouveau dossier (9)/projet pfe vf/VF/T_cote_Maroc_MAJ_MAI_06_2021.shp")  # Morocco map
grid_shapefile = gpd.read_file("C:/Users/ADAM/Desktop/Nouveau dossier (9)/projet pfe vf/VF/Grill_22122024_VF.shp")  # Grid squares with labels

# 2. Load the CSV data with proper delimiter
csv_file = "C:/Users/ADAM/Desktop/Nouveau dossier (9)/projet pfe vf/VF/a.csv"
df = pd.read_csv(csv_file, delimiter=';')  # Use ';' as the separator

# Debug: Print the corrected column names
print("Columns in the CSV file after splitting:", df.columns)

# 3. Replace commas with dots and convert to float for coordinates
longitude_col = "LONG_D"
latitude_col = "LAT_D"
df[longitude_col] = df[longitude_col].str.replace(',', '.').astype(float)
df[latitude_col] = df[latitude_col].str.replace(',', '.').astype(float)

# 4. Convert the dataframe to a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=[Point(lon, lat) for lon, lat in zip(df[longitude_col], df[latitude_col])], crs="EPSG:4326")

# 5. Plot the map of Morocco
fig, ax = plt.subplots(figsize=(10, 12))
morocco_shapefile.plot(ax=ax, color='lightgray', edgecolor='black')

# 6. Plot the grid with transparency
grid_shapefile.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=0.5, alpha=0.2)  # Apply transparency to grid

# 7. Iterate over each grid square to count how many points are present in each square
for _, grid_row in grid_shapefile.iterrows():
    label = grid_row['LABEL']
    
    # Find the points in the dataframe that are inside this grid square
    points_in_square = gdf[gdf.geometry.within(grid_row.geometry)]
    
    # Get the Pres_Esp1 value for the points inside this square
    pres_esp1_values = points_in_square['Pres_Esp1']
    
    # If any point has Pres_Esp1 == 1, color the grid square red, otherwise yellow
    if (pres_esp1_values == 1).any():
        grid_shapefile[grid_shapefile['LABEL'] == label].plot(ax=ax, edgecolor='black', facecolor='red', alpha=0.5)  # Red if Pres_Esp1 == 1
    else:
        grid_shapefile[grid_shapefile['LABEL'] == label].plot(ax=ax, edgecolor='black', facecolor='yellow', alpha=0.5)  # Yellow if no Pres_Esp1 == 1

# 8. Customize the plot
ax.set_title("Grid Square Coloring Based on Species Presence (Pres_Esp1)", fontsize=16)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# 9. Show the plot
plt.show()
