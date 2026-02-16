"""
Baseline feature engineering + model training for the challenge.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CHANGE_TYPE_MAP = {
       "Demolition": 0,
       "Road": 1,
       "Residential": 2,
       "Commercial": 3,
       "Industrial": 4,
       "Mega Projects": 5,
}


def add_geometry_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
       """Create numeric geometry-derived features."""
       if gdf.crs is not None and gdf.crs.is_geographic:
              gdf = gdf.to_crs(3857)
       geom = gdf.geometry
       
       # Safely compute geometry properties, replacing errors with NaN.
       with np.errstate(divide='ignore', invalid='ignore'):
              area = pd.Series(geom.area, index=gdf.index).astype(float)
              perimeter = pd.Series(geom.length, index=gdf.index).astype(float)
              bbox = geom.bounds
              bbox_width = (bbox["maxx"] - bbox["minx"]).astype(float)
              bbox_height = (bbox["maxy"] - bbox["miny"]).astype(float)
              bbox_area = bbox_width * bbox_height
              compactness = (4.0 * np.pi * area) / (perimeter * perimeter + 1e-9)
              compacity = compactness
              aspect_ratio = bbox_width / (bbox_height + 1e-9)
              centroid = geom.centroid
              centroid_x = pd.Series(centroid.x, index=gdf.index).astype(float)
              centroid_y = pd.Series(centroid.y, index=gdf.index).astype(float)
              
              # Additional geometric properties for irregular polygons
              # Safely compute convex hull (handle invalid geometries)
              def safe_convex_hull_area(g):
                     try:
                            if g is None or g.is_empty:
                                   return np.nan
                            return g.convex_hull.area
                     except:
                            return np.nan
              
              convex_hull_area = pd.Series([safe_convex_hull_area(g) for g in geom], 
                                           index=gdf.index).astype(float)
              convexity = area / (convex_hull_area + 1e-9)
              
              # Perimeter-to-area ratio (irregularity indicator)
              perimeter_area_ratio = perimeter / (area + 1e-9)
              
              # Shape complexity: number of vertices (for polygons)
              def safe_num_vertices(g):
                     try:
                            if hasattr(g, 'exterior') and g.exterior is not None:
                                   return len(g.exterior.coords)
                            return 0
                     except:
                            return 0
              
              num_vertices = pd.Series([safe_num_vertices(g) for g in geom], 
                                      index=gdf.index).astype(float)
              
              # Circularity: how close to a circle
              circularity = (4.0 * np.pi * area) / (perimeter * perimeter + 1e-9)
              
              # Log transformations for skewed distributions
              log_area = np.log1p(area)
              log_perimeter = np.log1p(perimeter)
              
              # Rectangularity: how well the shape fits its bounding box
              rectangularity = area / (bbox_area + 1e-9)
              
              # Solidity: ratio of area to convex hull area (another measure)
              solidity = area / (convex_hull_area + 1e-9)
              
              # Elongation: ratio of bbox dimensions
              elongation = bbox_height / (bbox_width + 1e-9)

       feat_df = pd.DataFrame(
              {
                     "geom_area": area,
                     "geom_perimeter": perimeter,
                     "geom_compactness": compactness,
                     "geom_compacity": compacity,
                     "geom_bbox_area": bbox_area.astype(float),
                     "geom_aspect_ratio": aspect_ratio.astype(float),
                     "geom_centroid_x": centroid_x,
                     "geom_centroid_y": centroid_y,
                     "geom_convexity": convexity,
                     "geom_convex_hull_area": convex_hull_area,
                     "geom_perimeter_area_ratio": perimeter_area_ratio,
                     "geom_num_vertices": num_vertices,
                     "geom_circularity": circularity,
                     "geom_log_area": log_area,
                     "geom_log_perimeter": log_perimeter,
                     "geom_rectangularity": rectangularity,
                     "geom_solidity": solidity,
                     "geom_elongation": elongation,
              },
              index=gdf.index,
       )
       feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
       return feat_df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
       """Create features from date columns (days between consecutive dates)."""
       date_cols = [c for c in df.columns if c.startswith('date') and c != 'date0']
       
       if not date_cols:
              return pd.DataFrame(index=df.index)
       
       # Convert date columns to datetime
       temporal_feats = {}
       all_date_cols = ['date0'] + date_cols
       
       for col in all_date_cols:
              if col in df.columns:
                     try:
                            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                     except:
                            pass
       
       # Calculate days between consecutive dates
       for i in range(len(all_date_cols) - 1):
              date_col1 = all_date_cols[i]
              date_col2 = all_date_cols[i + 1]
              
              if date_col1 in df.columns and date_col2 in df.columns:
                     days_diff = (df[date_col2] - df[date_col1]).dt.days
                     temporal_feats[f'days_between_{date_col1}_{date_col2}'] = days_diff
       
       # Total time span (date0 to last date)
       if 'date0' in df.columns and all_date_cols[-1] in df.columns:
              total_span = (df[all_date_cols[-1]] - df['date0']).dt.days
              temporal_feats['total_time_span'] = total_span
       
       return pd.DataFrame(temporal_feats, index=df.index)


def add_image_difference_features(df: pd.DataFrame) -> pd.DataFrame:
       """Create features from image color changes between dates."""
       image_feats = {}
       
       # RGB mean differences between consecutive dates
       for i in range(1, 5):
              for color in ['red', 'green', 'blue']:
                     col1 = f'img_{color}_mean_date{i}'
                     col2 = f'img_{color}_mean_date{i+1}'
                     
                     if col1 in df.columns and col2 in df.columns:
                            image_feats[f'{color}_diff_date{i}_to_{i+1}'] = df[col2] - df[col1]
                            image_feats[f'{color}_abs_diff_date{i}_to_{i+1}'] = np.abs(df[col2] - df[col1])
       
       # Total color change magnitude
       for i in range(1, 5):
              cols1 = [f'img_{c}_mean_date{i}' for c in ['red', 'green', 'blue']]
              cols2 = [f'img_{c}_mean_date{i+1}' for c in ['red', 'green', 'blue']]
              
              if all(c in df.columns for c in cols1 + cols2):
                     euclidean_dist = np.sqrt(
                            (df[cols2[0]] - df[cols1[0]])**2 +
                            (df[cols2[1]] - df[cols1[1]])**2 +
                            (df[cols2[2]] - df[cols1[2]])**2
                     )
                     image_feats[f'color_magnitude_change_date{i}_to_{i+1}'] = euclidean_dist
       
       # RGB ratios and interactions
       for i in range(1, 6):
              red_col = f'img_red_mean_date{i}'
              green_col = f'img_green_mean_date{i}'
              blue_col = f'img_blue_mean_date{i}'
              red_std_col = f'img_red_std_date{i}'
              green_std_col = f'img_green_std_date{i}'
              blue_std_col = f'img_blue_std_date{i}'
              
              if all(c in df.columns for c in [red_col, green_col, blue_col]):
                     image_feats[f'rg_ratio_date{i}'] = df[red_col] / (df[green_col] + 1e-9)
                     image_feats[f'rb_ratio_date{i}'] = df[red_col] / (df[blue_col] + 1e-9)
                     image_feats[f'gb_ratio_date{i}'] = df[green_col] / (df[blue_col] + 1e-9)
                     
                     # NDVI-like vegetation index (green-red normalized)
                     image_feats[f'vegetation_index_date{i}'] = (df[green_col] - df[red_col]) / (df[green_col] + df[red_col] + 1e-9)
                     
                     # Brightness
                     image_feats[f'brightness_date{i}'] = (df[red_col] + df[green_col] + df[blue_col]) / 3.0
              
              # Standard deviation features (texture/heterogeneity)
              if all(c in df.columns for c in [red_std_col, green_std_col, blue_std_col]):
                     image_feats[f'texture_mean_date{i}'] = (df[red_std_col] + df[green_std_col] + df[blue_std_col]) / 3.0
                     image_feats[f'texture_max_date{i}'] = df[[red_std_col, green_std_col, blue_std_col]].max(axis=1)
       
       # Change in brightness between dates
       for i in range(1, 5):
              bright1 = f'brightness_date{i}'
              bright2 = f'brightness_date{i+1}'
              if bright1 in image_feats and bright2 in image_feats:
                     image_feats[f'brightness_change_date{i}_to_{i+1}'] = image_feats[bright2] - image_feats[bright1]
       
       return pd.DataFrame(image_feats, index=df.index)


def build_feature_frame(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
       """Combine non-geometry columns with engineered geometry features."""
       base_df = gdf.drop(columns=["geometry"], errors="ignore").copy()
       geom_df = add_geometry_features(gdf)
       temporal_df = add_temporal_features(base_df)
       image_df = add_image_difference_features(base_df)
       return pd.concat([base_df, geom_df, temporal_df, image_df], axis=1)


def split_feature_types(df: pd.DataFrame):
       numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
       categorical_cols = [c for c in df.columns if c not in numeric_cols]
       return numeric_cols, categorical_cols


def make_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
       numeric_pipe = Pipeline(
              steps=[
                     ("imputer", SimpleImputer(strategy="median")),
              ]
       )
       categorical_pipe = Pipeline(
              steps=[
                     ("imputer", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore")),
              ]
       )
       return ColumnTransformer(
              transformers=[
                     ("num", numeric_pipe, numeric_cols),
                     ("cat", categorical_pipe, categorical_cols),
              ]
       )


def evaluate_cv(model, x, y, folds=3) -> float:
       print(f"Starting {folds}-fold cross-validation...")
       skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
       scores = []
       for i, (train_idx, val_idx) in enumerate(skf.split(x, y)):
              x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
              y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
              print(f"  Fold {i+1}/{folds}...")
              model.fit(x_train, y_train)
              pred = model.predict(x_val)
              score = f1_score(y_val, pred, average="macro")
              scores.append(score)
              print(f"    F1 = {score:.4f}")
       cv_mean = float(np.mean(scores))
       print(f"Mean CV F1 (macro): {cv_mean:.4f}")
       return cv_mean


def build_model(preprocessor: ColumnTransformer) -> Pipeline:
       clf = XGBClassifier(
              n_estimators=1000,
              learning_rate=0.05,
              max_depth=10,
              min_child_weight=2,
              subsample=0.85,
              colsample_bytree=0.85,
              colsample_bylevel=0.85,
              reg_alpha=0.05,
              reg_lambda=1.0,
              gamma=0.05,
              random_state=42,
              n_jobs=-1,
              tree_method='hist',
              enable_categorical=True,
       )
       return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def build_submission(pred_y: np.ndarray, test_df: gpd.GeoDataFrame) -> pd.DataFrame:
       ids = np.arange(0, len(test_df))
       return pd.DataFrame({"Id": ids, "change_type": pred_y})


def main() -> None:
       print("Loading data...")
       train_gdf = gpd.read_file("train.geojson")
       test_gdf = gpd.read_file("test.geojson")
       print(f"Train size: {len(train_gdf)}, Test size: {len(test_gdf)}")

       print("Building features...")
       train_y = train_gdf["change_type"].map(CHANGE_TYPE_MAP)
       train_x = build_feature_frame(train_gdf)
       train_x = train_x.drop(columns=["change_type"], errors="ignore")
       test_x = build_feature_frame(test_gdf)
       test_x = test_x.drop(columns=["change_type"], errors="ignore")
       print(f"Feature shape: {train_x.shape}")

       numeric_cols, categorical_cols = split_feature_types(train_x)
       print(f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
       preprocessor = make_preprocessor(numeric_cols, categorical_cols)
       model = build_model(preprocessor)

       cv_score = evaluate_cv(model, train_x, train_y, folds=3)

       print("\nTraining final model...")
       model.fit(train_x, train_y)
       print("Generating predictions...")
       pred_y = model.predict(test_x)
       print(f"Predictions shape: {pred_y.shape}")

       submission = build_submission(pred_y, test_gdf)
       submission.to_csv("sample.csv", index=False)
       print("\nDone! Saved to sample.csv")


if __name__ == "__main__":
       main()