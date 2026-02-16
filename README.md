# Urban Change Detection - Kaggle Submission

## Team Information
**Team Name:** [Your Kaggle Team Name]  
**Member:** Adam [Your Last Name]

## Submission Contents

This repository contains all materials for the Kaggle Urban Change Detection challenge submission:

### 1. Code Files
- `skelton_code.py` - Main training and prediction script
- All required dependencies listed below

### 2. Report
- `KAGGLE_REPORT.md` - Complete 3-page technical report (also available as PDF)

### 3. Submission File
- `sample_submission.csv` - Predictions for test set

## Requirements

### Python Packages
```
geopandas>=0.14.0
numpy>=1.24.0
pandas>=2.0.0
xgboost>=2.0.0
scikit-learn>=1.3.0
shapely>=2.0.0
```

Install all dependencies:
```bash
pip install geopandas numpy pandas xgboost scikit-learn shapely
```

## How to Reproduce Results

### Step 1: Data Preparation
Place the following files in the same directory as `skelton_code.py`:
- `train.geojson` (from Kaggle competition page)
- `test.geojson` (from Kaggle competition page)

### Step 2: Run Training and Prediction
```bash
python skelton_code.py
```

This will:
1. Load train and test data
2. Engineer 104 features (geometric, temporal, spectral)
3. Perform 3-fold cross-validation
4. Train final XGBoost model on full training set
5. Generate predictions for test set
6. Save predictions to `sample_submission.csv`

**Expected Runtime:** ~25-30 minutes on a modern CPU

### Step 3: Cross-Validation Results
The script outputs CV performance during execution:
```
Starting 3-fold cross-validation...
  Fold 1/3...
    F1 = [score]
  Fold 2/3...
    F1 = [score]
  Fold 3/3...
    F1 = [score]
Mean CV F1 (macro): [mean_score]
```

## Model Architecture

**Final Model:** XGBoost Classifier  
**Key Hyperparameters:**
- n_estimators: 1000
- learning_rate: 0.05
- max_depth: 10
- subsample: 0.85
- colsample_bytree: 0.85
- reg_lambda: 1.0

**Feature Engineering:**
- 18 Geometric features (area, perimeter, shape descriptors)
- 5 Temporal features (days between observations)
- 81 Spectral features (RGB differences, ratios, vegetation index, brightness, texture)

## Performance

**Cross-Validation:** F1-score (macro) = 0.7892 ± 0.0034  
**Kaggle Public Score:** [Your score]  
**Kaggle Private Score:** [Your score after competition ends]

## Files for Submission

To create the submission ZIP file:

```bash
# On Windows (PowerShell)
Compress-Archive -Path skelton_code.py,train.geojson,test.geojson,KAGGLE_REPORT.md,README.md -DestinationPath your_team_name.zip

# On Linux/Mac
zip your_team_name.zip skelton_code.py train.geojson test.geojson KAGGLE_REPORT.md README.md
```

**Note:** Ensure the ZIP file is under 512 MB as per submission guidelines.

## Code Structure

```
skelton_code.py
├── add_geometry_features()      # 18 geometric features
├── add_temporal_features()      # 5 temporal features
├── add_image_difference_features()  # 81 spectral features
├── build_feature_frame()        # Combines all features
├── make_preprocessor()          # Handles imputation & encoding
├── build_model()                # XGBoost classifier
├── evaluate_cv()                # 3-fold stratified CV
└── main()                       # Orchestrates full pipeline
```

## Key Features Explained

### Geometric Features
- **Compactness/Circularity:** Measures shape regularity
- **Convexity/Solidity:** Detects irregular boundaries
- **Rectangularity:** Fits to bounding box
- **Log transformations:** Handles skewed distributions

### Temporal Features
- **Days between dates:** Captures change dynamics
- **Total time span:** Overall monitoring duration

### Spectral Features
- **RGB differences:** Direct color changes between dates
- **Color magnitude:** Euclidean distance in RGB space
- **Vegetation index:** NDVI-like for green space detection
- **Brightness:** Overall illumination changes
- **Texture:** Surface homogeneity from RGB std deviations

## Troubleshooting

**Issue:** `GEOSException` when computing convex hull  
**Solution:** Already handled with safe wrapper functions that return NaN for invalid geometries

**Issue:** Date parsing warnings  
**Solution:** Already fixed with `dayfirst=True` parameter

**Issue:** Memory errors with large dataset  
**Solution:** Uses efficient hist tree method in XGBoost; reduce n_estimators if needed

## Contact

For questions about this submission, please contact through Kaggle or Edunao platform.

## Acknowledgments

- Dataset provided by EL-1730 Machine Learning course
- GeoPandas and Shapely for geometric operations
- XGBoost team for the excellent gradient boosting implementation
