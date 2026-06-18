import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class AIDataEngine:
    def __init__(self, data):
        self.raw_data = data
        self.scaler = StandardScaler()

    def preprocess(self):
        """
        Extracts numeric columns and handles scaling. 
        Critical for multi-modal data like Audio (MFCCs) and Video pixels.
        """
        try:
            # Select only numeric data
            numeric_df = self.raw_data.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                return np.array([])
            
            # Fill NaN with median (much safer than 0 for sensor data/pixels)
            numeric_df = numeric_df.fillna(numeric_df.median())
                
            return self.scaler.fit_transform(numeric_df)
        except Exception:
            return np.array([])

    def run_hi_dbsf(self, scaled_data):
        """
        Hybrid Intelligence - Density Based Sigma Filter (HI-DBSF).
        Optimized to prevent RAM crashes during large file processing.
        """
        if scaled_data is None or scaled_data.size == 0:
            return self.raw_data

        n_samples = len(scaled_data)
        
        # --- 1. THE COMPLEXITY GUARD ---
        # DBSCAN memory usage is O(n^2). We bypass it for very large datasets
        # to ensure your laptop doesn't freeze during the presentation.
        use_dbscan = n_samples < 40000 

        # --- 2. ADAPTIVE ANOMALY DETECTION (Isolation Forest) ---
        # We use 'auto' contamination so the AI decides how much noise exists.
        # n_jobs=-1 uses all CPU cores for maximum speed.
        iso = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
        iso_preds = iso.fit_predict(scaled_data)

        # --- 3. LOCAL DENSITY CHECK (DBSCAN) ---
        if use_dbscan:
            # We use a slightly larger eps to avoid over-cleaning dense clusters
            db = DBSCAN(eps=0.6, min_samples=3).fit_predict(scaled_data)
            
            # HYBRID LOGIC: 
            # A data point is kept if EITHER Isolation Forest OR DBSCAN thinks it's valid.
            # This 'Union' logic prevents 'over-cleaning' your project data.
            mask = (iso_preds != -1) | (db != -1)
        else:
            # Speed mode for Video/High-Res Images
            mask = (iso_preds != -1)

        # --- 4. DATA INTEGRITY RECOVERY ---
        # If the AI accidentally flags everything as noise, we force-recovery 
        # of the 'cleanest' 20% to ensure the user gets a working file back.
        if not any(mask):
            scores = iso.decision_function(scaled_data)
            threshold = np.percentile(scores, 20)
            mask = scores >= threshold

        # Apply the boolean mask to the original raw dataframe
        return self.raw_data[mask]
