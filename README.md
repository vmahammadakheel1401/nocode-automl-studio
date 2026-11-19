# AutoML Studio

**100% Client-Side • Zero APIs**

AutoML Studio is a browser-based, no-code machine learning platform that lets you audit datasets, train models, compare performance, run real-time what-if predictions, and export production-ready Python code—all locally in your browser with zero external API calls.

---

## ✨ Key Features

- **1. Upload & Audit:** Upload custom CSV datasets or explore built-in sample data (e.g., `house_prices.csv`).
- **2. Configure & Train:** Automatically handles preprocessing—including median/most-frequent imputation, standard scaling for numerical features, and one-hot encoding for categorical variables—with an 80/20 train/test split.
- **3. Model Leaderboard:** Train and compare multiple machine learning models across standard evaluation metrics ($R^2$, RMSE, MAE) to identify the top performer.
- **4. Live What-If Predictor:** Interactively adjust numeric sliders and categorical dropdowns to see real-time target predictions alongside dataset min/max bounds.
- **5. One-Click Export:**
  - **Python Script (`.py`):** Download a clean, reproducible `scikit-learn` pipeline script tailored to your dataset and winning model, ready for `joblib` serialization.
  - **Model State (`.json`):** Export trained browser model weights for lightweight portability.

---

## 📸 Screenshots

### 1. Upload & Data Audit
<img width="1913" height="957" alt="Screenshot 2026-11-19 144508" src="https://github.com/user-attachments/assets/faa9018f-1272-4cde-8dc5-c3b6e96bf172" />


### 2. Model Leaderboard
<img width="1916" height="932" alt="Screenshot 2026-11-19 144524" src="https://github.com/user-attachments/assets/21862d7b-b33b-457c-b25f-cfcadb2c3a43" />


### 3. Live What-If Predictor & Code Export
<img width="1919" height="945" alt="Screenshot 2026-11-19 144545" src="https://github.com/user-attachments/assets/f8df8808-5737-4145-a815-9d7bfb357b87" />


---predictor (<img width="1915" height="950" alt="Screenshot 2026-11-19 144613" src="https://github.com/user-attachments/assets/74fa2f6b-1a0e-4bd8-80b6-c7f5b06e9126" />
)

## 🛠️ Tech Stack & Generated Pipeline

While training and inference run 100% client-side in the browser, exported scripts target standard Python data science tooling:

- **Preprocessing:** `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`
- **Modeling:** `scikit-learn` (e.g., `Ridge` Regression)
- **Evaluation & Persistence:** `r2_score`, `root_mean_squared_error`, `mean_absolute_error`, `joblib`

---

## 🚀 Getting Started

### Using the Web App
1. Open the [AutoML Studio Web App](https://nocode-automl-studio-five.vercel.app/) in your browser.
2. Click **Load Sample Data** (or upload your own `.csv` file) in **Step 1: Upload & Audit**.
3. Select your target variable and train models in **Step 2: Configure & Train**.
4. Review top-performing models in **Step 3: Model Leaderboard**.
5. Test scenarios in **Step 4: Predict & Export** and download the `.py` script or `.json` weights.

### Running an Exported Python Pipeline
If you export the generated `scikit-learn` script, install the required dependencies and run it locally:

```bash
pip install pandas scikit-learn joblib
python train_pipeline.py
