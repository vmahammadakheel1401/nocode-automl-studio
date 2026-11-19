import { AutoMLRunResult } from './types';

export function generatePythonScript(run: AutoMLRunResult): string {
  const { preprocessor, winner, fileName, targetColumn, taskType } = run;

  const numericFeatures = preprocessor.specs
    .filter((s) => s.type === 'numeric')
    .map((s) => s.rawName);

  const categoricalFeatures = preprocessor.specs
    .filter((s) => s.type === 'categorical')
    .map((s) => s.rawName);

  const droppedIds = preprocessor.droppedIdColumns;

  const formatPyList = (arr: string[]) =>
    arr.length === 0
      ? '[]'
      : `[\n    ${arr.map((s) => JSON.stringify(s)).join(',\n    ')}\n]`;

  const sklearnModelImport = (() => {
    switch (winner.sklearnName) {
      case 'Ridge':
        return 'from sklearn.linear_model import Ridge';
      case 'LogisticRegression':
        return 'from sklearn.linear_model import LogisticRegression';
      case 'DecisionTreeRegressor':
        return 'from sklearn.tree import DecisionTreeRegressor';
      case 'DecisionTreeClassifier':
        return 'from sklearn.tree import DecisionTreeClassifier';
      case 'RandomForestRegressor':
        return 'from sklearn.ensemble import RandomForestRegressor';
      case 'RandomForestClassifier':
        return 'from sklearn.ensemble import RandomForestClassifier';
      case 'GradientBoostingRegressor':
        return 'from sklearn.ensemble import GradientBoostingRegressor';
      case 'GradientBoostingClassifier':
      default:
        return 'from sklearn.ensemble import GradientBoostingClassifier';
    }
  })();

  const modelInitExpr = (() => {
    switch (winner.sklearnName) {
      case 'Ridge':
        return 'Ridge(alpha=1.5, random_state=42)';
      case 'LogisticRegression':
        return 'LogisticRegression(max_iter=1000, C=1.0, random_state=42)';
      case 'DecisionTreeRegressor':
        return 'DecisionTreeRegressor(max_depth=6, min_samples_leaf=2, random_state=42)';
      case 'DecisionTreeClassifier':
        return 'DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, random_state=42)';
      case 'RandomForestRegressor':
        return 'RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)';
      case 'RandomForestClassifier':
        return 'RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)';
      case 'GradientBoostingRegressor':
        return 'GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)';
      case 'GradientBoostingClassifier':
      default:
        return 'GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)';
    }
  })();

  const metricsImport =
    taskType === 'regression'
      ? 'from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error'
      : 'from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report';

  const stratifyLine =
    taskType === 'classification'
      ? '    stratify=y,\n'
      : '';

  const evalBlock =
    taskType === 'regression'
      ? `y_pred = pipeline.predict(X_test)
print(f"R2 Score : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE     : {root_mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE      : {mean_absolute_error(y_test, y_pred):.2f}")`
      : `y_pred = pipeline.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))`;

  return `"""
AutoML Studio — Generated Scikit-Learn Pipeline
Dataset: ${fileName}
Task: ${taskType === 'regression' ? 'Regression' : 'Classification'}
Target Column: ${targetColumn}
Winning Model: ${winner.name} (${winner.sklearnName})
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
${sklearnModelImport}
${metricsImport}

# 1. Load dataset and drop missing targets / identifier columns
df = pd.read_csv(${JSON.stringify(fileName)})
TARGET_COL = ${JSON.stringify(targetColumn)}
DROPPED_ID_COLS = ${formatPyList(droppedIds)}

df = df.dropna(subset=[TARGET_COL])
X = df.drop(columns=[c for c in [TARGET_COL, *DROPPED_ID_COLS] if c in df.columns])
y = df[TARGET_COL]

# 2. Define feature schema
NUMERIC_FEATURES = ${formatPyList(numericFeatures)}
CATEGORICAL_FEATURES = ${formatPyList(categoricalFeatures)}

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ]
)

# 3. Build end-to-end training pipeline (${winner.name})
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", ${modelInitExpr}),
    ]
)

# 4. 80/20 Train/Test Split & Fit
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
${stratifyLine})

pipeline.fit(X_train, y_train)

# 5. Evaluate on holdout test split
${evalBlock}

# 6. Persist trained pipeline for production inference
joblib.dump(pipeline, "automl_winning_pipeline.joblib")
print("Saved trained pipeline to automl_winning_pipeline.joblib")
`;
}

export function generateModelStateJson(run: AutoMLRunResult): string {
  const artifact = {
    schemaVersion: '1.0.0',
    generatedBy: 'AutoML Studio (Client-Side Engine)',
    createdAt: run.createdAt,
    dataset: {
      name: run.datasetName,
      fileName: run.fileName,
      taskType: run.taskType,
      targetColumn: run.targetColumn,
      trainRows: run.preprocessor.trainCount,
      testRows: run.preprocessor.testCount,
      droppedIdColumns: run.preprocessor.droppedIdColumns,
    },
    winningModel: {
      key: run.winner.key,
      name: run.winner.name,
      sklearnEquivalent: run.winner.sklearnName,
      primaryScore: run.winner.primaryScore,
      metrics: run.winner.metrics,
      trainingTimeMs: run.winner.trainingTimeMs,
      featureImportances: run.winner.featureImportances,
      weights: run.winner.weights,
    },
    leaderboard: run.models.map((m, idx) => ({
      rank: idx + 1,
      name: m.name,
      family: m.family,
      primaryScore: m.primaryScore,
      metrics: m.metrics,
      trainingTimeMs: m.trainingTimeMs,
      isWinner: m.isWinner,
    })),
    preprocessing: {
      encodedFeatureCount: run.preprocessor.encodedFeatureCount,
      classLabels: run.preprocessor.classLabels ?? null,
      features: run.preprocessor.specs,
    },
  };

  return JSON.stringify(artifact, null, 2);
}
