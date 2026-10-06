import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
import joblib
import os
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier


df = pd.read_csv("data/loan_dataset.csv")

print(df.head())
print(df.shape)
print(df.info())

X = df.drop("Default", axis=1)
y = df["Default"]

X_temp, X_test, y_temp, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_temp,y_temp,test_size=0.25,random_state=42,stratify=y_temp)

print("\nDataset Split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nTarget distribution:")
print("Train:")
print(y_train.value_counts(normalize=True))

print("\nValidation:")
print(y_val.value_counts(normalize=True))

print("\nTest:")
print(y_test.value_counts(normalize=True))

numeric_columns = X_train.select_dtypes(include=["number"]).columns

X_train_num = X_train[numeric_columns].copy()
X_val_num = X_val[numeric_columns].copy()

train_medians = X_train_num.median()

X_train_num = X_train_num.fillna(train_medians)
X_val_num = X_val_num.fillna(train_medians)

depths = [1, 3, 5, 10, None]

train_scores = []
val_scores = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train_num, y_train)

    train_prediction = model.predict(X_train_num)
    val_prediction = model.predict(X_val_num)

    train_accuracy = accuracy_score(y_train, train_prediction)
    val_accuracy = accuracy_score(y_val, val_prediction)

    train_scores.append(train_accuracy)
    val_scores.append(val_accuracy)

    print(
        f"Depth {depth}: "
        f"Train Accuracy = {train_accuracy:.4f}, "
        f"Validation Accuracy = {val_accuracy:.4f}"
    )

plt.figure(figsize=(8, 5))

plt.plot(
    [str(d) for d in depths],
    train_scores,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    [str(d) for d in depths],
    val_scores,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Decision Tree Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Overfitting")
plt.legend()
plt.grid(True)

plt.savefig("outputs/overfitting.png")
plt.show()

logistic_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

cv_scores = cross_val_score(
    logistic_pipeline,
    X_train_num,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\n5-Fold Cross-Validation Results:")
print("Fold scores:", cv_scores)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

scaler_wrong = StandardScaler()

X_scaled_wrong = scaler_wrong.fit_transform(X_train_num)

X_train_wrong, X_val_wrong, y_train_wrong, y_val_wrong = train_test_split(
    X_scaled_wrong,y_train,test_size=0.25,random_state=42,stratify=y_train)

model_wrong = LogisticRegression(max_iter=1000, random_state=42)

model_wrong.fit(X_train_wrong, y_train_wrong)

wrong_prediction = model_wrong.predict(X_val_wrong)

wrong_accuracy = accuracy_score(y_val_wrong,wrong_prediction)

print("\nWrong Scaling Accuracy:", round(wrong_accuracy, 4))

X_train_correct, X_val_correct, y_train_correct, y_val_correct = train_test_split(
 X_train_num,y_train,test_size=0.25,random_state=42,stratify=y_train)

scaler_correct = StandardScaler()

X_train_scaled = scaler_correct.fit_transform(X_train_correct)
X_val_scaled = scaler_correct.transform(X_val_correct)

model_correct = LogisticRegression(max_iter=1000,random_state=42)

model_correct.fit(X_train_scaled,y_train_correct)

correct_prediction = model_correct.predict(X_val_scaled)

correct_accuracy = accuracy_score(y_val_correct,correct_prediction)

print("Correct Scaling Accuracy:", round(correct_accuracy, 4))
print("\nData Leakage Explanation:")
print("1. Scaling before splitting allows information from validation data to influence preprocessing.")
print("2. This is called data leakage and can give an overly optimistic result.")
print("3. The correct approach is to fit the scaler only on training data.")
print("4. Validation and test data should only be transformed using the fitted training scaler.")


print("\n--- Supervised vs Unsupervised Learning ---")

problems = {
    "Spam detection": "Supervised Learning",
    "Customer segmentation": "Unsupervised Learning",
    "House price prediction": "Supervised Learning",
    "Anomaly detection": "Unsupervised Learning"
}

for problem, learning_type in problems.items():
    print(f"{problem}: {learning_type}")


print("\n--- 7-Step Machine Learning Workflow ---")

workflow = [
    "1. Collect the data",
    "2. Understand and explore the data",
    "3. Clean and preprocess the data",
    "4. Perform feature engineering and feature selection",
    "5. Split the data into train, validation and test sets",
    "6. Train, evaluate and tune the machine learning model",
    "7. Test the final model and deploy/monitor it"
]

for step in workflow:
    print(step)
print("\n--- Numerical Columns ---")
numeric_columns = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
print(numeric_columns)

print("\n--- Categorical Columns ---")
categorical_columns = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
print(categorical_columns)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Data Types ---")
print(df.dtypes)

def create_features(data):
    data = data.copy()

    data["Income_Loan_Ratio"] = (
        data["Income"] / (data["LoanAmount"] + 1)
    )

    data["Credit_DTI_Ratio"] = (
        data["CreditScore"] / (data["DTIRatio"] + 1)
    )

    data["Log_Income"] = np.log1p(data["Income"])

    return data


X_train_fe = create_features(X_train)
X_val_fe = create_features(X_val)

print("\nNew Features Created:")
print([
    "Income_Loan_Ratio",
    "Credit_DTI_Ratio",
    "Log_Income"
])

print("\nNew Training Shape:", X_train_fe.shape)
print("New Validation Shape:", X_val_fe.shape)

X_train_fe = X_train_fe.drop("LoanID", axis=1)
X_val_fe = X_val_fe.drop("LoanID", axis=1)

numeric_features = X_train_fe.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

ordinal_features = ["Education"]

categorical_features = [
    col for col in X_train_fe.select_dtypes(
        include=["object", "category"]
    ).columns
    if col not in ordinal_features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal_encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("ord", ordinal_pipeline, ordinal_features),
    ("cat", categorical_pipeline, categorical_features)
])

print("\nPreprocessing Pipeline Created Successfully!")

print("\nNumerical Features:")
print(numeric_features)

print("\nOrdinal Feature:")
print(ordinal_features)

print("\nCategorical Features:")
print(categorical_features)

classification_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("\nRunning 5-Fold Cross-Validation...")

pipeline_cv_scores = cross_val_score(
    classification_pipeline,
    X_train_fe,
    y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("\nPipeline CV Scores:")
print(pipeline_cv_scores)

print(
    f"Pipeline CV Accuracy: "
    f"{pipeline_cv_scores.mean():.4f}"
)

print(
    f"Pipeline CV Standard Deviation: "
    f"{pipeline_cv_scores.std():.4f}"
)

classification_pipeline.fit(X_train_fe, y_train)

os.makedirs("models", exist_ok=True)

model_path = "models/loan_pipeline.joblib"

joblib.dump(classification_pipeline, model_path)

print("\nPipeline saved successfully:")
print(model_path)

loaded_pipeline = joblib.load(model_path)

print("Pipeline loaded successfully!")

new_rows = X_val.iloc[:3].copy()

new_rows_fe = create_features(new_rows)

predictions = loaded_pipeline.predict(new_rows_fe)

print("\nPredictions for 3 new rows:")
print(predictions)

logistic_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

logistic_model.fit(X_train_fe, y_train)
logistic_pred = logistic_model.predict(X_val_fe)

# Validation accuracy
logistic_accuracy = accuracy_score(y_val,logistic_pred)

print("\n--- Logistic Regression ---")
print(f"Validation Accuracy: {logistic_accuracy:.4f}")


decision_tree_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ))
])

decision_tree_model.fit(X_train_fe, y_train)
decision_tree_pred = decision_tree_model.predict(X_val_fe)

decision_tree_accuracy = accuracy_score(y_val,decision_tree_pred)

print("\n--- Decision Tree ---")
print(f"Validation Accuracy: {decision_tree_accuracy:.4f}")

from sklearn.ensemble import RandomForestClassifier

random_forest_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        oob_score=True,
        n_jobs=-1
    ))
])

random_forest_model.fit(X_train_fe, y_train)
random_forest_pred = random_forest_model.predict(X_val_fe)

random_forest_accuracy = accuracy_score(y_val,random_forest_pred)

random_forest_oob = (
    random_forest_model
    .named_steps["model"]
    .oob_score_
)

print("\n--- Random Forest ---")
print(f"Validation Accuracy: {random_forest_accuracy:.4f}")
print(f"OOB Score: {random_forest_oob:.4f}")

preprocessor.fit(X_train_fe, y_train)

X_train_xgb = preprocessor.transform(X_train_fe)
X_val_xgb = preprocessor.transform(X_val_fe)

xgb_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1
    ))
])

xgb_model.fit(
    X_train_fe,
    y_train,
    model__eval_set=[(X_val_xgb, y_val)],
    model__verbose=False
)

xgb_pred = xgb_model.predict(X_val_fe)

xgb_accuracy = accuracy_score(y_val,xgb_pred)

print("\n--- XGBoost ---")
print(f"Validation Accuracy: {xgb_accuracy:.4f}")

xgb_estimator = xgb_model.named_steps["model"]

print(f"Best Iteration: {xgb_estimator.best_iteration}")

results = {
    "Logistic Regression": logistic_accuracy,
    "Decision Tree": decision_tree_accuracy,
    "Random Forest": random_forest_accuracy,
    "XGBoost": xgb_accuracy
}

print("\n--- CLASSIFICATION RESULTS ---")

for model_name, score in results.items():
    print(f"{model_name}: {score:.4f}")

print("\n--- LOGISTIC REGRESSION PROBABILITIES ---")

five_rows = X_val_fe.iloc[:5]

probabilities = logistic_model.predict_proba(five_rows)
predictions = logistic_model.predict(five_rows)

print("Predict probabilities:")
print(probabilities)

print("\nPredicted classes:")
print(predictions)

print("\n--- RANDOM FOREST TOP 5 FEATURES ---")

rf_estimator = random_forest_model.named_steps["model"]

rf_importances = rf_estimator.feature_importances_

feature_names = (
    random_forest_model
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

rf_feature_importance = pd.Series(
    rf_importances,
    index=feature_names
).sort_values(ascending=False)

print(rf_feature_importance.head(5))

print("\n--- XGBOOST TOP 5 FEATURES ---")

xgb_estimator = xgb_model.named_steps["model"]

xgb_importances = xgb_estimator.feature_importances_

xgb_feature_importance = pd.Series(
    xgb_importances,
    index=feature_names
).sort_values(ascending=False)

print(xgb_feature_importance.head(5))

X_reg = df.drop(
    ["LoanAmount", "Default", "LoanID"],
    axis=1
)

y_reg = df["LoanAmount"]

X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

reg_numeric = X_reg_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

reg_categorical = X_reg_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

reg_preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        reg_numeric
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]),
        reg_categorical
    )
])


linear_model = Pipeline([
    ("preprocessing", reg_preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_reg_train, y_reg_train)
linear_pred = linear_model.predict(X_reg_val)

print("\n--- Linear Regression ---")
print(f"R² Score: {r2_score(y_reg_val, linear_pred):.4f}")


ridge_model = Pipeline([
    ("preprocessing", reg_preprocessor),
    ("model", Ridge(alpha=1.0))
])

ridge_model.fit(X_reg_train, y_reg_train)

ridge_pred = ridge_model.predict(X_reg_val)

print("\n--- Ridge Regression ---")
print(f"R² Score: {r2_score(y_reg_val, ridge_pred):.4f}")

lasso_model = Pipeline([
    ("preprocessing", reg_preprocessor),
    ("model", Lasso(alpha=0.1, max_iter=10000))
])

lasso_model.fit(X_reg_train, y_reg_train)

lasso_pred = lasso_model.predict(X_reg_val)

print("\n--- Lasso Regression ---")
print(f"R² Score: {r2_score(y_reg_val, lasso_pred):.4f}")

reg_feature_names = (
    linear_model
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

linear_coefficients = linear_model.named_steps["model"].coef_
ridge_coefficients = ridge_model.named_steps["model"].coef_
lasso_coefficients = lasso_model.named_steps["model"].coef_

coefficient_comparison = pd.DataFrame({
    "Feature": reg_feature_names,
    "Linear": linear_coefficients,
    "Ridge": ridge_coefficients,
    "Lasso": lasso_coefficients
})

print("\n--- Coefficient Comparison ---")
print(coefficient_comparison.head(10))

lasso_zero_count = np.sum(
    np.isclose(lasso_coefficients, 0)
)

print(
    f"\nLasso Zero Coefficients: "
    f"{lasso_zero_count}"
)

cluster_features = [
    "Income",
    "CreditScore",
    "LoanAmount"
]

X_cluster = df[cluster_features].copy()

X_cluster = X_cluster.fillna(X_cluster.median())

cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

sample_size = min(10000, len(X_cluster_scaled))

rng = np.random.RandomState(42)

sample_indices = rng.choice(
    len(X_cluster_scaled),
    size=sample_size,
    replace=False
)

X_cluster_sample = X_cluster_scaled[sample_indices]
k_values = range(2, 9)

inertia_values = []
silhouette_values = []

print("\n--- K-MEANS CLUSTERING ---")

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_cluster_scaled)

    inertia_values.append(kmeans.inertia_)

    sample_labels = cluster_labels[sample_indices]

    silhouette = silhouette_score(
        X_cluster_sample,
        sample_labels
    )

    silhouette_values.append(silhouette)

    print(
        f"k={k}: "
        f"Inertia={kmeans.inertia_:.2f}, "
        f"Silhouette={silhouette:.4f}"
    )

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Curve")
plt.grid(True)

plt.savefig("outputs/elbow_curve.png")
plt.show()


best_k = list(k_values)[
    np.argmax(silhouette_values)
]

print(
    f"\nBest k based on Silhouette Score: {best_k}"
)

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

final_labels = final_kmeans.fit_predict(
    X_cluster_scaled
)

clustered_data = df[cluster_features].copy()
clustered_data["Cluster"] = final_labels

print("\n--- Customer Segment Summary ---")

segment_summary = (
    clustered_data
    .groupby("Cluster")[cluster_features]
    .mean()
    .round(2)
)

print(segment_summary)


print("\n--- Cluster Sizes ---")

print(
    clustered_data["Cluster"]
    .value_counts()
    .sort_index()
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

models_predictions = {
    "Logistic Regression": logistic_pred,
    "Decision Tree": decision_tree_pred,
    "Random Forest": random_forest_pred,
    "XGBoost": xgb_pred
}

print("\n================ CLASSIFICATION METRICS ================")

for model_name, predictions in models_predictions.items():

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)
    cm = confusion_matrix(y_val, predictions)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

print("\n--- Why Accuracy Can Mislead ---")

print(
    "Accuracy can be misleading when the target classes are imbalanced."
)

print(
    "For example, if 90% of customers do not default, "
    "a model predicting 'no default' for everyone gets 90% accuracy."
)

print(
    "However, it would fail to identify most actual defaulters."
)

print(
    "Therefore, precision, recall, F1-score and ROC-AUC "
    "are also important for evaluating an imbalanced classification problem."
)

logistic_proba = logistic_model.predict_proba(X_val_fe)[:, 1]

random_forest_proba = random_forest_model.predict_proba(
    X_val_fe
)[:, 1]

xgb_proba = xgb_model.predict_proba(X_val_fe)[:, 1]

fpr_logistic, tpr_logistic, _ = roc_curve(
    y_val,
    logistic_proba
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_val,
    random_forest_proba
)

fpr_xgb, tpr_xgb, _ = roc_curve(
    y_val,
    xgb_proba
)

auc_logistic = roc_auc_score(
    y_val,
    logistic_proba
)

auc_rf = roc_auc_score(
    y_val,
    random_forest_proba
)

auc_xgb = roc_auc_score(
    y_val,
    xgb_proba
)


print("\n================ ROC-AUC ================")

print(f"Logistic Regression AUC: {auc_logistic:.4f}")
print(f"Random Forest AUC:       {auc_rf:.4f}")
print(f"XGBoost AUC:             {auc_xgb:.4f}")


plt.figure(figsize=(8, 6))

plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=f"Logistic Regression (AUC = {auc_logistic:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.3f})"
)

plt.plot(
    fpr_xgb,
    tpr_xgb,
    label=f"XGBoost (AUC = {auc_xgb:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)

plt.savefig("outputs/roc_curves.png")
plt.show()

xgb_probabilities = xgb_model.predict_proba(X_val_fe)[:, 1]

thresholds = np.arange(0.01, 1.00, 0.01)

selected_threshold = None
selected_precision = None
selected_recall = None

for threshold in thresholds:

    threshold_predictions = (
        xgb_probabilities >= threshold
    ).astype(int)

    recall = recall_score(
        y_val,
        threshold_predictions,
        zero_division=0
    )

    precision = precision_score(
        y_val,
        threshold_predictions,
        zero_division=0
    )

    if recall >= 0.75:
        selected_threshold = threshold
        selected_precision = precision
        selected_recall = recall
        break


print("\n================ THRESHOLD ANALYSIS ================")

print(
    f"Selected Threshold: {selected_threshold:.2f}"
)

print(
    f"Recall at Threshold: {selected_recall:.4f}"
)

print(
    f"Precision at Threshold: {selected_precision:.4f}"
)


print("\n================ HYPERPARAMETER TUNING ================")

tuning_sample_size = min(50000, len(X_train_fe))

X_tune = X_train_fe.iloc[:tuning_sample_size].copy()
y_tune = y_train.iloc[:tuning_sample_size].copy()

rf_pipeline_tuning = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        oob_score=True,
        n_jobs=-1
    ))
])

rf_param_grid = {
    "model__n_estimators": [100, 150],
    "model__max_depth": [5, 10],
    "model__min_samples_split": [2, 5]
}

rf_grid_search = GridSearchCV(
    estimator=rf_pipeline_tuning,
    param_grid=rf_param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearchCV for Random Forest...")

rf_grid_search.fit(X_tune, y_tune)

print("\nRandom Forest Best Parameters:")
print(rf_grid_search.best_params_)

print(
    f"Random Forest Best CV ROC-AUC: "
    f"{rf_grid_search.best_score_:.4f}"
)

xgb_pipeline_tuning = Pipeline([
    ("preprocessing", preprocessor),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

xgb_param_distributions = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}

xgb_random_search = RandomizedSearchCV(
    estimator=xgb_pipeline_tuning,
    param_distributions=xgb_param_distributions,
    n_iter=15,
    cv=3,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nRunning RandomizedSearchCV for XGBoost...")

xgb_random_search.fit(X_tune, y_tune)

print("\nXGBoost Best Parameters:")
print(xgb_random_search.best_params_)

print(
    f"XGBoost Best CV ROC-AUC: "
    f"{xgb_random_search.best_score_:.4f}"
)


linear_mae = mean_absolute_error(y_reg_val, linear_pred)

linear_rmse = np.sqrt(
    mean_squared_error(y_reg_val, linear_pred)
)

linear_r2 = r2_score(
    y_reg_val,
    linear_pred
)

mean_prediction = np.full(
    len(y_reg_val),
    y_reg_train.mean()
)

baseline_mae = mean_absolute_error(
    y_reg_val,
    mean_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_reg_val,
        mean_prediction
    )
)

baseline_r2 = r2_score(
    y_reg_val,
    mean_prediction
)

print("\nLinear Regression:")
print(f"MAE : {linear_mae:.4f}")
print(f"RMSE: {linear_rmse:.4f}")
print(f"R²  : {linear_r2:.4f}")

print("\nMean Baseline:")
print(f"MAE : {baseline_mae:.4f}")
print(f"RMSE: {baseline_rmse:.4f}")
print(f"R²  : {baseline_r2:.4f}")

classifier_auc_scores = {
    "Logistic Regression": auc_logistic,
    "Random Forest": auc_rf,
    "XGBoost": auc_xgb
}

best_model_name = max(
    classifier_auc_scores,
    key=classifier_auc_scores.get
)

best_auc = classifier_auc_scores[best_model_name]

print("\n================ BEST MODEL ================")

print(f"Best Model: {best_model_name}")
print(f"Validation ROC-AUC: {best_auc:.4f}")

X_test_fe = create_features(X_test)

if best_model_name == "Logistic Regression":
    best_model = logistic_model

elif best_model_name == "Random Forest":
    best_model = random_forest_model

else:
    best_model = xgb_model

test_predictions = best_model.predict(X_test_fe)

test_probabilities = best_model.predict_proba(
    X_test_fe
)[:, 1]

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)

test_cm = confusion_matrix(
    y_test,
    test_predictions
)
print(f"\nFinal Model: {best_model_name}")

print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall   : {test_recall:.4f}")
print(f"Test F1 Score : {test_f1:.4f}")
print(f"Test ROC-AUC  : {test_auc:.4f}")

print("\nTest Confusion Matrix:")
print(test_cm)
