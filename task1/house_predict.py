import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Target variable
y = train["SalePrice"]
X = train.drop(["SalePrice"], axis=1)

# Select some important numerical & categorical features
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF",
            "FullBath", "YearBuilt", "Neighborhood"]

X = X[features]
X_test = test[features]

# Preprocessing for numerical and categorical features
numeric_features = ["OverallQual", "GrLivArea", "GarageCars", 
                    "TotalBsmtSF", "FullBath", "YearBuilt"]
categorical_features = ["Neighborhood"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

# Create pipeline
model = Pipeline(steps=[("preprocessor", preprocessor),
                        ("regressor", LinearRegression())])

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print("Validation RMSE:", rmse)

# Predict on test set
preds_test = model.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": preds_test
})

submission.to_csv("submission.csv", index=False)
print("submission.csv file created!")
