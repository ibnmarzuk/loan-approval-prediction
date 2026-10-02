"""Repeat the Phase 5 split and print Phase 6 test metrics."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

df = pd.read_csv("loan_cleaned.csv")
y = df["Loan_Status"]
X = df.drop(columns=["Loan_Status"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
cont = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "TotalIncome", "Loan_Income_Ratio", "Dependents",
]
scaler = StandardScaler()
Xtr, Xte = X_train.copy(), X_test.copy()
Xtr[cont] = scaler.fit_transform(X_train[cont])
Xte[cont] = scaler.transform(X_test[cont])
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(Xtr, y_train)
pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)[:, 1]
print("train", len(X_train), "test", len(X_test))
print("accuracy", round(accuracy_score(y_test, pred), 4))
print("roc_auc", round(roc_auc_score(y_test, proba), 4))
print("confusion", confusion_matrix(y_test, pred).tolist())
print(classification_report(y_test, pred, digits=3, target_names=["Rejected", "Approved"]))
