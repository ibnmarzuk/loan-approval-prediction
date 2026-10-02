# Loan Approval Prediction

End-to-end classification project. A bank file comes in. The model returns the probability the loan is approved, then a class at 0.50.

This is a teaching project on the public Dream Housing Finance loan dataset. It is not a lending decision system.

## Problem

Predict whether a loan application is approved (`Loan_Status = 1`) or rejected (`Loan_Status = 0`).

The useful output is the probability. The class is a cutoff on that probability. 0.50 is the default, not a law.

## Dataset

- Source: Dream Housing Finance loan applications (public training file, 614 rows).
- Target: `Loan_Status`, encoded Y → 1, N → 0.
- After cleaning: 614 rows, 14 features, 0 missing cells.
- Base rate: 68.7% approved. An always-approve rule scores 69.1% on the test fold.

Columns used: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, TotalIncome, Loan_Income_Ratio, Property_Semiurban, Property_Urban.

`Loan_ID` is dropped. `Property_Area` is one-hot encoded with Rural as the reference, so Rural is both flags at 0.

## Preprocessing

1. Inspect missing values. Categorical gaps filled with the mode. Skewed numeric gaps filled with the median.
2. Drop duplicate rows if any. None remained.
3. Encode binaries as 0/1. One-hot the area. Encode the target.
4. Engineer `TotalIncome` and `Loan_Income_Ratio`.
5. Split 80/20, stratified, `random_state=42` → 491 train, 123 test.
6. `StandardScaler` fit on the training fold only, and only on continuous columns.

## Exploratory findings

- Credit history dominates. No file: 7.9% approved. Has a file: 79.0% approved. Pearson r = 0.54.
- Semiurban 76.8%, married 71.8%, graduate 70.8%.
- Applicant income r ≈ 0. A second income is the closer money signal.

## Model

Logistic Regression (`sklearn`, `max_iter=1000`, `random_state=42`). Converged in 21 iterations.

Largest weight after scaling: `Credit_History` +3.14. Then semiurban +0.70, married +0.47, education +0.35.

## Evaluation (test fold, threshold 0.50)

| Metric | Value |
|---|---|
| Accuracy | 0.862 |
| Always-approve baseline | 0.691 |
| ROC-AUC | 0.871 |
| Precision / recall / F1, approved | 0.840 / 0.988 / 0.908 |
| Precision / recall / F1, rejected | 0.957 / 0.579 / 0.721 |
| Confusion matrix | TN 22, FP 16, FN 1, TP 84 |

Accuracy beats the baseline by 17 points. The miss is one-sided: 1 good file refused, 16 bad files approved. When the model says no, it is usually right. It only catches 22 of 38 rejects.

## Predictions on new files

The app encodes a new applicant the same way, scales with the train-fold mean and scale, and applies the fitted weights.

| Case | Profile | P(approve) | Class |
|---|---|---|---|
| A | History, semiurban, income 4,583 | 88.2% | Approve |
| B | Same shape, credit history 0 | 13.9% | Reject |
| C | History, thin rural file | 56.7% | Approve |
| D | Income 18,000, no history | 13.1% | Reject |

Income does not buy a missing credit file.

## Run the app

Open `index.html` in a browser. No server. The weights live in the page.

GitHub Pages: Settings → Pages → Deploy from branch `main` / root. The app is then at:

https://ibnmarzuk.github.io/loan-approval-prediction/

## Reproduce the model

```bash
pip install pandas scikit-learn
python train.py
```

`train.py` reads `loan_cleaned.csv`, repeats the split, and prints the test metrics.

## Limitations

- 614 rows. One split. Not cross-validated.
- The threshold sweep was read on the same test fold. It is a diagnostic, not a tuned operating point.
- Credit history dominates. The other columns add a little, not a new story.
- This does not decide a loan. A false approval is the expensive miss, and the model still makes 16 of them on 123 test rows.
