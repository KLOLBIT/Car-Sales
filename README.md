

# Car Sales Prediction - Machine Learning with SVM

This repository contains a machine learning pipeline that predicts **vehicle type** from car sales data. The project uses **data preprocessing, feature encoding, and SVM-based models (LinearSVC & LinearSVR)** to classify and predict outcomes based on the `Car_sales.csv` dataset.


## Project Structure

```
.
├── Car_sales.csv         # Dataset
├── submission_lsvc.csv   # Predictions using LinearSVC
├── submission_lsvr.csv   # Predictions using LinearSVR
├── main.py               # Main script (your provided code)
└── README.md             # Project documentation
```

## Requirements

Install required libraries:

```
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## Data Preprocessing

Steps applied to `Car_sales.csv`:

* **Dropped Column**:

  * `Latest_Launch` (not useful for prediction)
* **Handled Missing Data**:

  * Numerical columns filled with **median**
  * Categorical columns filled with **mode**
* **Feature Encoding**:

  * Converted `Manufacturer` and `Model` into integer codes
* **Target Encoding**:

  * Converted `Vehicle_type` into integer labels

⚠️ *Note:*
The script raises **FutureWarnings** in `pandas` because of `inplace=True` usage inside chained assignments.
This will break in future pandas versions.
✅ Fix: Instead of

```python
data[i].fillna(data[i].median(), inplace=True)
```

use:

```python
data[i] = data[i].fillna(data[i].median())
```

---

## Models Used

1. **Linear Support Vector Classifier (LinearSVC)**

   * Trained to classify vehicle types.
   * Evaluated with:

     * **F1-score**
     * **Precision**
     * **Accuracy**

2. **Linear Support Vector Regressor (LinearSVR)**

   * Used as a regression alternative to predict `Vehicle_type`.

---

## Results (Sample Run)

* **Class Balance**: \~26% positive class
* **LinearSVC Performance**:

  * F1-score: `0.67`
  * Precision: `0.67`
  * Accuracy: `0.87`
* **LinearSVR**: Predictions generated but not directly evaluated as classification.

⚠️ *Convergence Warning*:
`LinearSVC` may fail to converge on default iterations.
✅ Fix: Increase iterations, e.g.:

```python
lsvc = LinearSVC(max_iter=5000).fit(X_train, y_train)
```

---

## How to Run

1. Place `Car_sales.csv` in the project directory.

2. Run the script:

   ```bash
   python main.py
   ```

3. Outputs:

   * `submission_lsvc.csv` → predictions from LinearSVC
   * `submission_lsvr.csv` → predictions from LinearSVR

---

## Future Improvements

* Tune `LinearSVC` hyperparameters (`C`, `max_iter`) for better convergence.
* Compare against other classifiers (RandomForest, XGBoost, etc.).
* Add proper cross-validation and scaling.
* Separate regression and classification pipelines.

---

## Author

Developed as a practice project for **SVM-based classification and regression** with real-world sales data.

---
