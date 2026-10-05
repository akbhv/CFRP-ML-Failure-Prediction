# CFRP-ML-Failure-Prediction

Physics-based and machine-learning framework for failure prediction of CFRP composite laminates.

## 1. Project Overview

This project develops a physics-based computational framework for analysing failure in carbon-fibre-reinforced polymer (CFRP) laminates and uses the resulting physics-generated data to train deep neural-network surrogate models.

The overall workflow combines:

- Micromechanics-based lamina property estimation
- Reduced stiffness matrix calculation
- Stiffness transformation
- Classical Lamination Theory (CLT)
- Multiple composite failure criteria
- Physics-based dataset generation
- Deep neural-network regression
- Deep neural-network failure classification
- Independent computational generalization
- WWFE Case 3 benchmarking

The machine-learning models learn the behaviour of the underlying CLT–Hashin computational framework rather than directly learning experimental laminate failure behaviour.

---

## 2. Physics-Based Framework

The physics-based pipeline follows:

Constituent material properties
            ↓
       Micromechanics
            ↓
     Lamina properties
            ↓
      Reduced Q matrix
            ↓
   Stiffness transformation
            ↓
 Classical Lamination Theory
            ↓
 Ply stresses and strains
            ↓
     Failure criteria
            ↓
      Hashin failure index

The implemented failure criteria are:

- Maximum Stress
- Tsai-Hill
- Tsai-Wu
- Hashin

Hashin failure index is used as the primary target for the machine-learning models.

---

## 3. Dataset

The final dataset contains 20,000 physics-generated loading cases for a fixed CFRP laminate configuration.

Each case contains six loading variables:

- Nx
- Ny
- Nxy
- Mx
- My
- Mxy

Additional outputs include:

- Hashin failure index
- Maximum Stress failure index
- Tsai-Hill failure index
- Tsai-Wu failure index
- Safe/failure classification
- Hashin failure mode
- Governing ply
- Governing ply surface
- Boundary failure scale
- Applied load multiplier
- Sampling region

The final dataset was independently audited for:

- Duplicate cases
- Hashin-label consistency
- Failure-index range
- Sampling-region distribution
- Failure-mode distribution

---

## 4. Machine Learning

Two deep neural-network models are used.

### 4.1 Hashin Failure Index Regression

The regression model predicts the continuous Hashin failure index.

Architecture:

6 → 64 → 64 → 32 → 1

The hidden layers use ReLU activation and the output layer uses a linear activation.

Training uses:

- Adam optimizer
- Learning rate = 0.001
- Mean squared error loss
- Early stopping using validation loss
- Batch size = 32

### 4.2 Failure Classification

The classification model predicts whether a loading case is safe or failed.

Architecture:

6 → 64 → 64 → 32 → 1

The output layer uses sigmoid activation.

Training uses:

- Adam optimizer
- Learning rate = 0.001
- Binary cross-entropy loss
- Early stopping using validation loss
- Batch size = 32

The classification threshold is selected using the validation dataset and then frozen before evaluation on the test dataset.

---

## 5. Dataset Splitting and Preprocessing

The final dataset is divided into:

Training: 14,000 samples
Validation: 3,000 samples
Test: 3,000 samples

The six loading variables are standardized using a StandardScaler.

The scaler is fitted using the training set only and subsequently applied to the validation and test sets.

---

## 6. Final Regression Results

The final DNN regression model was evaluated on the held-out test set.

MAE: 0.00435
RMSE: 0.00861
R²: 0.99975

The median absolute prediction error was 0.00260.

The model maintained low error near the physically important Hashin failure boundary at FI = 1.

---

## 7. Independent Computational Generalization

The trained regression model was additionally evaluated on 5,000 independently generated loading cases that were not part of the training, validation, or test datasets.

Results:

MAE: 0.00478
RMSE: 0.01066
R²: 0.99961

This demonstrates strong generalization to previously unseen loading combinations generated using the same CLT–Hashin computational framework.

This should not be interpreted as experimental validation of the DNN.

---

## 8. Final Classification Results

The classification threshold was selected using the validation dataset.

Selected threshold:

0.65

The frozen threshold was then evaluated on the untouched test set.

Accuracy: 96.60%
Precision: 95.96%
Recall: 97.25%
F1-score: 96.60%

Final test confusion matrix:

              Predicted
              Safe  Failure
Actual Safe   1449     61
Actual Fail     41   1449

---

## 9. Physics Validation

The physics implementation is supported by dedicated validation scripts covering:

- Micromechanics
- Lamina properties
- Reduced stiffness matrix
- Stiffness transformation
- Classical Lamination Theory
- Local ply stresses
- Laminate failure detection
- Maximum Stress
- Tsai-Hill
- Tsai-Wu
- Hashin
- Failure-boundary detection
- Signed loading directions
- Multiple failure boundaries
- Dataset evaluation

---

## 10. WWFE Benchmark

The underlying CLT–Hashin framework was benchmarked against WWFE Case 3 experimental data for the quasi-isotropic AS4/3501-6 laminate.

The comparison uses the published experimental biaxial failure data and evaluates the predicted first-ply-failure envelope.

The WWFE comparison is treated separately from DNN validation.

The DNN is trained using CLT–Hashin-generated data, while the WWFE comparison evaluates the underlying physics-based failure framework against experimental observations.

Therefore, the WWFE comparison should not be described as direct experimental validation of the machine-learning model.

---

## 11. Repository Structure

CFRP-ML-Failure-Prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
│
├── src/
│   ├── materials.py
│   ├── micromechanics.py
│   ├── lamina.py
│   ├── transformation.py
│   ├── clt.py
│   ├── failure_criteria.py
│   ├── dataset_generator.py
│   ├── generate_dataset_final.py
│   ├── generate_unseen_dataset.py
│   ├── prepare_ml_dataset.py
│   ├── train_dnn_regression.py
│   ├── train_dnn_classifier.py
│   ├── analyze_regression_errors.py
│   ├── analyze_validation_threshold.py
│   ├── evaluate_final_classifier.py
│   └── evaluate_generalization.py
│
├── validation/
│   ├── test_micromechanics.py
│   ├── test_Q.py
│   ├── test_transformation.py
│   ├── test_clt.py
│   ├── test_failure_criteria.py
│   └── wwfe_cases/
│
├── notebooks/
├── results/
├── docs/
├── README.md
├── requirements.txt
└── .gitignore

---

## 12. Key Scripts

### Physics

src/micromechanics.py
src/lamina.py
src/transformation.py
src/clt.py
src/failure_criteria.py

### Dataset generation

src/dataset_generator.py
src/generate_dataset_final.py
src/generate_unseen_dataset.py

### Machine learning

src/prepare_ml_dataset.py
src/train_dnn_regression.py
src/train_dnn_classifier.py

### Evaluation

src/analyze_regression_errors.py
src/analyze_validation_threshold.py
src/evaluate_final_classifier.py
src/evaluate_generalization.py

### WWFE

validation/wwfe_cases/

---

## 13. Reproducibility

Create and activate the Python virtual environment:

python -m venv .venv
.venv\Scripts\activate

Install the required packages:

pip install -r requirements.txt

Physics validation modules can be executed using:

python -m validation.test_micromechanics
python -m validation.test_Q
python -m validation.test_transformation
python -m validation.test_clt
python -m validation.test_failure_criteria

The final machine-learning preprocessing pipeline is:

python src\prepare_ml_dataset.py

The final DNN training scripts are:

python src\train_dnn_regression.py
python src\train_dnn_classifier.py

The final evaluation scripts are:

python src\analyze_regression_errors.py
python src\analyze_validation_threshold.py
python src\evaluate_final_classifier.py
python src\evaluate_generalization.py

---

## 14. Limitations

The machine-learning models learn the behaviour of the underlying CLT–Hashin computational framework.

The current framework does not directly model complete progressive damage evolution or all specimen-level failure mechanisms.

The WWFE comparison highlights the distinction between first-ply failure predicted by the simplified CLT–Hashin framework and experimentally observed laminate-level failure.

Consequently, the machine-learning model should be interpreted as a surrogate for the computational physics framework rather than as a direct replacement for experimental testing.

---

## 15. Project Status

The core project workflow has been implemented:

Physics model
      ↓
Failure criteria
      ↓
Physics-generated dataset
      ↓
DNN regression
      ↓
DNN classification
      ↓
Independent computational generalization
      ↓
WWFE benchmark

The final models and validation results are being prepared for the final-year project report and evaluation.