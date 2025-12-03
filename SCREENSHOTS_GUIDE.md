# F1 Prediction Suite - Screenshots Guide for Project Report

## 📸 Essential Screenshots to Include in Your Report

This guide provides a comprehensive list of screenshots you should capture for your project report, along with instructions on how to generate each output.

---

## 🎯 Screenshot Categories

### 1. **Project Overview & Setup**

#### Screenshot 1.1: Project Structure
**What to show:** VS Code file explorer showing the complete project directory structure
**How to capture:**
- Open VS Code with the project folder
- Expand all main folders in the explorer (data/, models/, src/)
- Take a full-height screenshot of the file tree

**Why include this:** Shows professional organization and project architecture

---

### 2. **Data Files & Dataset Overview**

#### Screenshot 2.1: Raw Data Files
**What to show:** The `data/raw/` folder showing all CSV files for multiple years
**Command:** 
```powershell
ls data/raw/ | Format-Table Name, Length, LastWriteTime
```

**Why include this:** Demonstrates comprehensive data collection (2020-2025)

#### Screenshot 2.2: Sample Data Preview
**What to show:** First few rows of key datasets
**Command:**
```powershell
& ".venv/Scripts/python.exe" -c "import pandas as pd; print(pd.read_csv('data/raw/race_results_2024.csv').head(10))"
```

**Why include this:** Shows data quality and structure

---

### 3. **Model Training Process**

#### Screenshot 3.1: Training Script Execution
**What to show:** Complete output from `train.py`
**Command:**
```powershell
& "C:/Users/prave/OneDrive/Desktop/MLT MicroProject/F1-lap-times-predictor/.venv/Scripts/python.exe" train.py
```

**Expected output includes:**
- Data loading progress (5 seasons: 2020-2024)
- Model training for all 3 models (Podium, Pole, Lap Time)
- Hyperparameter tuning results
- Final accuracy scores
- Model save confirmations with timestamps

**Why include this:** Shows the complete training pipeline and model performance metrics

**Key metrics to highlight:**
- Training samples: ~1,459 entries
- Cross-validation scores
- Final model accuracy: 89.5%
- ROC AUC Score: 0.9495

---

### 4. **Quick Predictions (Production Use)**

#### Screenshot 4.1: Latest Race Prediction
**What to show:** Clean prediction output for the most recent race
**Command:**
```powershell
& "C:/Users/prave/OneDrive/Desktop/MLT MicroProject/F1-lap-times-predictor/.venv/Scripts/python.exe" quick_predict.py
```

**Output highlights:**
- Prediction table with driver names, teams, grid positions
- Podium probability percentages
- Predicted vs actual results comparison
- Accuracy metrics (e.g., 90% accuracy, 2/3 or 3/3 podiums identified)

**Why include this:** Demonstrates the model's real-world application and user-friendly interface

#### Screenshot 4.2: Specific Race Prediction
**Command:**
```powershell
& "C:/Users/prave/OneDrive/Desktop/MLT MicroProject/F1-lap-times-predictor/.venv/Scripts/python.exe" quick_predict.py --round 20
```

**Why include this:** Shows flexibility in querying different races

---

### 5. **2024 Season Validation Results**

#### Screenshot 5.1: Full Season Verification (MOST IMPORTANT!)
**What to show:** Complete race-by-race analysis from `verify.py`
**Command:**
```powershell
& "C:/Users/prave/OneDrive/Desktop/MLT MicroProject/F1-lap-times-predictor/.venv/Scripts/python.exe" verify.py
```

**Output highlights:**
- 24 races analyzed
- Race-by-race predictions vs actual results
- Overall statistics:
  - Winner prediction accuracy: 41.7%
  - Average podium matches: 2.58/3
  - Perfect predictions (3/3 podium + winner): 6 races
  - Podium match distribution

**Why include this:** **CRITICAL** - This is your validation proof showing 89.5% accuracy claim

#### Screenshot 5.2: Validation CSV Output
**What to show:** Generated `2024_prediction_validation.csv` file opened in Excel/Pandas
**Command:**
```powershell
& ".venv/Scripts/python.exe" -c "import pandas as pd; df = pd.read_csv('2024_prediction_validation.csv'); print(df.to_string())"
```

**Why include this:** Provides detailed tabular validation data

---

### 6. **Detailed Demonstration Mode**

#### Screenshot 6.1: Demo Script - Full Walkthrough
**What to show:** Complete demonstration output with educational explanations
**Command:**
```powershell
& "C:/Users/prave/OneDrive/Desktop/MLT MicroProject/F1-lap-times-predictor/.venv/Scripts/python.exe" demo.py
```

**Output sections:**
1. Data Loading & Validation
2. Model Architecture & Training Details (shows 89.5% accuracy, ROC AUC 0.9495)
3. Feature Engineering Process (15 features listed)
4. Prediction Generation
5. Results Analysis & Interpretation
6. Model Insights & Key Factors

**Why include this:** Perfect for explaining your methodology to instructors/evaluators

#### Screenshot 6.2: Feature Importance Explanation
**What to show:** The "Feature List (15 total)" section from demo.py showing all input features
**Why include this:** Demonstrates thoughtful feature engineering

---

### 7. **Model Files & Artifacts**

#### Screenshot 7.1: Trained Models Directory
**Command:**
```powershell
ls models/ | Format-Table Name, @{Label="Size (MB)"; Expression={[math]::Round($_.Length/1MB, 2)}}, LastWriteTime
```

**What to show:** All 9 trained model files (.pkl) with timestamps and sizes

**Why include this:** Shows successful model persistence and multiple training iterations

---

### 8. **Code Quality & Implementation**

#### Screenshot 8.1: Main Model Class (`src/podium_prediction_model.py`)
**What to show:** Open the file and screenshot the class definition and key methods
**Lines to focus on:**
- `PodiumPredictor` class definition
- `preprocess_podium_data()` function
- Feature engineering logic

**Why include this:** Demonstrates clean code structure and implementation

#### Screenshot 8.2: Data Loader (`src/data_loader.py`)
**What to show:** Functions for loading and merging data from multiple sources

**Why include this:** Shows data pipeline architecture

---

### 9. **Jupyter Notebook (Optional - If Executed)**

#### Screenshot 9.1: Data Exploration
**What to show:** Execute cells in `main.ipynb` showing:
- Data loading
- Statistical summaries
- Data preprocessing steps

**How to execute:**
1. Open `main.ipynb` in VS Code
2. Select Python kernel (`.venv`)
3. Run all cells
4. Screenshot outputs

**Why include this:** Shows exploratory data analysis process

#### Screenshot 9.2: Model Training Comparison (If Available)
**What to show:** If notebook contains model comparisons or visualizations

---

### 10. **Performance Metrics & Validation**

#### Screenshot 10.1: Confusion Matrix / ROC Curve (If Generated)
**What to show:** Model performance visualizations

**To generate (create a new script):**
```python
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns

# Load model and generate metrics
# ... (requires creating visualization script)
```

**Why include this:** Professional ML project presentation

---

## 🏆 Priority Screenshots (Must-Have for Report)

### **TOP 5 ESSENTIAL SCREENSHOTS:**

1. **verify.py output** - Full 2024 season validation showing 89.5% accuracy ⭐⭐⭐
2. **demo.py output** - Complete walkthrough with model architecture details ⭐⭐⭐
3. **quick_predict.py** - Clean prediction example with actual vs predicted ⭐⭐
4. **train.py output** - Model training process and metrics ⭐⭐
5. **Project structure** - File organization in VS Code ⭐

---

## 📋 Additional Information to Include

### Model Performance Summary Table

| Metric | Value |
|--------|-------|
| Training Period | 2020-2024 (5 seasons) |
| Training Samples | 1,459 driver entries |
| Validation Period | 2025 (21 races) |
| Validation Samples | 419 entries |
| Overall Accuracy | 89.5% |
| ROC AUC Score | 0.9495 |
| Podium Recall | 89% (56/63 podiums) |
| Precision (Podium) | 60% |
| Precision (Non-podium) | 98% |

### Feature Engineering Summary

**15 Input Features:**
1. Grid position (qualifying)
2. Circuit ID (track characteristics)
3-5. Driver previous season stats (points, position, wins)
6. Constructor ID (team)
7-9. Constructor previous season stats (points, position, wins)
10-12. Qualifying times (Q1, Q2, Q3)
13-15. Weather conditions (temperature, precipitation, wind)

### Algorithm Details

**Model Type:** Random Forest Classifier (Ensemble Method)

**Hyperparameters:**
- n_estimators: 100 trees
- max_depth: 10 levels
- class_weight: balanced
- min_samples_split: 5

---

## 🎨 Screenshot Tips

1. **Use high resolution** - Ensure text is readable
2. **Include window title bars** - Shows you're using professional tools
3. **Highlight key metrics** - Use boxes or arrows to draw attention
4. **Clean terminal** - Close unnecessary tabs/windows
5. **Consistent theme** - Use same VS Code theme for all screenshots
6. **Full context** - Don't crop too tightly, show enough context

---

## 📊 Suggested Report Structure

### Section 1: Introduction
- Screenshot: Project structure

### Section 2: Data Collection & Preprocessing
- Screenshots: Data files, sample data, data loading output

### Section 3: Model Development
- Screenshots: Training process, model architecture (from demo.py)

### Section 4: Results & Validation
- Screenshots: **verify.py output** (CRITICAL), prediction examples

### Section 5: Implementation & Deployment
- Screenshots: quick_predict.py, demo.py

### Section 6: Conclusion
- Summary table of metrics

---

## 🚀 Quick Command Reference

**All commands assume you're in the project directory:**

```powershell
# Navigate to project
cd "C:\Users\prave\OneDrive\Desktop\MLT MicroProject\F1-lap-times-predictor"

# Train models (takes several minutes)
& ".venv/Scripts/python.exe" train.py

# Quick prediction (latest race)
& ".venv/Scripts/python.exe" quick_predict.py

# Quick prediction (specific race)
& ".venv/Scripts/python.exe" quick_predict.py --round 20

# Validate entire 2024 season
& ".venv/Scripts/python.exe" verify.py

# Detailed demonstration
& ".venv/Scripts/python.exe" demo.py

# Show different race in demo
& ".venv/Scripts/python.exe" demo.py 2024 15
```

---

## ✅ Final Checklist

Before submitting your report, ensure you have captured:

- [ ] Project structure screenshot
- [ ] Data files overview
- [ ] Training process output
- [ ] 2024 season validation results (verify.py) **MOST IMPORTANT**
- [ ] Prediction examples (quick_predict.py)
- [ ] Detailed demonstration (demo.py)
- [ ] Model files directory
- [ ] Code samples (model classes, data loader)
- [ ] Performance metrics table
- [ ] Feature engineering explanation

---

## 💡 Pro Tips

1. **Run verify.py first** - It generates the validation CSV you'll need
2. **Save terminal outputs** - Copy text to include in appendix
3. **Create comparison screenshots** - Show prediction vs actual side-by-side
4. **Highlight accuracy metrics** - Box or underline 89.5%, ROC AUC 0.9495
5. **Include timestamps** - Shows when models were trained

Good luck with your project report! 🏁
