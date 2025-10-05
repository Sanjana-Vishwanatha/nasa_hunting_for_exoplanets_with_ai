# 🌌 NASA Space Apps Challenge 2025 - Hunting for Exoplanets using AI

An AI-powered web application for detecting and classifying exoplanets using machine learning models trained on NASA's Kepler and TESS mission data.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Information](#model-information)
- [Dataset Requirements](#dataset-requirements)
- [Contributing](#contributing)

---

## 🎯 Overview

This project uses Random Forest Classification models to predict exoplanet dispositions based on observational data from two major NASA missions:

- **Kepler Mission**: Analyzes Kepler Objects of Interest (KOI)
- **TESS Mission**: Analyzes TESS Objects of Interest (TOI)

The system classifies candidates into categories such as **CONFIRMED**, **CANDIDATE**, or **FALSE POSITIVE**.

---

## ✨ Features

- **Dual Model Support**: Separate models for Kepler and TESS datasets
- **Single Prediction**: Predict individual exoplanet candidates via JSON input
- **Batch Prediction**: Upload CSV/JSON files for bulk predictions
- **RESTful API**: Flask-based backend with CORS support
- **Automated Preprocessing**: Handles missing values, scaling, and categorical encoding
- **Export Results**: Download predictions as CSV files

---

## 🛠️ Technology Stack

- **Python 3.13**
- **Flask** - Web framework
- **scikit-learn** - Machine learning models
- **pandas & numpy** - Data manipulation
- **joblib** - Model serialization
- **uv** - Fast Python package manager

---

## 📦 Prerequisites

- Python 3.13 or higher
- `uv` package manager (recommended) or `pip`
- Git

---

## 🚀 Installation

### Step 1: Install `uv` Package Manager

#### **Windows (PowerShell)**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### **macOS (Homebrew)**
```bash
brew install uv
```

#### **macOS/Linux (Install Script)**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:
```bash
uv --version
```

---

### Step 2: Clone the Repository

```bash
git clone https://github.com/your-username/exoplanet-detection-ai.git
cd exoplanet-detection-ai
```

---

### Step 3: Set Up Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate the environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On macOS/Linux:
source .venv/bin/activate
```

---

### Step 4: Install Dependencies

```bash
# Sync all dependencies from pyproject.toml
uv sync
```

This will install:
- flask
- flask-cors
- numpy
- pandas
- scikit-learn
- xgboost
- torch
- ipykernel
- python-dotenv

---

### Step 5: Prepare Datasets

Create a `dataset/` directory in the project root and add your data files:

```bash
mkdir dataset
```

Required files:
- `dataset/kepler_exoplanet_data.csv` (Kepler mission data)
- `dataset/TESS_exoplanet_data.csv` (TESS mission data)

**Note**: These CSV files should have headers starting at specific rows (row 53 for Kepler, row 90 for TESS) as indicated in the code.

---

### Step 6: Prepare Pre-trained Models

Create a `models/` directory and ensure trained models are present:

```bash
mkdir models
```

Required model files:
- `models/random_forest_exoplanet_model.joblib` (Kepler model)
- `models/random_forest_exoplanet_model_tess.joblib` (TESS model)

**To train models from scratch**, run:

```python
from kepler import ExoplanetModel
from tess import ExoplanetModel_TESS

# Train Kepler model
kepler_model = ExoplanetModel()
kepler_model.train_model()

# Train TESS model
tess_model = ExoplanetModel_TESS()
tess_model.train_model()
```

---

## 📁 Project Structure

```
exoplanet-detection-ai/
│
├── .venv/                  # Virtual environment (not in git)
├── dataset/                # Dataset folder
│   ├── kepler_exoplanet_data.csv
│   └── TESS_exoplanet_data.csv
├── models/                 # Trained models
│   ├── random_forest_exoplanet_model.joblib
│   └── random_forest_exoplanet_model_tess.joblib
│
├── kepler.py              # Kepler model class
├── tess.py                # TESS model class
├── main.py                # Flask API server
├── pyproject.toml         # Project dependencies
├── .python-version        # Python version (3.13)
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

---

## 🎮 Usage

### Starting the Server

```bash
# Using uv (recommended)
uv run python main.py

# Or if virtual environment is activated
python main.py
```

The server will start on `http://0.0.0.0:5000` (accessible via `http://localhost:5000`)

---

## 🔌 API Endpoints

### 1. **Kepler Single Prediction**

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "user_input": {
    "koi_fpflag_nt": 0,
    "koi_fpflag_ss": 0,
    "koi_fpflag_co": 0,
    "koi_fpflag_ec": 0,
    "koi_period": 14.45,
    "koi_period_err1": 0.00012,
    "koi_period_err2": -0.00012,
    "koi_time0bk": 170.538,
    "koi_impact": 0.146,
    "koi_duration": 2.95,
    "koi_depth": 615.8,
    "koi_prad": 2.26,
    "koi_teq": 793,
    "koi_insol": 93.59,
    "koi_steff": 5455,
    "koi_slogg": 4.467,
    "koi_srad": 0.927
  }
}
```

**Response**:
```json
{
  "predicted_class": 1,
  "predicted_label": "CONFIRMED"
}
```

---

### 2. **Kepler Batch Prediction**

**Endpoint**: `POST /upload`

**Request**: Multipart form-data with file (CSV or JSON)

**Response**: CSV file with predictions added as new columns

---

### 3. **TESS Single Prediction**

**Endpoint**: `POST /predict_tess`

**Request Body**: Similar structure to Kepler with TESS-specific features

**Response**:
```json
{
  "predicted_class": 0,
  "predicted_label": "CANDIDATE"
}
```

---

### 4. **TESS Batch Prediction**

**Endpoint**: `POST /upload_tess`

**Request**: Multipart form-data with file (CSV or JSON)

**Response**: CSV file with predictions

---

## 🤖 Model Information

### Kepler Model (`kepler.py`)

- **Algorithm**: Random Forest Classifier (100 estimators)
- **Preprocessing**:
  - Drops identifier columns (kepid, kepoi_name, kepler_name)
  - Converts categorical variables to numerical codes
  - Handles missing values using median imputation
  - Replaces -inf values with median
  - Applies RobustScaler for feature scaling
- **Target Variable**: `koi_disposition`
- **Categories**: CANDIDATE, CONFIRMED, FALSE POSITIVE

### TESS Model (`tess.py`)

- **Algorithm**: Random Forest Classifier (200 estimators)
- **Preprocessing**:
  - Drops identifier columns (toi, tid, timestamps)
  - Similar preprocessing pipeline to Kepler model
- **Target Variable**: `tfopwg_disp`
- **Categories**: Mission-specific dispositions

---

## 📊 Dataset Requirements

### Kepler Dataset Columns (Example)
Key features include:
- False positive flags (koi_fpflag_*)
- Orbital period (koi_period)
- Transit parameters (koi_duration, koi_depth)
- Planetary characteristics (koi_prad, koi_teq)
- Stellar properties (koi_steff, koi_slogg, koi_srad)

### TESS Dataset Columns
Similar structure with TESS-specific nomenclature.

**Note**: The models automatically handle missing columns by filling with default values (0 or median).

---

## 🧪 Testing the API

### Using cURL (Kepler Prediction)

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": {
      "koi_period": 14.45,
      "koi_prad": 2.26,
      "koi_teq": 793,
      "koi_steff": 5455
    }
  }'
```

### Using Python (Batch Upload)

```python
import requests

url = "http://localhost:5000/upload"
files = {'file': open('test_data.csv', 'rb')}
response = requests.post(url, files=files)

with open('predictions.csv', 'wb') as f:
    f.write(response.content)
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):
```env
FLASK_ENV=development
FLASK_DEBUG=True
```

Load using:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Module not found errors**
```bash
uv sync
```

**2. Model file not found**
Ensure models are in `models/` directory or train them first.

**3. Dataset loading errors**
Check that CSV files exist in `dataset/` and have correct skiprows parameter.

**4. CORS errors**
CORS is enabled by default via `flask-cors`. Ensure it's installed.

---

## 📝 Development Notes

- Models use `RobustScaler` to handle outliers
- Categorical variables are encoded as numerical codes
- Missing values are imputed with median values from training data
- The API is stateless - models are loaded once at startup

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is developed for NASA Space Apps Challenge 2025.

---

## 👥 Team

Sanjana - sanjana.vishwanatha2000@gmail.com
Nagarjun - nagarjunts008@gmail.com

---

## 🙏 Acknowledgments

- NASA Kepler Mission
- NASA TESS Mission
- NASA Exoplanet Archive

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Happy Exoplanet Hunting! 🚀🪐**
