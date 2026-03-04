# Yelp Review Star Rating Prediction
## Author

Ben Funk

**Version: 1.0.0**

A basic machine learning project that predicts Yelp review star ratings (1-5) from review text using traditional ML models and deep learning approaches. Built with Python, scikit-learn, XGBoost, and PyTorch.

## Overview

This project tackles a 5 class text classification problem on the Yelp academic dataset. The goal is to predict the star rating a reviewer gave based solely on their written review text, along with optional business and user metadata.

The dataset includes nearly 7 million reviews spanning 2005-2022. After cleaning and balancing, we train on 2 million reviews (400k per star rating) and test on 500k reviews (100k per star rating).

## Project Structure

```
DS-Capstone/
├── Data/
│   ├── Raw/                    # Original Yelp dataset (not in repo)
│   └── Processed/              # Cleaned train/test splits
├── EDA/
│   └── EDA-And-Data-Cleaning.ipynb    # Exploratory analysis and preprocessing
├── Model Training/
│   ├── Statistical Modeling.ipynb     # Random Forest and XGBoost models
│   ├── Neural Models.ipynb            # Bidirectional LSTM with PyTorch
│   └── Statistical Inference.ipynb    # Prediction interface for trained models
└── Outputs/
    ├── Models/                 # Saved trained models
    └── Plots/                  # Visualization outputs
```

## Dataset

**Source:** Yelp Academic Dataset
**Size:** 6.99 million reviews initially, 2.5 million after stratified sampling
**Features:**
- Review text (primary predictor)
- Star rating (target: 1-5)
- Business metadata (categories, location, hours)
- User engagement (useful/funny/cool votes)
- Temporal features (year, month, day of week)

The dataset exhibits significant class imbalance, with 5-star reviews being most common (46% of raw data). We address this through stratified sampling to create a perfectly balanced training set.

## Methodology

### 1. Exploratory Data Analysis

**Notebook:** `EDA/EDA-And-Data-Cleaning.ipynb`

Key findings:
- Strong class imbalance in raw data (5.94:1 ratio)
- Average review length: 105 words
- 2-4 star reviews are hardest to classify (neutral sentiment)
- Text length negatively correlates with star rating
- 99.98% data retention rate after cleaning

Data cleaning steps:
- Remove duplicates and invalid entries
- Filter extremely short reviews (less than 5 words)
- Handle missing values in metadata
- Create stratified train/test splits

### 2. Statistical Models

**Notebook:** `Model Training/Statistical Modeling.ipynb`

**Models:**
- Random Forest (text-only): Baseline using TF-IDF features
- Random Forest (text + metadata): Adds business/user context
- XGBoost (text + metadata): Gradient boosting for improved performance

**Feature Engineering:**
- TF-IDF vectorization (7,000 features, unigrams + bigrams)
- Metadata encoding (43 features including location, categories, engagement)
- Total feature space: 7,043 dimensions

**Results:**
- Baseline (most frequent): 20.0% accuracy
- Random Forest (text): 51.0% accuracy
- Random Forest (combined): 52.3% accuracy
- **XGBoost (combined): 57.9% accuracy** (best statistical model)

Adding metadata provides a 1.3% accuracy boost. The models excel at predicting extreme ratings (1 and 5 stars) but struggle with neutral 3-star reviews.

### 3. Deep Learning Model

**Notebook:** `Model Training/Neural Models.ipynb`

**Architecture:** Bidirectional LSTM
- Embedding layer (100 dimensions)
- 2-layer Bidirectional LSTM (256 hidden units)
- Dropout regularization (0.5)
- Fully connected output layer
- Total parameters: 150M+

**Training:**
- PyTorch with CUDA acceleration (NVIDIA RTX 5070 Ti)
- Adam optimizer (lr=0.001)
- Cross-entropy loss with class weights
- 10 epochs (~30 minutes on GPU)
- Vocabulary size: 245,599 words

**Results:**
- **LSTM: 57.9% accuracy** (matches XGBoost)
- Better at capturing long-range text dependencies
- More computationally expensive than statistical models

### 4. Inference System

**Notebook:** `Model Training/Statistical Inference.ipynb`

A user-friendly interface for making predictions on new reviews. Loads all trained models and provides:
- Single review predictions with probability distributions
- Batch prediction capabilities
- Consensus predictions across models
- Visualization of model confidence

## Key Results

| Model | Accuracy | F1-Score (Macro) |
|-------|----------|------------------|
| Baseline (Most Frequent) | 20.0%    | N/A              |
| Random Forest (Text) | 51.0%    | 0.509            |
| Random Forest (Combined) | 52.3%    | 0.523            |
| XGBoost (Combined) | 57.9%    | 0.579            |
| Bidirectional LSTM | 68.4%    | 0.683            |

**Per-Class Performance (XGBoost):**
- 1 star: 68% precision, 71% recall
- 2 star: 41% precision, 39% recall
- 3 star: 37% precision, 35% recall
- 4 star: 52% precision, 53% recall
- 5 star: 72% precision, 72% recall

The models perform best on extreme ratings and worst on neutral reviews, which aligns with the inherent ambiguity in 3-star sentiment.

## Technical Stack

**Core:**
- Python 3.13
- Jupyter notebooks

**Data Processing:**
- pandas 2.3.3
- numpy 2.4.1
- scikit-learn 1.8.0

**Machine Learning:**
- XGBoost 3.1.3
- PyTorch 2.10.0 (with CUDA 12.8)
- torchvision 0.22.0

**Visualization:**
- matplotlib
- seaborn

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd DS-Capstone
```

2. Install dependencies using uv:
```bash
pip install uv
uv sync
```

The project uses uv for dependency management. PyTorch is configured to use CUDA 12.8 through a custom PyPI index.

## Usage

### Training Models

Run notebooks in order:

1. **Data Preparation:**
```bash
jupyter notebook "EDA/EDA-And-Data-Cleaning.ipynb"
```
Downloads and preprocesses the Yelp dataset, creates train/test splits.

2. **Statistical Models:**
```bash
jupyter notebook "Model Training/Statistical Modeling.ipynb"
```
Trains Random Forest and XGBoost models, saves to `Outputs/Models/`.

3. **Deep Learning:**
```bash
jupyter notebook "Model Training/Neural Models.ipynb"
```
Trains Bidirectional LSTM with PyTorch. Requires CUDA-capable GPU for reasonable training time.

### Making Predictions

```python
# Run the inference notebook
jupyter notebook "Model Training/Statistical Inference.ipynb"

# Or use trained models directly:
import pickle
with open('Outputs/Models/gbm.pkl', 'rb') as f:
    model = pickle.load(f)
```

## License

This project uses the Yelp Academic Dataset, which is available for academic use. Check Yelp's terms of service before using this code for commercial purposes.
