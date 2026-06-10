# Store Sales Forecasting — Project Breakdown

A complete walkthrough of a Convolutional Neural Network (CNN) built to forecast retail store sales 16 days into the future. This document explains every step of the project in plain language, from raw CSV files to a final Kaggle submission.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [The Data](#3-the-data)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Creating the Sliding Window Dataset](#5-creating-the-sliding-window-dataset)
6. [PyTorch Setup](#6-pytorch-setup)
7. [The TCN Model Architecture](#7-the-tcn-model-architecture)
8. [Training the Model](#8-training-the-model)
9. [Validation](#9-validation)
10. [Final Model: Training on Full Data](#10-final-model-training-on-full-data)
11. [Generating Predictions](#11-generating-predictions)
12. [Creating the Submission File](#12-creating-the-submission-file)
13. [Results Summary](#13-results-summary)
14. [Dependencies & Setup](#14-dependencies--setup)

---

## 1. Project Overview

This project uses a deep learning model to forecast daily sales for a large grocery retailer in Ecuador. The data comes from the [Kaggle Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition.

**The goal:** Given the last 120 days of historical sales data across 54 stores and 33 product families, predict the next 16 days of sales.

| Property | Detail |
|---|---|
| Competition | Kaggle Store Sales — Time Series Forecasting |
| Country | Ecuador (Corporación Favorita grocery chain) |
| Model Type | Temporal Convolutional Network (TCN) |
| Input | 120 days of historical sales |
| Output | 16-day sales forecast |
| Store-Product Combinations | 1,782 (54 stores × 33 product families) |
| Final Output | `submission.csv` with 28,512 predictions |

---

## 2. Project Structure

```
Store Sales Forecasting/
│
├── main.ipynb                    # The entire project: data loading, model, training, submission
├── submission.csv                # Final predictions exported for Kaggle submission
├── requirements.txt              # Python package dependencies
│
└── data/
    ├── train.csv                 # 116 MB — historical daily sales from 2013 to 2017
    ├── test.csv                  # Sales periods to predict (Aug 16–31, 2017)
    ├── stores.csv                # Metadata about each store (city, type, cluster)
    ├── transactions.csv          # Daily transaction counts per store
    ├── oil.csv                   # Daily oil price (Ecuador's economy is oil-dependent)
    ├── holidays_events.csv       # National and regional holidays
    └── sample_submission.csv     # Example of Kaggle's expected output format
```

> **Note:** Only `train.csv` and `test.csv` are used directly in this model. The supplementary files (`oil.csv`, `holidays_events.csv`, etc.) were not incorporated as additional features in this version.

---

## 3. The Data

### What's in `train.csv`?

Each row represents the sales of **one product family at one store on one day**.

| Column | Description |
|---|---|
| `id` | Unique row identifier |
| `date` | The calendar date |
| `store_nbr` | Which store (1–54) |
| `family` | Product category (e.g., `BEVERAGES`, `DAIRY`, `PRODUCE`) |
| `sales` | Total units sold that day |
| `onpromotion` | Number of items in that family that were on promotion |

The dataset contains over **3 million rows** covering roughly 6 years of daily sales.

### Store-Family Combinations

With 54 stores and 33 product families, there are **54 × 33 = 1,782 unique combinations**. Each combination is its own time series. The model needs to learn patterns across all of them simultaneously.

A new column called `store_family` is created by combining the store number and family name:

```python
df['store_family'] = df['store_nbr'].astype(str) + '_' + df['family']
# Example: "3_BEVERAGES", "12_DAIRY", "47_PRODUCE"
```

### Pivoting the Data

The raw data is in **long format** — one row per store-family-date combination. For the neural network, we need **wide format** — one row per date, with each column representing a different store-family combination.

```
LONG FORMAT (raw):
date         store_family    sales
2013-01-01   1_BEVERAGES     100
2013-01-01   1_DAIRY          45
2013-01-02   1_BEVERAGES      98
...

WIDE FORMAT (after pivot):
date         1_BEVERAGES   1_DAIRY   ...   54_PRODUCE
2013-01-01       100          45     ...      210
2013-01-02        98          52     ...      198
...
```

This transformation creates a matrix with:
- **Rows:** Each day in the dataset
- **Columns:** 1,782 store-family time series

This format allows the model to treat each store-family combination as a separate "channel" — similar to how an image has red, green, and blue channels.

---

## 4. Data Preprocessing

### Train-Test Split

The data is split into 80% training and 20% testing — but **without shuffling**.

> **Why no shuffling?** In a standard machine learning problem (like classifying images), you can randomly mix your data. But time series data has a strict order: the past must come before the future. If we shuffled, we might accidentally train the model on data from 2016 and test it on data from 2014 — which doesn't reflect how the model would actually be used. We always train on earlier data and test on later data.

### StandardScaler Normalization

After splitting, each column (store-family series) is normalized using `StandardScaler` from scikit-learn. This transforms each series to have:
- **Mean = 0**
- **Standard deviation = 1**

```python
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)   # Learn mean/std from training data
test_scaled  = scaler.transform(test_data)         # Apply same scaling to test data
```

> **Why scale?** Different store-family combinations sell very different quantities. A busy store selling thousands of beverages per day and a small store selling a few specialty items per day are on completely different numerical scales. Without normalization, the model would pay far more attention to the large-valued series. Scaling puts everything on equal footing.

> **Why fit only on training data?** If we included test data when calculating the mean and standard deviation, information from the "future" would leak into the training process. This is called **data leakage** and would make our model appear better than it actually is. We always learn statistics only from training data and apply them to test data.

---

## 5. Creating the Sliding Window Dataset

### What is a Sliding Window?

A sliding window is a technique for turning a long time series into many smaller training examples. We pick a fixed-size input window and a fixed-size output window, then slide them forward through time one step at a time.

```
Full time series: [day1, day2, day3, day4, day5, day6, day7, day8 ...]

Window 1:  Input=[day1..day4]  →  Output=[day5..day6]
Window 2:  Input=[day2..day5]  →  Output=[day6..day7]
Window 3:  Input=[day3..day6]  →  Output=[day7..day8]
...
```

### Parameters Used

| Parameter | Value | Meaning |
|---|---|---|
| `input_length` | 120 days | How much history the model sees |
| `output_length` | 16 days | How far ahead the model predicts |

### Implementation

```python
def create_X_y(data, input_length, output_length):
    X, y = [], []
    for i in range(len(data) - input_length - output_length + 1):
        X.append(data[i : i + input_length])
        y.append(data[i + input_length : i + input_length + output_length])
    return np.array(X), np.array(y)
```

At each position `i`, we slice out 120 days as the input (`X`) and the immediately following 16 days as the target (`y`).

### Resulting Data Shapes

```
X_train shape: (1211, 120, 1782)
  └─ 1211 training windows
     └─ each window: 120 days of history
        └─ for each of 1782 store-family channels

y_train shape: (1211, 16, 1782)
  └─ 1211 corresponding targets
     └─ each target: 16 future days
        └─ for each of 1782 store-family channels

X_test shape:  (201, 120, 1782)
y_test shape:  (201,  16, 1782)
```

Think of each training sample as a "snapshot" of 120 days of data across all 1,782 series, paired with the answer (the next 16 days) the model should learn to predict.

---

## 6. PyTorch Setup

### What is PyTorch?

PyTorch is a deep learning framework — a library that makes it easy to define, train, and run neural networks. It handles the complex mathematics (like computing gradients) automatically.

### Tensors

In PyTorch, data is stored in **tensors** — essentially multi-dimensional arrays, similar to NumPy arrays but with two key advantages:
1. They can live on a GPU for much faster computation
2. PyTorch can automatically compute gradients through them (needed for training)

```python
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
```

### GPU Acceleration (Apple Silicon / MPS)

The model automatically detects and uses Apple's **Metal Performance Shaders (MPS)** — the GPU acceleration framework for M-series chips (M1, M2, M3, etc.). If MPS isn't available, it falls back to the CPU.

```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

Training on a GPU can be 5–50× faster than on a CPU for neural network workloads.

### DataLoaders

A `DataLoader` handles feeding batches of data to the model during training. Instead of processing all 1,211 training samples at once (which would require a lot of memory), the model processes them in smaller groups called **batches**.

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
```

| Setting | Training | Test |
|---|---|---|
| Batch size | 32 | 32 |
| Shuffle | Yes | No |

> **Why shuffle training but not test?** Shuffling training batches prevents the model from accidentally learning patterns based on the order examples were fed in (which would be meaningless). Test data is never shuffled because we need predictions to match the correct rows.

---

## 7. The TCN Model Architecture

### What is a Convolutional Neural Network (CNN)?

A CNN is a type of neural network originally designed for images. It uses **filters** (small learned patterns) that slide across the input, detecting features like edges in an image or trends in a time series.

### What is a Temporal CNN (TCN)?

A **Temporal CNN** applies the same idea to sequential (time-ordered) data. Instead of 2D image filters, it uses **1D filters** that slide along the time axis. TCNs have become a strong alternative to recurrent networks (like LSTMs) for time series because they:
- Train faster (can be parallelized)
- Don't suffer from "vanishing gradients" as much
- Can look back over long time spans using dilation

### Model Architecture

```
Input
  Shape: (batch_size, 120 days, 1782 channels)
  ↓ Transpose to: (batch_size, 1782 channels, 120 days)
  │
  ├─ [Conv Layer 1]  Conv1d(1782 → 64, kernel=3, dilation=1, padding=2)
  │    ↓ ReLU activation
  │    ↓ Remove padding (trim 2 values from each end)
  │
  ├─ [Conv Layer 2]  Conv1d(64 → 64, kernel=3, dilation=2, padding=4)
  │    ↓ ReLU activation
  │    ↓ Remove padding (trim 4 values from each end)
  │
  ├─ [Conv Layer 3]  Conv1d(64 → 64, kernel=3, dilation=4, padding=8)
  │    ↓ ReLU activation
  │    ↓ Remove padding (trim 8 values from each end)
  │
  ├─ Extract last time step  →  shape: (batch_size, 64)
  │
  └─ [Fully Connected Layer]  Linear(64 → 16 × 1782 = 28,512)
       ↓ Reshape to: (batch_size, 16, 1782)

Output
  Shape: (batch_size, 16 days, 1782 channels)
```

### Understanding Each Component

#### Conv1d — 1D Convolution

A `Conv1d` layer applies a small sliding filter along the time dimension. Think of it as a pattern-detector: the filter learns to recognize things like "sales tend to dip after a spike" or "a weekly cycle is present."

- **in_channels / out_channels:** How many input signals and output feature maps. Layer 1 takes 1,782 channels and compresses them into 64 internal features.
- **kernel_size=3:** The filter looks at 3 time steps at a time.

#### Dilation — Expanding the Receptive Field

Dilation is a technique that allows a small filter to "see" further back in time without adding more parameters.

```
No dilation (dilation=1):   Looks at adjacent steps
  Input:  [t1] [t2] [t3] [t4] [t5] [t6] [t7]
  Filter:       [x]  [x]  [x]              ← kernel_size=3, sees 3 steps

Dilation=2:   Skips every other step
  Input:  [t1] [t2] [t3] [t4] [t5] [t6] [t7]
  Filter:  [x]       [x]       [x]         ← still 3 parameters, but sees span of 5

Dilation=4:   Skips 3 steps between each
  Input:  [t1] [t2] [t3] [t4] [t5] [t6] [t7] [t8] [t9]
  Filter:  [x]            [x]            [x]          ← sees span of 9
```

By stacking layers with dilation 1 → 2 → 4, the model can see short-term patterns (daily fluctuations), medium-term patterns (weekly cycles), and longer-term trends — all with a lightweight architecture.

#### Padding & Trimming

Padding adds zeros around the edges of the input so the convolution output matches the input length. After each convolution, the added padding is trimmed off to keep the temporal alignment clean.

| Layer | Dilation | Padding | Trim Each Side |
|---|---|---|---|
| Conv 1 | 1 | 2 | 2 |
| Conv 2 | 2 | 4 | 4 |
| Conv 3 | 4 | 8 | 8 |

#### ReLU Activation

After each convolution, a **ReLU (Rectified Linear Unit)** function is applied:

```
ReLU(x) = max(0, x)
```

This simply sets any negative values to zero. Without activation functions, a neural network is just a series of linear transformations — which can only learn linear patterns no matter how many layers you stack. ReLU introduces **non-linearity**, letting the network learn complex, curved relationships.

#### Fully Connected Layer

After the three convolutional layers, we take only the **last time step** from the output (the most "recent" processed features) and feed it into a single **fully connected (linear) layer**.

This layer expands 64 learned features into **16 × 1,782 = 28,512 output values** — one prediction for each of the 16 forecast days × 1,782 store-family combinations.

The output is then reshaped from a flat vector into `(batch_size, 16, 1782)`.

---

## 8. Training the Model

### Setup

```python
model     = TCNModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
```

### The Optimizer: Adam

The **Adam optimizer** is the algorithm that adjusts the model's weights to reduce the prediction error. It works by computing gradients (the direction in which weights should change) and taking small steps in the direction that lowers the loss.

> **Analogy:** Imagine you're blindfolded on a hilly landscape trying to find the lowest point (minimum loss). Adam figures out which direction is "downhill" and takes a step. It also adapts its step size based on recent history — taking smaller steps when it's been oscillating and larger steps when it's consistently moving in one direction.

The **learning rate (0.001)** controls how large each step is. Too large and you overshoot; too small and training takes forever.

### The Loss Function: RMSE

The loss function measures how wrong the model's predictions are. This model uses **Root Mean Squared Error (RMSE)**:

```
RMSE = sqrt( mean( (predicted - actual)² ) )
```

Why square the errors? Squaring penalizes large mistakes disproportionately — a prediction that's off by 10 contributes 100 to the loss, while one off by 1 contributes only 1. This pushes the model to avoid large blunders.

```python
loss = torch.sqrt(criterion(predictions, targets))  # criterion = MSELoss
```

### Training Loop

```python
for epoch in range(30):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()          # Reset gradients from previous step
        predictions = model(X_batch)   # Forward pass: make predictions
        loss = torch.sqrt(criterion(predictions, y_batch))  # Calculate error
        loss.backward()                # Backward pass: compute gradients
        optimizer.step()               # Update weights
```

Each pass through the entire dataset is called an **epoch**. The model trains for 30 epochs.

### Training Loss Over Time

| Epoch | Training Loss |
|---|---|
| 5 | 0.784 |
| 10 | 0.740 |
| 15 | 0.719 |
| 20 | 0.705 |
| 25 | 0.695 |
| 30 | 0.686 |

The loss decreases consistently with each epoch, indicating the model is learning. The loss here is on **scaled data** (after StandardScaler), so 0.686 is in units of standard deviations, not actual sales.

---

## 9. Validation

After training, the model is evaluated on the held-out test set (the most recent 20% of the data it has never seen):

```python
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = torch.sqrt(criterion(test_predictions, y_test_tensor))
    # Result: tensor(171.86)
```

### Training Loss vs Test Loss

| Metric | Value |
|---|---|
| Final Training Loss (scaled) | 0.686 |
| Test Loss | 171.86 |

The test loss appears dramatically higher than training loss. This happens because:

1. **Different scales:** Training loss is computed on normalized (scaled) data. The test loss here is computed on the **raw, original sales values** after inverse-transforming the predictions. A 171-unit RMSE in absolute sales is not directly comparable to a 0.686 RMSE in normalized space.

2. **Outliers:** Some store-family combinations have occasional extreme sales spikes (promotional events, holidays) that are hard to predict and inflate the RMSE significantly.

This is a known characteristic of retail sales forecasting — the metric gap is expected and doesn't necessarily mean the model is performing poorly in practice.

---

## 10. Final Model: Training on Full Data

Before generating the actual Kaggle predictions, the model is retrained on **100% of the available training data** (no train-test split):

```python
X_full, y_full = create_X_y(full_scaled_data, input_length=120, output_length=16)
# ... convert to tensors, train for 30 epochs
```

### Why Retrain on Full Data?

> **The logic:** During development, we held back 20% of the data to honestly evaluate the model. But once we're confident the model works, there's no reason to leave that data on the table. More training data generally means a better model. This is standard practice in Kaggle competitions.

### Full Data Training Loss

| Epoch | Loss |
|---|---|
| 5 | 0.791 |
| 10 | 0.748 |
| 15 | 0.729 |
| 20 | 0.717 |
| 25 | 0.707 |
| 30 | 0.701 |

The trajectory mirrors the validation training run, confirming the model behaves consistently.

---

## 11. Generating Predictions

With the final model trained, we generate the 16-day forecast using the **last 120 days** of all available training data as the input context.

```python
last_120_days = full_scaled_data[-120:]              # Shape: (120, 1782)
input_tensor  = torch.FloatTensor(last_120_days)
input_tensor  = input_tensor.unsqueeze(0).to(device)  # Shape: (1, 120, 1782)

final_model.eval()
with torch.no_grad():
    predictions = final_model(input_tensor)           # Shape: (1, 16, 1782)
```

### Post-Processing the Predictions

**Step 1 — Move to CPU and convert to NumPy:**
```python
predictions = predictions.to('cpu').numpy().squeeze(0)  # Shape: (16, 1782)
```
The GPU tensor is moved back to CPU memory and converted to a regular NumPy array. `squeeze(0)` removes the batch dimension since we only have one sample.

**Step 2 — Inverse scale:**
```python
predictions = scaler.inverse_transform(predictions)
```
The StandardScaler transformation is reversed, converting predictions from normalized units back to actual sales quantities.

**Step 3 — Clip negative values:**
```python
predictions = np.maximum(predictions, 0)
```
The model might occasionally predict slightly negative sales (an artifact of the scaling and linear transformations). Since sales cannot be negative in reality, any negative value is set to 0.

---

## 12. Creating the Submission File

The Kaggle submission requires a CSV with two columns: `id` and `sales`. Each `id` maps to a specific store-family-date combination.

### Step-by-Step

**1. Load `test.csv` to get the required IDs and dates:**
```python
test_df = pd.read_csv('data/test.csv')
test_df['store_family'] = test_df['store_nbr'].astype(str) + '_' + test_df['family']
```

**2. Build a prediction DataFrame (wide format):**
```python
pred_df = pd.DataFrame(
    predictions,
    index=test_dates,          # 16 dates: Aug 16–31, 2017
    columns=df_pivoted.columns # 1782 store-family columns
)
```

**3. Unpivot (stack) back to long format:**
```python
pred_long = pred_df.stack().reset_index()
pred_long.columns = ['date', 'store_family', 'sales']
```

This converts the wide matrix (16 rows × 1,782 columns) into a long table (28,512 rows × 3 columns).

**4. Merge with test.csv to recover the original IDs:**
```python
submission_df = test_df.merge(pred_long, on=['date', 'store_family'], how='left')
```

**5. Export:**
```python
submission_df[['id', 'sales']].to_csv('submission.csv', index=False)
```

### Submission Sample

```
id         sales
3000888    4.33
3000889    0.00
3000890    3.71
3000891    2313.43
3000892    0.14
...
3029399    16.60
```

Total rows: **28,512** (16 days × 1,782 store-family combinations)

---

## 13. Results Summary

### Key Metrics

| Metric | Value |
|---|---|
| Training RMSE (scaled, validation run) | 0.686 |
| Test RMSE (absolute sales, validation run) | 171.86 |
| Training RMSE (scaled, full data run) | 0.701 |
| Total predictions generated | 28,512 |
| Forecast horizon | 16 days |

### Key Design Decisions

| Decision | What Was Chosen | Why |
|---|---|---|
| Temporal order | No shuffling in split or DataLoader | Time series must respect chronological order |
| Scaling | StandardScaler (mean=0, std=1) | Handles outliers better than min-max scaling |
| Scaler fit | Training data only | Prevents data leakage from the future |
| Loss function | RMSE | Penalizes large errors more than MAE |
| Optimizer | Adam (lr=0.001) | Fast, adaptive, reliable default choice |
| Dilation | 1 → 2 → 4 | Captures short-, medium-, and longer-term patterns efficiently |
| Activation | ReLU | Simple, effective, avoids vanishing gradients |
| Final model | Trained on 100% of data | Maximizes information before submission |
| Negative clipping | `np.maximum(predictions, 0)` | Sales cannot realistically be negative |

---

## 14. Dependencies & Setup

### Requirements (`requirements.txt`)

| Package | Purpose |
|---|---|
| `pandas` | Loading CSVs, data manipulation, pivot/merge operations |
| `numpy` | Array operations, numerical computing |
| `scikit-learn` | `StandardScaler` for normalization |
| `matplotlib` | Data visualization |
| `torch` | PyTorch deep learning framework (model, training, tensors) |
| `ipykernel` | Jupyter notebook kernel |

### Installation

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Project

Open `main.ipynb` in Jupyter and run all cells from top to bottom. Ensure the `data/` directory is populated with the Kaggle competition files before running.

---

## Full Pipeline at a Glance

```
train.csv (116 MB, 3M+ rows)
    │
    ▼
Create store_family column (e.g. "3_BEVERAGES")
    │
    ▼
Pivot to wide format  →  matrix: (days × 1782)
    │
    ├── 80% Train ──┐
    └── 20% Test ───┤
                    │
                    ▼
            StandardScaler (fit on train only)
                    │
                    ▼
        Sliding window (input=120, output=16)
                    │
                    ├── X_train (1211, 120, 1782)
                    └── y_train (1211,  16, 1782)
                    │
                    ▼
        Convert to PyTorch tensors → GPU (MPS)
                    │
                    ▼
         TCN Model (3 dilated Conv1d + FC layer)
                    │
                    ▼
         Train 30 epochs with Adam + RMSE loss
                    │
                    ▼
         Validate on test set  →  evaluate performance
                    │
                    ▼
     Retrain on FULL data (100%) for final model
                    │
                    ▼
    Feed last 120 days  →  predict next 16 days
                    │
                    ▼
    Inverse scale  →  clip negatives  →  format
                    │
                    ▼
             submission.csv (28,512 rows)
```
