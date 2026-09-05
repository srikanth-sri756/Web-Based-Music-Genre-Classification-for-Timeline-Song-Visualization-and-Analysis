# Web-Based Music Genre Classification for Timeline Song Visualization and Analysis

A Django web application that classifies audio tracks into ten musical genres, comparing SVM, Decision Tree, Feed-Forward Neural Network and LSTM models on features extracted with librosa.

## Overview

Automatic genre tagging underpins recommendation, playlist generation and music-library analytics. This project builds the full pipeline as a web app: register, log in, train any of four models, inspect their confusion matrices, then upload a `.wav` file and get a genre prediction back.

The audio corpus follows the GTZAN layout — ten genres with sample tracks each — and features are precomputed into NumPy arrays so the web requests stay responsive.

## Supported genres

`blues` · `classical` · `country` · `disco` · `hiphop` · `jazz` · `metal` · `pop` · `reggae` · `rock`

## Features

- **User accounts** — signup and login backed by MySQL.
- **Four trainable models** — SVM (linear kernel), Decision Tree, MLP feed-forward network, and a stacked LSTM.
- **Metric reporting** — accuracy, average precision, recall and F-score rendered as an HTML table per model.
- **Confusion matrices** — 10×10 Seaborn heatmaps showing exactly which genres get confused.
- **Model caching** — the trained LSTM is serialised to `model/model.json` + `model_weights.h5` and reloaded rather than retrained.
- **Audio classification** — upload a `.wav` track and receive its predicted genre.
- **Precomputed features** — `model/X.txt.npy` and `Y.txt.npy` hold the librosa-derived feature matrix and labels.

## Model architectures

**LSTM**

```
LSTM(128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True)
LSTM(32,  dropout=0.05, recurrent_dropout=0.35)
Dense(10, softmax)

optimizer: adam  ·  loss: categorical_crossentropy  ·  batch: 16  ·  epochs: 150
```

**Baselines** — `SVC(kernel='linear')`, `DecisionTreeClassifier()`, `MLPClassifier()`.

## Tech stack

| Layer | Technology |
|---|---|
| Web framework | Django |
| Language | Python 3.7 |
| Audio | librosa |
| Deep learning | Keras / TensorFlow |
| Classical ML | scikit-learn |
| Database | MySQL via PyMySQL |
| Plots | Matplotlib, Seaborn |

## Project structure

```
.
├── manage.py                 # Django entry point
├── run.bat                   # python manage.py runserver
├── DB.txt                    # MySQL schema (MusicGenre database + signup table)
├── AudioSetData/             # Training corpus, one folder per genre
│   ├── blues/ classical/ country/ disco/ hiphop/
│   └── jazz/ metal/ pop/ reggae/ rock/
├── testMusicFiles/           # 20 unlabelled .wav samples for prediction
├── model/
│   ├── X.txt.npy, Y.txt.npy  # Precomputed features and labels
│   ├── model.json            # Serialised LSTM architecture
│   ├── model_weights.h5      # Trained LSTM weights
│   └── history.pckl          # Training history
├── MusicGenre/               # Project settings, URLs, WSGI
└── MusicGenreApp/
    ├── views.py              # Handlers, training routines, metrics
    ├── urls.py
    ├── static/               # CSS and images
    └── templates/            # Login, signup, classify, results pages
```

## Getting started

### Prerequisites

- Python 3.7
- MySQL server running on `localhost:3306`

### 1. Create the database

Run the statements in `DB.txt`:

```sql
CREATE DATABASE MusicGenre;
USE MusicGenre;
-- then the signup table definition from DB.txt
```

The app connects as `root` with an empty password — adjust the `pymysql.connect(...)` calls in `MusicGenreApp/views.py` if needed.

### 2. Install dependencies

```bash
git clone https://github.com/srikanth-sri756/9.Web-Based-Music-Genre-Classification-for-Timeline-Song-Visualization-and-Analysis.git
cd 9.Web-Based-Music-Genre-Classification-for-Timeline-Song-Visualization-and-Analysis

pip install django librosa numpy scikit-learn keras tensorflow matplotlib seaborn pymysql
```

### 3. Run

```bash
python manage.py runserver
```

or double-click `run.bat` on Windows, then open <http://127.0.0.1:8000/>.

## Usage

1. **Signup** for an account, then **Login**.
2. Train any of the four models — **SVM**, **Decision Tree**, **Feed Forward NN** or **LSTM**. Each returns a metrics table and opens a confusion-matrix heatmap.
3. **Classification** — upload a `.wav` file from `testMusicFiles/` (or your own) to get the predicted genre.

## Notes

- Database credentials are hard-coded in `views.py`; move them to environment variables before deploying.
- Confusion-matrix plots open in blocking Matplotlib windows on the server process — suitable for a local demo only.
- `keras.layers.recurrent`, `keras.utils.np_utils` and `_make_predict_function()` were removed in modern Keras; pin TF 1.x/2.x-early or migrate the imports if you upgrade.
- `settings.py` ships with `DEBUG = True` and a development secret key. Change both before exposing the app.

## License

Released for academic and educational use.
