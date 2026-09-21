# 💬 Sentiment Analysis using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/NLP-Sentiment%20Analysis-0A66C2?style=for-the-badge&logo=python&logoColor=white" alt="NLP">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/NLTK-Natural%20Language%20Processing-154F5B?style=for-the-badge" alt="NLTK">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
</p>

<p align="center">
  <b>🧠 An NLP-based project for analyzing sentiment in Twitter and application user-review data</b>
</p>

<p align="center">
  <i>Text preprocessing → Feature Extraction → Machine Learning → Sentiment Analysis → Visualization</i>
</p>

---

## 📌 Project Overview

**Sentiment Analysis** is a Natural Language Processing (NLP) project that explores how machine learning techniques can be applied to classify and analyze textual data according to sentiment.

The primary dataset used in the project is a **Twitter sentiment dataset**, containing text along with sentiment categories representing negative, neutral, and positive sentiment.

The project also explores **Google Play Store application and user-review datasets** to visualize sentiment distributions, ratings, reviews, and application-related characteristics.

The workflow covers:

* 🧹 Text preprocessing
* 🔤 Stopword removal
* 🌱 Lemmatization
* 🔢 Bag-of-Words feature extraction
* 🤖 Naive Bayes classification
* 🧪 SVM classification experiment
* 📊 Confusion matrix visualization
* 📈 Sentiment distribution analysis
* 📉 Exploratory visualization of application reviews and ratings
* 🧊 3D visualization experiments

---

# 🎯 Objectives

The main objectives of this project are to:

* Understand the fundamentals of **Natural Language Processing**
* Preprocess raw textual data for machine learning
* Remove common English stopwords
* Apply lemmatization to normalize words
* Convert text into numerical features
* Apply machine learning classification techniques
* Analyze sentiment categories in textual data
* Visualize sentiment distributions
* Explore relationships between application ratings, reviews, and sentiment
* Gain practical experience with NLP and text classification workflows

---

# 📊 Datasets

The project works with three datasets.

## 🐦 1. Twitter Sentiment Dataset

The primary classification dataset is:

```text
Twitter_Data.csv
```

Example records contain:

| Text                                       | Category |
| ------------------------------------------ | -------: |
| `when modi promised minimum government...` |       -1 |
| `talk all the nonsense and continue...`    |        0 |
| `what did just say vote for modi...`       |        1 |

The sentiment category is represented numerically:

```text
-1 → Negative
 0 → Neutral
 1 → Positive
```

This dataset is used for the main NLP preprocessing and machine-learning workflow.

---

## 📱 2. Google Play Store Apps Dataset

```text
apps.csv
```

This dataset contains application-level information such as:

* App name
* Category
* Rating
* Number of reviews
* Size
* Number of installs
* Type
* Price
* Content rating
* Genres
* Last updated date
* Android version

Example:

| App                         | Category       | Rating | Reviews | Type |
| --------------------------- | -------------- | -----: | ------: | ---- |
| Photo Editor & Candy Camera | ART_AND_DESIGN |    4.1 |     159 | Free |
| Coloring book moana         | ART_AND_DESIGN |    3.9 |     967 | Free |

---

## 💬 3. Google Play Store User Reviews Dataset

```text
user_reviews.csv
```

This dataset contains application reviews along with sentiment-related information.

Important columns include:

| Column                   | Description                   |
| ------------------------ | ----------------------------- |
| `App`                    | Application name              |
| `Translated_Review`      | User review text              |
| `Sentiment`              | Positive / Neutral / Negative |
| `Sentiment_Polarity`     | Polarity score                |
| `Sentiment_Subjectivity` | Subjectivity score            |

This dataset is primarily used for **sentiment exploration and visualization**.

---

# 🧠 NLP Workflow

The main Twitter sentiment-analysis pipeline follows:

```text
             🐦 Twitter Dataset
                    │
                    ▼
          ┌──────────────────┐
          │  Text Cleaning   │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Lowercasing      │
          │ Stopword Removal │
          │ Lemmatization    │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ CountVectorizer  │
          │ Bag-of-Words     │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Train / Test     │
          │ Split            │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Classification   │
          │ Experiments      │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Evaluation &     │
          │ Visualization    │
          └──────────────────┘
```

---

# 🧹 Text Preprocessing

A preprocessing function is used to prepare the Twitter text for machine learning.

```python
def preprocess_text(text):
    if pd.isna(text):
        return ''

    text = text.lower()

    stop_words = set(stopwords.words('english'))

    text = ' '.join(
        [word for word in text.split()
         if word not in stop_words]
    )

    lemmatizer = WordNetLemmatizer()

    text = ' '.join(
        [lemmatizer.lemmatize(word)
         for word in text.split()]
    )

    return text
```

### Preprocessing Steps

#### 1️⃣ Handle Missing Text

Missing text values are converted to empty strings.

#### 2️⃣ Lowercasing

All text is converted to lowercase to maintain consistency.

#### 3️⃣ Stopword Removal

Common English words are removed using NLTK's English stopword list.

#### 4️⃣ Lemmatization

Words are reduced to their base dictionary form using:

```python
WordNetLemmatizer()
```

---

# 🔢 Feature Extraction

Machine learning algorithms require numerical input.

The project uses **CountVectorizer** to convert text into a Bag-of-Words representation.

```python
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(
    twitter_df['clean_text']
)
```

The resulting sparse matrix represents the frequency of words appearing in each document.

---

# ✂️ Train-Test Split

The dataset is divided into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Configuration

| Parameter     | Value |
| ------------- | ----: |
| Training data |   80% |
| Testing data  |   20% |
| Random state  |    42 |

The fixed random state allows the experiment to be reproduced.

---

# 🤖 Machine Learning Models

Two classification approaches are explored in the notebook:

### 1. Multinomial Naive Bayes

### 2. Support Vector Machine (SVM)

---

# 🧮 Multinomial Naive Bayes

The project uses:

```python
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()

nb_model.fit(
    X_train,
    y_train
)
```

Predictions are generated using:

```python
y_pred_nb = nb_model.predict(X_test)
```

Multinomial Naive Bayes is particularly suitable for text classification problems involving discrete word-count features.

---

# 📊 Confusion Matrix

A confusion matrix is generated to visualize the classification results of the Naive Bayes model.

```python
conf_matrix = confusion_matrix(
    y_test,
    y_pred_nb
)
```

The matrix is visualized using a Seaborn heatmap.

```python
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    xticklabels=nb_model.classes_,
    yticklabels=nb_model.classes_
)
```

The visualization helps examine how predictions are distributed across the sentiment classes.

---

# 🧪 Support Vector Machine Experiment

The notebook also attempts to train an **SVM classifier** using:

```python
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)
```

Before training, missing target values are checked.

The notebook reports:

```text
Number of NaN values in target: 7
```

These rows are removed before the subsequent train-test split.

### ⚠️ Notebook Execution Note

The current SVM section contains a preprocessing/vectorization issue.

The code attempts to pass an already-vectorized sparse matrix into `CountVectorizer.fit_transform()`, resulting in:

```text
AttributeError:
'csr_matrix' object has no attribute 'lower'
```

This occurs because `CountVectorizer` expects raw text documents, while `X_train` at that point is already a sparse numerical matrix.

The intended workflow should be either:

```text
Raw Text
   ↓
CountVectorizer
   ↓
Sparse Matrix
   ↓
SVM
```

or, when `X_train` is already vectorized:

```text
Vectorized X_train
   ↓
SVM
```

This is documented here deliberately so the repository accurately reflects the current notebook state.

---

# 📱 Play Store Review Analysis

The project additionally explores sentiment information from Google Play Store user reviews.

The review text is processed using the same preprocessing function:

```python
user_reviews_df['Translated_Review'] = (
    user_reviews_df['Translated_Review']
    .apply(preprocess_text)
)
```

The sentiment distribution is then visualized.

### Sentiment Categories

```text
🟢 Positive
🟡 Neutral
🔴 Negative
```

This provides a visual overview of how user opinions are distributed across the review dataset.

---

# 📊 Sentiment Distribution

A count plot is used to visualize the number of reviews belonging to each sentiment category.

```python
sns.countplot(
    data=user_reviews_df,
    x='Sentiment',
    hue='Sentiment'
)
```

This helps identify the relative distribution of positive, neutral, and negative user reviews.

---

# 📈 Exploratory Visualizations

The notebook contains several additional visualization experiments.

## 🧊 3D Twitter Sentiment Visualization

A 3D scatter plot is created using:

* Polarity
* Subjectivity
* Dataset index

The visualization provides a three-dimensional representation of the generated sentiment-related variables.

> **Note:** The polarity and subjectivity values in this visualization are simulated from the existing sentiment category and random values. They should therefore be treated as an exploratory visualization rather than measured NLP sentiment scores.

---

## 📱 3D App Ratings vs Reviews

The Play Store application dataset is used to construct a 3D surface visualization involving:

* ⭐ Rating
* 💬 Number of Reviews
* 💰 Price

The price field is cleaned and converted to numerical form before visualization.

---

## 📦 3D User Review Sentiment Visualization

The review sentiment categories are mapped numerically:

```python
sentiment_mapping = {
    'positive': 1,
    'neutral': 0,
    'negative': -1
}
```

The resulting values are used in a 3D visualization across applications.

---

## 📊 3D Histogram of Ratings and Reviews

A two-dimensional histogram is generated from:

* Application ratings
* Number of reviews

and displayed using 3D bars.

This provides an exploratory view of the distribution of ratings and review counts.

---

# 🛠️ Technology Stack

| Technology              | Purpose                         |
| ----------------------- | ------------------------------- |
| 🐍 **Python**           | Programming language            |
| 🐼 **Pandas**           | Data loading and manipulation   |
| 🔢 **NumPy**            | Numerical computation           |
| 📊 **Matplotlib**       | Visualization                   |
| 🎨 **Seaborn**          | Statistical visualization       |
| 🤖 **Scikit-learn**     | Machine learning                |
| 🧠 **NLTK**             | Natural Language Processing     |
| 📓 **Jupyter Notebook** | Development and experimentation |

---

# 📦 Libraries Used

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
```

The notebook also uses:

```text
CountVectorizer
MultinomialNB
SVC
classification_report
confusion_matrix
WordNetLemmatizer
stopwords
```

---

# 📁 Project Structure

```text
SENTIMENT-ANALYSIS-PROJECT/
│
├── 📓 Sentiment Analysis Notebook
├── 🐦 Twitter_Data.csv
├── 📱 apps.csv
├── 💬 user_reviews.csv
└── 📖 README.md
```

> Dataset filenames and notebook filenames should match the files currently present in the repository.

---

# ▶️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SASWATIMATHAN/OIBSIP-Intern-Saswati-Sentiment-Analysis-PROJECT-4-LEVEL-1.git
```

### 2️⃣ Navigate to the Project

```bash
cd OIBSIP-Intern-Saswati-Sentiment-Analysis-PROJECT-4-LEVEL-1
```

### 3️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

### 4️⃣ Download Required NLTK Resources

The notebook uses NLTK stopwords and WordNet lemmatization.

Run:

```python
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
```

### 5️⃣ Open the Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Then open the project notebook and execute the cells sequentially.

---

# ⚠️ Reproducibility Note

The original notebook currently contains **Windows-specific local file paths**, for example:

```text
C:\Users\MATHAN\Downloads\SENTIMENT ANALYSIS\
```

For portability, the datasets should ideally be referenced using relative paths:

```python
twitter_df = pd.read_csv('Twitter_Data.csv')
apps_df = pd.read_csv('apps.csv')
user_reviews_df = pd.read_csv('user_reviews.csv')
```

This allows the notebook to run on other systems after cloning the repository.

---

# 💡 Key Concepts Demonstrated

This project provides practical exposure to:

* 🧠 Natural Language Processing
* 🧹 Text preprocessing
* 🔤 Token-level text normalization
* 🛑 Stopword removal
* 🌱 Lemmatization
* 🔢 Bag-of-Words representation
* 🤖 Text classification
* 📊 Confusion matrices
* 🐦 Social-media sentiment analysis
* 📱 App-review sentiment analysis
* 📈 Exploratory data visualization
* 🧊 3D data visualization

---

# 🚀 Future Improvements

The project can be extended by:

* 🔧 Fixing and completing the SVM pipeline
* ⚡ Using **TF-IDF** instead of only Bag-of-Words
* 🧪 Comparing Naive Bayes, SVM, Logistic Regression, and other classifiers
* 📊 Reporting accuracy, precision, recall, and F1-score consistently
* ⚖️ Handling class imbalance
* 🧹 Improving text cleaning with punctuation and URL removal
* 😊 Handling emojis and social-media-specific language
* #️⃣ Processing hashtags and mentions
* 🔤 Exploring n-grams
* 🧠 Experimenting with modern NLP models
* 🌐 Building an interactive sentiment-analysis application
* 🔌 Deploying the trained model through an API

---

# 🎓 Internship Project

This project was developed as part of the:

**Oasis Infobyte Internship Program (OIBSIP)**

### Project 4 — Level 1

**Sentiment Analysis**

The project focuses on applying fundamental NLP and machine-learning concepts to real-world textual and user-review data.

---

# 👤 Author

## **Saswati Mathan**

🎓 **M.Tech — Electronics & Communication Engineering**

🔬 Specialization: **Communication**

### 🔗 GitHub

<p>
<a href="https://github.com/SASWATIMATHAN">
<img src="https://img.shields.io/badge/GitHub-SASWATIMATHAN-181717?style=for-the-badge&logo=github&logoColor=white">
</a>
</p>

---

# ⭐ Project Summary

```text
🐦 Twitter Data
      ↓
🧹 Text Preprocessing
      ↓
🔤 NLP Normalization
      ↓
🔢 Bag-of-Words
      ↓
🤖 Machine Learning
      ↓
📊 Sentiment Classification
      ↓
📈 Visualization
```

<p align="center">
  <b>💬 Turning text into insights with Natural Language Processing</b>
</p>

<p align="center">
  <i>Explore • Process • Classify • Visualize</i>
</p>

---
