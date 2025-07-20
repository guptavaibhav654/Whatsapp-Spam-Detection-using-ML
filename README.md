# 📱 WhatsApp Spam Detection Using Machine Learning
Live Demo:https://whatsapp-spam-detection-using-ml.streamlit.app/

A machine learning-based system designed to classify WhatsApp messages as **spam** or **ham (non-spam)**. Leverages data preprocessing, feature extraction, and model training to accurately detect unwanted messages.

## 🎯 Key Features

- **Data Cleaning & Preprocessing**: Text normalization, stop word removal, punctuation stripping.
- **Feature Extraction**: TF-IDF vectorization for transforming message text into meaningful features.
- **Model Training**: Trains several classification models (e.g., Naive Bayes, Logistic Regression).
- **Model Evaluation**: Evaluates accuracy, precision, recall, F1-score, and confusion matrix.
- **Spam vs Ham Predictions**: Classify new WhatsApp messages into spam or ham.

## 🛠️ Tech Stack

- **Language**: Python 3.x  
- **Libraries**:  
  - `pandas`, `numpy` — Data manipulation  
  - `scikit-learn` — Model training and evaluation  
  - `matplotlib`, `seaborn` — Visualization  
- **Dataset**: Pre-collected labeled WhatsApp messages (spam vs ham)

## 📁 Project Structure

/
├── data/
│ ├── spam_messages.csv # Labeled dataset (spam/ham)
│ └── cleaned_data.csv # Cleaned and preprocessed messages
├── notebooks/
│ └── exploration.ipynb # EDA, preprocessing steps, visualizations
├── models/
│ └── spam_classifier.pkl # Trained model saved with pickle
├── src/
│ ├── preprocess.py # Text cleaning functions
│ ├── train_model.py # Model training pipeline
│ └── predict.py # Script for loading model and making predictions
├── requirements.txt # Required Python packages
└── README.md # Project readme

## 🚀 Getting Started

### 1. Clone the Repository
git clone https://github.com/guptavaibhav654/Whatsapp-Spam-Detection-using-ML.git
cd Whatsapp-Spam-Detection-using-ML
2. Install Dependencies
pip install -r requirements.txt
3. Preprocess Data & Train Model
python src/train_model.py --data data/spam_messages.csv
4. Run Predictions
python src/predict.py "Free entry in 2 a weekly competition to win FA Cup final tickets"
# → Output: SPAM / HAM
📊 Model Evaluation
Accuracy

Precision & Recall

F1-score

Confusion Matrix

(Add actual metrics here based on your evaluation)

🧠 How It Works
Preprocessing: Text messages are cleaned (lowercased, stripped of non-alphanumeric characters, stop words removed).

Feature Extraction: TF-IDF transforms messages into numeric vectors for classification.

Model Training: A classifier (e.g., Naive Bayes) is trained to distinguish between spam and ham.

Prediction: The model predicts labels for new chat messages with confidence scores.

📌 Limitations & Next Steps
Dataset Bias: Model accuracy can be impacted by dataset imbalance.

Phrase Variable Spams: Needs periodic retraining with updated data.

Future Improvements:

Integration into WhatsApp through a bot or API

SMS/email spam detection

Replace TF-IDF with word embeddings (e.g., Word2Vec, BERT)

🛠 Contributions
Contributions are welcome! Feel free to:

Fork the repo

Create a feature branch

Submit a pull request

👤 Authors
Vaibhav Gupta – @guptavaibhav654

📄 License
This project is licensed under the MIT License. Enjoy and share freely!

📫 Contact
Have questions or feedback? Contact me at your-email@example.com!
