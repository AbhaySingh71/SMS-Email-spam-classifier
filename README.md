# 📩 SMS/Email Spam Classifier

## Overview
The **SMS/Email Spam Classifier** is a machine learning-based application designed to classify messages as either **Spam** or **Not Spam**. It uses Natural Language Processing (NLP) techniques for preprocessing and a trained classification model to predict the spam status of user input messages. This app is deployed on Streamlit for easy interaction and real-time predictions.

## Features
- **Text Preprocessing:** The application processes the input message by:
    - Converting to lowercase
    - Tokenizing the text into words
    - Removing stopwords and punctuation
    - Stemming words using Porter Stemmer
- **Prediction:** Using a machine learning model, the app classifies the input message as either:
    - **Spam** 🚨
    - **Not Spam** ✅
- **User-friendly Interface:** The app is powered by Streamlit, providing a simple and intuitive user interface for interaction.

## Technologies Used
- **Python**: Programming language used for the application logic and machine learning.
- **Streamlit**: Framework for building the user interface and deploying the app.
- **Scikit-learn**: For creating the machine learning model and performing text vectorization (TF-IDF).
- **NLTK (Natural Language Toolkit)**: Used for text preprocessing, including tokenization, stopword removal, and stemming.
- **Pickle**: Used to save and load the trained model and vectorizer for predictions.

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/sms-email-spam-classifier.git
    cd sms-email-spam-classifier
    ```

2. **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the necessary NLTK datasets:** Run the following Python code to download the required NLTK datasets:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

5. **Run the app:**
    ```bash
    streamlit run app.py
    ```

## Model Training
To train the spam classification model:
1. **Data:** Prepare a labeled dataset containing SMS and Email messages categorized as "spam" or "ham" (not spam).
2. **Preprocessing:** Clean the text data (lowercase, tokenization, stopword removal, stemming).
3. **Feature Extraction:** Use TF-IDF Vectorization to convert text data into numerical features.
4. **Model:** Train a classifier (e.g., Random Forest, Naive Bayes) on the processed data.
5. **Save Model and Vectorizer:** Save the trained model and vectorizer using pickle for future use.

## How to Use
1. **Enter a message:** In the text box, type the message you want to classify.
2. **Click 'Predict':** The model will analyze the input and predict if the message is **Spam** or **Not Spam**.
3. **View Result:** Based on the model's prediction, the app will display either **🚨 Spam!** or **✅ Not Spam!**.

## Contributing
Feel free to fork this project, make improvements, or suggest new features. If you find any bugs or issues, please open an issue on GitHub.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Streamlit** for providing a fantastic platform to build and deploy this app easily.
- **Scikit-learn** for the machine learning tools that make the spam classification possible.
- **NLTK** for simplifying text preprocessing tasks.
