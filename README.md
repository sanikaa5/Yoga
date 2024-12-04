# Yoga Pose Classification and Recommendation System

## Overview

This project aims to classify yoga poses based on their descriptions and benefits using natural language processing (NLP) techniques. The model predicts the corresponding yoga pose (Asana) from a given description of its benefits. Additionally, it evaluates the model's accuracy and can predict new poses based on user input.

### **Key Features:**
- **Text Classification**: Classify yoga poses based on benefits using a Logistic Regression classifier.
- **Data Preprocessing**: Uses **TF-IDF** vectorization to convert yoga pose benefits into numerical features.
- **Model Evaluation**: Measures model performance using accuracy on a test dataset.
- **Prediction**: Predicts yoga poses from new input descriptions of their benefits.

---

## Requirements

To run this project, ensure you have the following Python libraries installed:

- `pandas` - For data manipulation and analysis.
- `scikit-learn` - For machine learning models and evaluation metrics.
- `nltk` - For text preprocessing.
- `gensim` - (Optional) If you need more advanced NLP methods such as Word2Vec.
- `flask` - (Optional) For deploying the model as an API.

You can install the dependencies using `pip`:

```bash
pip install pandas scikit-learn nltk gensim flask
```

---

## Dataset

The dataset used in this project is a CSV file (`test.csv`) that contains information about yoga poses and their associated benefits. The key columns in the dataset are:

- **Asana**: Name of the yoga pose.
- **Benefits**: Description of the health benefits of each yoga pose.

Sample data:
| Asana             | Benefits                                  |
|-------------------|-------------------------------------------|
| PADOTTHANASANA    | Strengthens abdominal muscles and tones core |
| PARVATASANA       | Improves posture and flexibility           |
| ARDHA TITALI ASANA| Prepares for advanced asanas              |

---

## Project Structure

```
yoga-pose-classification/
├── data/                        # Folder to store the dataset (test.csv)
│   └── test.csv                 # Yoga pose dataset
├── yoga_pose_model.py          # Python script to train, evaluate, and predict
├── requirements.txt            # File with project dependencies
└── README.md                   # Project documentation (this file)
```

---

## How to Use

### 1. **Prepare the Data**:
Ensure the dataset `test.csv` is in the root directory or adjust the path in the code.

### 2. **Run the Model**:
Execute the `yoga_pose_model.py` script to train the model, evaluate its accuracy, and make predictions on new input.

```bash
python yoga_pose_model.py
```

The script will:
1. Load the dataset and preprocess the text data.
2. Split the data into training and testing sets.
3. Vectorize the text data using **TF-IDF**.
4. Train a **Logistic Regression** model.
5. Evaluate the model’s **accuracy** on the test data.
6. Predict the yoga pose for a sample description of benefits.

### 3. **Predict a New Pose**:
You can input new yoga pose descriptions into the script to get predictions. Update the `new_input` variable with a description of your choice.

```python
new_input = "This pose strengthens the legs and improves posture."
predicted_asana = predict_new_data(clf, vectorizer, new_input)
print(f'Predicted Asana: {predicted_asana}')
```

---

## Model Evaluation

The **Logistic Regression** model is trained on the dataset, and its performance is evaluated using accuracy. Accuracy is calculated by comparing the model’s predictions to the actual yoga poses in the test set.

Sample Output:
```
Accuracy: 85.56%
Predicted Asana: Tadasana
```

---

## Future Improvements

- **Model Optimization**: You can experiment with other classifiers such as **Random Forest**, **Support Vector Machines (SVM)**, or deep learning models like **BERT** for better performance.
- **Hyperparameter Tuning**: Fine-tune model hyperparameters and vectorization parameters to improve accuracy.
- **Deployment**: You can deploy the model as an API using frameworks like **Flask** or **FastAPI** to make it accessible for real-time predictions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- The dataset was provided by the user.
- The project makes use of libraries like `scikit-learn` and `gensim` for machine learning and NLP tasks.

