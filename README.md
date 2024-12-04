# Yoga Pose Classification and Recommendation System

## Overview

This project is designed to classify yoga poses based on their descriptions and benefits using natural language processing (NLP) techniques. The system predicts the corresponding yoga pose (Asana) from a given description of its benefits, evaluates the model’s accuracy, and predicts yoga poses for new inputs.

### **Key Features:**
- **Text Classification**: Classify yoga poses based on their benefits using a machine learning classifier.
- **Data Preprocessing**: Uses **TF-IDF** vectorization to convert yoga pose benefits into numerical features.
- **Model Evaluation**: Measures the model's performance using accuracy.
- **Prediction**: Predicts yoga poses from new descriptions of their benefits.

---

## Requirements

To run this project, ensure you have the following Python libraries installed:

- `pandas` - For data manipulation and analysis.
- `scikit-learn` - For machine learning models and evaluation metrics.
- `nltk` - For text preprocessing.
- `gensim` - For advanced NLP tasks (optional, used for word embeddings).
- `flask` - For deploying the model as an API (optional).

You can install the dependencies using `pip`:

```bash
pip install pandas scikit-learn nltk gensim flask
```

---

## Dataset

The project uses the following datasets:

- **`test.csv`**: Contains yoga poses and their corresponding benefits. This dataset is used for training and evaluating the text classification model.
- **`word_embeddings.csv`**: An optional dataset for working with word embeddings (if you choose to use them for advanced NLP tasks like similarity or clustering).

Sample data (`test.csv`):
| Asana             | Benefits                                  |
|-------------------|-------------------------------------------|
| PADOTTHANASANA    | Strengthens abdominal muscles and tones core |
| PARVATASANA       | Improves posture and flexibility           |
| ARDHA TITALI ASANA| Prepares for advanced asanas              |

---

## Project Structure

```
yoga-pose-classification/
├── data/                        # Folder to store datasets
│   ├── test.csv                 # Yoga pose dataset (required)
│   └── word_embeddings.csv     # Word embeddings data (optional)
├── ml_yoga.ipynb                # Jupyter Notebook with data exploration, model training, and evaluation
├── yoga_pose_model.py          # Python script to train, evaluate, and predict
├── requirements.txt            # File with project dependencies
└── README.md                   # Project documentation (this file)
```

---

## How to Use

### 1. **Prepare the Data**:
Ensure the dataset (`test.csv`) is placed in the correct directory or adjust the path in the code accordingly. If you want to use word embeddings, you can use the `word_embeddings.csv` file.

### 2. **Run the Model**:
Execute the `yoga_pose_model.py` script to train the model, evaluate its accuracy, and make predictions on new input:

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

### 3. **Jupyter Notebook**:
You can also explore the **`ml_yoga.ipynb`** notebook for detailed data exploration, model training, and evaluation steps.

```bash
jupyter notebook ml_yoga.ipynb
```

### 4. **Predict a New Pose**:
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

- **Model Optimization**: Experiment with other classifiers such as **Random Forest**, **Support Vector Machines (SVM)**, or deep learning models like **BERT** for better performance.
- **Hyperparameter Tuning**: Fine-tune model hyperparameters and vectorization parameters to improve accuracy.
- **Deployment**: Deploy the model as an API using **Flask** or **FastAPI** to make it accessible for real-time predictions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- The dataset was provided by the user.
- The project uses libraries such as `scikit-learn` for machine learning, `gensim` for NLP tasks, and `flask` for deployment.

