# SpamDetectAI: NLP & ML-Based Spam Classification Intelligent Message Filtering System
# Author

Rahul Ashwanth
## Project Overview
This project implements a machine learning-based spam detection system. By utilizing a dataset of text messages labeled as "spam" or "ham" (non-spam), multiple machine learning algorithms are applied and compared to identify the most effective model for this task. The goal is to maximize classification accuracy while ensuring robustness and reliability.
## Key Features
- Data Preprocessing: Techniques such as text cleaning, tokenization, and vectorization using TF-IDF were applied to transform raw text data into a suitable format for modeling.
* Exploratory Data Analysis (EDA): Insights into the dataset were obtained using visualizations and statistical summaries to understand the characteristics of spam and non-spam messages.
* Model Implementation: Several machine learning algorithms were implemented and tuned to optimize their performance:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
- Model Evaluation: Metrics such as
  - Accuracy,
  - Precision,
  - Recall, 
  - F1-score,
  - Confusion Matrix,
  - Log loss,
  - ROC & AUC Curve,
  - Cross-Validation were calculated for each algorithm to determine the best fit for spam detection.
## Dataset
The project uses a publicly available dataset of text messages. Each entry consists of:
- A label: "spam" or "ham"
- The message text
## Tools & Libraries Used
- Python for scripting and model development
- Pandas & NumPy for data manipulation
- Seaborn & Matplotlib for visualization
- Scikit-learn for machine learning models
## Steps Involved
**1. Data Preprocessing**
- Load and explore the dataset.
- Handle missing or irrelevant data.
- Convert text labels into numerical form.
  
**2. Exploratory Data Analysis (EDA)**
- Visualize the distribution of spam vs. ham messages.
- Analyze word frequency and patterns.
  
**3. Feature Engineering**
- Convert text messages into numerical representations (TF-IDF, Count Vectorization, etc.).
  
**4. Machine Learning Models Used**
- Naïve Bayes
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Decision Tree
  
**5. Model Evaluation**
- Train and test each model.
- Compare accuracy, precision, recall, and F1-score.
- Select the best-performing model.
## How It Works
- Preprocessing:
  - Remove noise (e.g., punctuation, stopwords).
  - Convert text to lowercase.
  - Apply vectorization using TF-IDF to represent text numerically.
- Modeling:
  - Train multiple models on the processed dataset.
  - Use cross-validation to fine-tune hyperparameters.
- Evaluation:
  - Compare models using classification metrics.
  - Select the algorithm with the highest F1-score for final deployment.
## Results
The model with the best performance achieved:
- Accuracy: 99%
- F1-Score: 95%
- Precision: 100%
- Recall: 94%
- Log Loss: 38%
- AUC: 97%
- Cross-Validation: 98%

## Technologies Used
- Programming Language: Python
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
 
# Conclusion: Best Fit for Spam Detection
* Naive Bayes (MultinomialNB) is often the best starting point for spam detection due to its simplicity, speed, and effectiveness for text classification tasks, especially when features are discrete (word counts).
* Logistic Regression works well for text classification and can outperform Naive Bayes in some cases, especially if the features have complex relationships.
* Support Vector Machines (SVM) and Random Forests can be used for more complex models if needed, but they require more computational power and may not be as interpretable as Naive Bayes.
### Recommendation from the project based:
* Start with Multinomial Naive Bayes for a quick, reliable baseline.
* Logistic Regression could be a good alternative if you want to capture more nuanced relationships between features and the target.
* Experiment with SVM or Random Forests if your model is underperforming and you suspect more complex patterns exist in the data.
## Future Scope
- Experiment with deep learning models (LSTMs, Transformers).
- Use advanced NLP techniques for better text representation.
- Optimize models using hyperparameter tuning.
## How to Use
- Clone the repository.
- Install the required dependencies using pip install -r requirements.txt.
- Run the Jupyter Notebook to train and test the models.







