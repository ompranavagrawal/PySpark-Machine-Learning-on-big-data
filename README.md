**Introduction**

This project aims to identify fake job postings within a dataset using PySpark, a powerful tool for handling big data. By employing different machine learning classifiers, we aim to develop a robust model that can accurately distinguish between genuine and fraudulent job listings.

**Dataset**

The dataset, "fake_job_postings.csv", consists of job listings with various attributes. Each listing is labeled as fraudulent or not, making this a binary classification problem.

**Methodology**

**Data Preprocessing**

**Data Loading**: The dataset is loaded from an HDFS path into a Spark DataFrame.

**Missing Value Analysis**: Columns with a significant percentage of missing values are identified and dropped.

**Text Cleaning**: Regular expressions are used to clean text columns, removing non-alphanumeric characters and converting text to lowercase.

**Feature Engineering**

**Text Vectorization**: The description column is tokenized and vectorized using TF-IDF to transform textual data into a numerical format suitable for machine learning models.

**Data Balancing**: The dataset is balanced by undersampling the majority class to improve model performance on minority classes.

**Model Training and Evaluation**

Four classifiers are evaluated:

**Logistic Regression**

**Linear Support Vector Classifier (LinearSVC)**

**Random Forest Classifier**

**Multilayer Perceptron Classifier**

For each model:

A Pipeline is constructed, including preprocessing and the classifier.

Hyperparameter tuning is performed using CrossValidator with a ParamGridBuilder.

Model performance is assessed using accuracy and F1 score.

**Results**

Model performance is documented, detailing the accuracy, F1 score, and best parameters for each classifier.

**Conclusion**

The project demonstrates the application of various machine learning models to detect fake job postings in a big data environment using PySpark. The effectiveness of each model varies, with detailed results providing insights into the best-performing models based on accuracy and F1 score.
