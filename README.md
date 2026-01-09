Project Overview:

This project aims to predict whether a loan application will be approved or rejected based on applicant
demographic and financial information. The focus of the project is on data preprocessing, feature filtering,
and validation-driven model improvement to achieve reliable generalization on unseen data.

Problem Statement:

Financial institutions need accurate and consistent methods to assess loan eligibility. Poor data quality and
outliers can significantly reduce prediction reliability. This project addresses these challenges by applying
domain-based data cleaning and structured validation before model training.

Approach:

Cleaned the dataset by handling missing values and removing irrelevant or low-quality records
Applied domain-driven feature filtering to improve data reliability
Encoded categorical variables and standardized numerical features
Removed outliers using IQR-based statistical analysis
Split the data into training, validation, and testing sets using stratified sampling
Trained a linear Support Vector Machine (SVM) classifier
Evaluated performance across all splits to ensure model stability and generalization

Results:

Improved testing accuracy by approximately 8% through preprocessing and validation
Achieved consistent performance across training, validation, and testing sets
Reduced overfitting and improved robustness on unseen loan applications

Technologies:

Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
