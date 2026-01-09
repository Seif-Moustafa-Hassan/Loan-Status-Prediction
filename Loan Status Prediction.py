import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")

# Drop unnecessary column
df = df.drop("Loan_ID", axis=1)


# Drop rows with any missing values
df = df.dropna().copy()  # <-- Important to copy the cleaned dataframe

print(df.columns)

df = df[df["Self_Employed"] == "No"]

df = df[df["Education"] != "Not Graduate"]

df = df[df["Credit_History"] > 0]

df = df.drop(["Self_Employed" , "Education" , "Credit_History"] , axis=1)



# Replace '3+' in Dependents with 4
df["Dependents"] = df["Dependents"].apply(lambda x: 4 if x == "3+" else int(x))

# Label encode categorical variables
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Married"] = le.fit_transform(df["Married"])
#df["Education"] = le.fit_transform(df["Education"])
#df["Self_Employed"] = le.fit_transform(df["Self_Employed"])
df["Property_Area"] = le.fit_transform(df["Property_Area"])
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])

# Ensure numeric columns are correctly typed
#df["Credit_History"] = df["Credit_History"].astype(int)
df["ApplicantIncome"] = df["ApplicantIncome"].astype(int)
df["CoapplicantIncome"] = df["CoapplicantIncome"].astype(int)
df["LoanAmount"] = df["LoanAmount"].astype(int)
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype(int)



# Remove outliers using IQR
Q1 = df["ApplicantIncome"].quantile(0.25)
Q3 = df["ApplicantIncome"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df["ApplicantIncome"] >= lower_bound) & (df["ApplicantIncome"] <= upper_bound)]




# Remove outliers using IQR
Q1 = df["CoapplicantIncome"].quantile(0.25)
Q3 = df["CoapplicantIncome"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df["CoapplicantIncome"] >= lower_bound) & (df["CoapplicantIncome"] <= upper_bound)]



df = df[(df["LoanAmount"] < 166) & (df["LoanAmount"] > 40)]

"""

plt.figure(figsize=(8, 5))
plt.boxplot(df["LoanAmount"], vert=False)
plt.title("Boxplot of Loan Amount")
plt.xlabel("Loan Amount")
plt.grid(True)
plt.show()

"""

df = df[df["CoapplicantIncome"] >= 500]



# Add the Income_Range column just for analysis
bins = list(range(0, df["ApplicantIncome"].max() + 1000, 1000))
labels = [f"{i}-{i+999}" for i in bins[:-1]]
df["Income_Range"] = pd.cut(df["ApplicantIncome"], bins=bins, labels=labels, right=False)


df = df.drop("Income_Range", axis=1)


df = df[~(
    ((df["ApplicantIncome"] >= 0) & (df["ApplicantIncome"] < 1000)) |
    ((df["ApplicantIncome"] >= 7000) & (df["ApplicantIncome"] < 8000))
)]





# Split features and target
x = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Double-check that x and y have the same number of rows
assert len(x) == len(y), f"x has {len(x)} rows, y has {len(y)} rows"

# Split into training and test sets
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
x_cv, x_test, y_cv, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_cv = scaler.transform(x_cv)
x_test = scaler.transform(x_test)

# Train SVM model
classifier = svm.SVC(kernel="linear")
print("Training started...")
classifier.fit(x_train, y_train)
print("Training completed!")

# Make predictions
y_train_pred = classifier.predict(x_train)
y_cv_pred = classifier.predict(x_cv)
y_test_pred = classifier.predict(x_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
cv_accuracy = accuracy_score(y_cv, y_cv_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Cross-Validation Accuracy:", cv_accuracy)
print("Testing Accuracy:", test_accuracy)



