import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Importing Data
data = pd.read_csv("flavour.csv")

# Encoding Gender
Enc = LabelEncoder()
Enc.fit(["Male", "Female"])  # Male - 1 & Female - 0
data["Gender"] = Enc.transform(data["Gender"])

# Making input & output
X = data.drop(columns=["Flavour"])
Y = data.drop(columns=["Age", "Gender"])

# Making Model
CModel = DecisionTreeClassifier()
CModel.fit(X, Y)

# Predicting
age = int(input("Enter age (-ve to quit) : "))

while age > 0:
    gender = int(input("Enter gender (1 for Male, 0 for Female) : "))
    print(f"Predicted flavour : {CModel.predict([[age, gender]])[0]}")
    age = int(input("Enter age (-ve to quit) : "))
