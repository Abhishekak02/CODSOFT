import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("tested.csv")

# Preprocess the data
def preprocess_data(df):
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
    return df

data = preprocess_data(data)

# Separate features and target
X = data.drop(columns=["Survived"])
y = data["Survived"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Create and train the model
model = XGBClassifier()
model.fit(X_train, y_train)

# Predict survival
def predict_survival():
    print("Enter the passenger information:")
    pclass = int(input("Pclass (1, 2, 3): "))
    sex = input("Sex (male or female): ")
    age = float(input("Age: "))
    sib_sp = int(input("Sib_Sp: "))
    parch = int(input("Parch: "))
    fare = float(input("Fare: "))
    embarked = input("Embarked (C, Q, or S): ")

    user_input = pd.DataFrame({
        "Pclass": [pclass],
        "Sex_male": [1 if sex.lower() == "male" else 0],
        "Embarked_Q": [1 if embarked.upper() == "Q" else 0],
        "Embarked_S": [1 if embarked.upper() == "S" else 0],
        "Age": [age],
        "SibSp": [sib_sp],
        "Parch": [parch],
        "Fare": [fare]
    })

    user_input = user_input[X_train.columns]
    prediction = model.predict_proba(user_input)

    plt.figure(figsize=(8, 5))
    plt.bar(["Not Survived", "Survived"], prediction[0], color=["red", "green"])
    plt.xlabel("Survival Prediction")
    plt.ylabel("Prediction Probability")
    plt.title("Passenger Survival Prediction Chances")
    plt.ylim(0, 1)

    if prediction[0][0] > prediction[0][1]:
        print("The passenger did not survive.")
    else:
        print("The passenger survived.")

    plt.show()

# Call the prediction function
predict_survival()
