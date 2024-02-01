import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pkl

def create_model(data):
    x = data.drop(["diagnosis"], axis=1)
    y = data["diagnosis"]
    
    # Scale The Data
    scaler = StandardScaler()    
    x = scaler.fit_transform(x)
    
    # Split the Data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    # Train The Model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    # Test The Model
    y_pred = model.predict(x_test)
    print("The accuracy of our model is: ", accuracy_score(y_test, y_pred))
    print(" ***** " * 5)
    print("The classification report: \n", classification_report(y_test, y_pred))
    
    
    return model,scaler
         

def get_clean_data():
    data = pd.read_csv("DataSet/data.csv")
    print(data.head())
 
    data = data.drop(["Unnamed: 32", "id"], axis=1)
 
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    
    return data


def main():
    data = get_clean_data()
    print(data.info())
    
    model, scaler = create_model(data)
    
    with open("Model/model.pkl", "wb") as f:
        pkl.dump(model, f)
        
    with open("Model/scaler.pkl", "wb") as f:
        pkl.dump(scaler, f)    

if __name__ == '__main__':
    main()
