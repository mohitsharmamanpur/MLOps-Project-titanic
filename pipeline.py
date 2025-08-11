from src.preprocessing import preprocess_data
from src.train import train_model
from src.deploy import upload_to_s3

if __name__ == "__main__":
    X, y = preprocess_data("data/titanic-dataset.csv")
    model = train_model(X, y, model_path="titanic_model.pkl")
    upload_to_s3("titanic_model.pkl", "mohit-titanic-mlops-lw-2025", "titanic_model.pkl")
