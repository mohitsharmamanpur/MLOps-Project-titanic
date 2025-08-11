# Titanic MLOps Project

## Steps to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure AWS credentials:
   ```bash
   aws configure
   ```
3. Run the pipeline:
   ```bash
   python pipeline.py
   ```

This will:
- Preprocess Titanic dataset
- Train Logistic Regression model
- Save model as `titanic_model.pkl`
- Upload it to your S3 bucket
# MLOps-Project-titanic
