# Multi-Class News Classifier

A production-ready pipeline for categorizing news articles into four categories: World, Sports, Business, and Sci/Tech. This project uses a fine-tuned BERT model and is wrapped in FastAPI and Docker for easy deployment.

## Technical Overview
* **Model:** bert-base-uncased fine-tuned on the AG News dataset.
* **Optimization:** Implemented label smoothing (0.15) and froze the first 6 encoder layers to balance training speed and accuracy.
* **Metrics:** Evaluated using Macro-F1 score to ensure reliable performance across all classes.
* **Training:** Developed on Kaggle using dual T4 GPUs with gradient accumulation and early stopping.

## Quick Start

### 1. Installation
```
git clone [https://github.com/MuhammadSafeer786/Multi_Class_News_Text_Classification.git](https://github.com/MuhammadSafeer786/Multi_Class_News_Text_Classification.git)
cd Multi_Class_News_Text_Classification
pip install -r requirements.txt
```
### 2. Run the API
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
### 3. Docker Deployment
```
docker build -t news-classifier .
docker run -p 8000:8000 news-classifier
```
### API Usage
Once the app is running, visit http://localhost:8000/docs to test the endpoints.

Example Request:

```
{
  "text": "The tech giant announced a new quantum processor today."
}
```
Example Response:

```
{
  "category": "Sci/Tech",
  "confidence": 0.97
}
```
### Optimization Details
- Gradient Accumulation: Used to simulate larger batch sizes on limited GPU memory.

- Early Stopping: Configured with a patience of 2 to avoid overfitting.

- Layer Freezing: Keeps the base language model stable while the classification head learns the specific task.
