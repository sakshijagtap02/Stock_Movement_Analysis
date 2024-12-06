# Stock_Movement_Analysis Based on Social Media Sentiment

## Overview
This project predicts stock price movements by analyzing sentiment from Reddit discussions. Using web scraping, Natural Language Processing (NLP), and machine learning, it processes user-generated content to extract valuable insights and forecasts stock trends.

The model uses sentiment analysis of discussions on subreddits like `r/stocks` to predict whether the stock market sentiment is positive, negative, or neutral, helping investors make informed decisions.


## Features
- Scrapes real-time data from Reddit using the PRAW API.
- Cleans and preprocesses text data for analysis.
- Performs sentiment analysis using NLTK's VADER.
- Extracts key features like sentiment polarity and word counts.
- Predicts stock movements using a Random Forest Classifier.
- Provides model evaluation metrics such as accuracy, precision, and recall.

## Technologies Used
- **Python**: Programming language.
- **PRAW**: For Reddit data scraping.
- **NLTK**: For Natural Language Processing and sentiment analysis.
- **pandas**: For data manipulation.
- **scikit-learn**: For machine learning model building.
- **matplotlib** and **seaborn**: For data visualization.

## Project Setup

### Prerequisites
1. Python 3.x installed on your machine.
2. A Reddit Developer account with API credentials (client ID, secret, username, password).

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/stock-movement-analysis.git
    cd stock-movement-analysis
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Add your Reddit API credentials in the scraping script:
    - Open `data_scraping.py` or `data_scraping.ipynb`.
    - Replace the placeholder credentials with your own:
      ```python
      reddit = praw.Reddit(
          client_id='YOUR_CLIENT_ID',
          client_secret='YOUR_CLIENT_SECRET',
          user_agent='YOUR_APP_NAME',
          username='YOUR_REDDIT_USERNAME',
          password='YOUR_REDDIT_PASSWORD'
      )
      ```

## How to Run

### 1. Scrape Reddit Data
1. Navigate to the `notebooks/` folder and open the `data_scraping.ipynb` file in Jupyter Notebook.
2. Run all cells to scrape stock-related discussions from `r/stocks` or a chosen subreddit.
3. Save the scraped data as `raw_data.csv` in the `data/` folder.

### 2. Perform Sentiment Analysis
1. Open `sentiment_analysis.ipynb` in Jupyter Notebook.
2. Run all cells to:
   - Clean and preprocess the data.
   - Perform sentiment analysis using NLTK's VADER tool.
   - Save the preprocessed data as `processed_data.csv`.

### 3. Train and Evaluate the Model
1. Open `model_training.ipynb` in Jupyter Notebook.
2. Run all cells to:
   - Load the preprocessed data.
   - Train a Random Forest Classifier on features like sentiment polarity and word count.
   - Evaluate the model using metrics such as accuracy, precision, and recall.

### 4. View Results
- Check the output of the prediction model.
- Visualizations such as sentiment distributions and model performance metrics are displayed in the notebooks.

## Results
- The Random Forest Classifier achieved an accuracy of **85%** on the test dataset.
- Positive sentiment posts correlated strongly with upward stock price movements.
- Key metrics:
  - Precision: 0.87
  - Recall: 0.84
  - F1 Score: 0.85
- Sentiment analysis results show a higher frequency of positive discussions in `r/stocks` during market uptrends.

## Challenges and Solutions
1. **Reddit API Rate Limits:**
   - Encountered limitations in the number of API calls.
   - Solution: Batched API requests and added sleep intervals to respect rate limits.

2. **Text Preprocessing:**
   - Handling noise, emojis, and incomplete sentences in Reddit posts was challenging.
   - Solution: Used Python’s regex and NLP tools for effective cleaning.

3. **Model Overfitting:**
   - The model initially overfit to the training data.
   - Solution: Tuned hyperparameters and used a larger test dataset for validation.
  
## Future Enhancements
1. Integrate data from multiple platforms like Twitter and Telegram for better predictions.
2. Use advanced NLP models like BERT or RoBERTa for more accurate sentiment analysis.
3. Develop a real-time dashboard for stock sentiment visualization.
4. Predict specific stock price movements instead of general sentiment trends.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
