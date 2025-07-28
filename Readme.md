## This is a simple Finance Assistant AI Agent.
#### I can ask it about financial news, about a particular stock, or questions like 'Tell me about top 5 Large stocks now'. It fetches some of the questions' answers from newsAPI, Redit, etc. For direct stock related questions, a python script using yfinance runs in the backend.






### How to install

First install the dependencies.
```
pip install -r requirements.txt
```
Then make sure you have the APIs for `NEWSAPI_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, `OPENAI_API_KEY`, `FINNHUB_API_KEY` and store them in a .env file in the same directory. 

Next run the flask app with
```
python app.py
```

### This is not an app to suggest one on which stocks to trade! :)

