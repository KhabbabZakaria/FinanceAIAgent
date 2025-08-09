from flask import Flask, render_template, request
from helpers import *


# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    question = ""
    if request.method == 'POST':
        question = request.form.get('question', '').strip()

        if not question:
            answer = "Please ask a question!"
            return render_template('index.html', answer=answer, question=question)

        classifier = QuestionClassifier()
        classification = classifier.classify(question)

        print('classification')
        print(classification)

        try:
            if classification['intent'] == 'rank_portfolio':
                tickerList, weightList = classification['tickers'], classification['weights']
                print(tickerList, weightList)
                answer=analyze_portfolio(tickerList, weightList)

            elif classification['intent'] == 'should_buy_stock':
                tickerList = classification['tickers']
                weightList = classification['weights']
                new_stock = classification['new_stock']
                if not new_stock:
                    answer = "Please specify the stock you want to consider buying."
                else:
                    answer = should_buy_stock(tickerList, weightList, new_stock)

            elif classification['intent'] == 'top_performers':
                typeStocks, n = classification['type'], classification['count']
                answer = get_top_performing_stocks(typeStocks, n)
            
            elif classification['intent'] == 'stock_trend':
                symbol = classification['symbol']
                answer = get_stock_trend(symbol)
            
            elif classification['intent'] == 'get_finance_news':
                news = fetch_newsapi()
                reddit_posts = fetch_reddit_posts()
                finnhub = get_finnhub_news()

                formatted_news = format_context_items(news)
                formatted_reddit = format_context_items(reddit_posts)
                formatted_finnhub = format_context_items(finnhub)
                context = "\n\n".join(formatted_news + formatted_reddit + formatted_finnhub)

                answer = ask_openai(question, context)

        except Exception as e:
            answer = f"Error processing your question: {str(e)}"
        
        return render_template('index.html', answer=answer, question=question)

        
    return render_template('index.html', answer="Welcome to Fina!", question=question)

if __name__ == '__main__':
    app.run(debug=True)