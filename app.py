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

        answer = ask_openai(
            """Is the question asking to find top performing stocks? If yes, return 'yes'. If not, return 'no'.""",
            question
        )

        print(f"Answer: {answer}")

        if answer.lower() == "yes":
            answer = ask_openai("Check what type of stock the user is asking about. If he is asking about large cap stocks, return 'large'. If he is asking about mid cap stocks, return 'mid'. If he is asking about small cap stocks, return 'small'. If he is asking about all stocks or not mentioning any specific type, return 'all'. Also, check how many stocks the user is asking for- top 3, top 5 or top 10, etc. Return it too. If the user does not mention any number of stocks then return 3. So the final output will be type of stock, number- that is, these 2 values separated by a comma without any spaces. If the question is not related to stocks, return 'no'.", question)
            if answer != "no":
                typeStocks, n = answer.split(',')
                typeStocks = typeStocks.strip().lower()
                n = int(n.strip()) if n.strip().isdigit() else 3
                if n> 10:
                    n = 10

                answer = get_top_performing_stocks(typeStocks, n)
                return render_template('index.html', answer=answer, question=question)
        else:
            symbol = ask_openai('Is the question about a particular stock? If yes, only return the stocks symbol in one word (for example, if the question is about Apple, return AAPL). else return "no".', question)
            if symbol.lower() != 'no':
                try:
                    answer = get_stock_trend(symbol)
                except Exception as e:
                    answer = f"Error fetching stock data: {str(e)}"

                return render_template('index.html', answer=answer, question=question)
    
        if question:
            try:
                news = fetch_newsapi()
                reddit_posts = fetch_reddit_posts()
                finnhub = get_finnhub_news()

                formatted_news = format_context_items(news)
                formatted_reddit = format_context_items(reddit_posts)
                formatted_finnhub = format_context_items(finnhub)
                context = "\n\n".join(formatted_news + formatted_reddit + formatted_finnhub)

                answer = ask_openai(question, context)
            except Exception as e:
                answer = f"Error: {str(e)}"
        else:
            answer = "Please enter a question."
    return render_template('index.html', answer=answer, question=question)

if __name__ == '__main__':
    app.run(debug=True)