1. scraper leszedi a SYMBOL változóban megadott tickere vonatkozó tweet-eket a stocktwits.com api-járól, majd beírja a stocktwits_SYMBOL.csv file-ba.
   Nagyobb mennyiséget 1-2 napig tart összeszedni az API korlátozása miatt. A file képezheti a training alapját mivel csak olyan tweet-et szed le ami meg van címkézve (Bearish, Bullish)
   A létrejött file-t a data\sentiments.csv file-ba kell másolni a training-hez
2. ml_model a  data\sentiments.csv-n tanulva képzi a modelét model\stocktwits_modelNB.pkl
3. get_stock_price futtatásával a baktest előtt le kell szedni a ticker-hez tartozó árakat (data\stock_price.pkl)
4. nltk_download.py futtatni ha még nem volt letöltve a szükséges nltk szó korpusz
5. preprocessing_backtesting.py elvégi a sentiment.csv predict-jét a 2. pontban képzett model segítségével, és leképzi a baktest eredményeit
    data\stock_price_merge.csv'
    data\strategy.csv'
6. ezután indítható a dash\app ami vizuálisan megjeleníti az eredményeket