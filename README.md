Daily Stock Forecast
=======

Daily Stock Forecasts optimizes and ranks machine learning models to predict the intraday movement of the stock market for the top 10 US Equities by Market Cap and a number of popular indicies.

<http://daily-stock-forecast.com/>

Screenshots of Daily Stock Forecast live and in action:<br />
![](https://github.com/DMTSource/daily-stock-forecast/blob/master/daily-stock-forecast.png)

Features
========

Every trading day, DSF builds a number of classification models using historical candle+volume data. Each model's hyperparameters are optimized as well as the length of the lookback period per sample. Classification reprots are generated using test data. The f1 score is used to rank models.

File Structure
============
Key files in the application hierarchy.
* polymer-site
  * a simple Polymer Starter Kit is used to build a responsive website.

* backend
    * forecast generation script & helpers

Installation
============

The frontend runs on a Google App Engine instance. It utilizes
python, WebApp2, Jinja2 templating, JQuery, Google Charts, and 
soon Polymer and web components.

The backend and analysis can run locally if the datastore writing 
is disabled, but the current datastore exchange expects that the 
forecast is performed "inside the project" on a Google Compute 
Engine instance with the ability to securely access the Datastore.

Dependencies
------------

* Python 2.7+
* numpy 
* pandas
* pytz
* scikit-learn
* googledatastore
* Polymer 2

Usage
------------

python daily-stock-forecast.py


Credits/Contact
============

Daily Stock Forecast was developed by Derek M Tishler,<br />
<https://www.linkedin.com/in/derekmtishler/>

