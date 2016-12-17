Daily Stock Forecast
=======

Daily Stock Forecast is a cloud based machine learning tool 
for day trading professionals. The site pulls from a current
list of ~7,000 stocks across the AMEX, Nasdaq, and NYSE
exchanges and performs epsilon-Support Vector Regression based
machine learning on historical values to forecast the open,
close, high, low, and volume of the next business day for each
stock in our expanding universe.

Daily Stock Forecast can be viewed today!<br />
<http://1.daily-stock-forecast.appspot.com/>

Screenshots of Daily Stock Forecast live and in action:<br />
![](https://github.com/DMTSource/daily-stock-forecast/blob/master/daily-stock-forecast.png)

Features
========

* Transparency of results. Each forecast comes with a 10 day performance
analysis to expose the Slope and R2 values of past predictions for each
metric. Additonaly, you can mouse over each histogram bar to see that 
metrics correlation scatter plot.

* Frontend powered by Google App Engine and Google NDB Datastore,
Google Charts, and in development is a new Polymer based site.

* Backend powered by Google Compute engine and scikit-learn to
perform machine learning on historical stock prices.

* Statistic and machine learning libraries like matplotlib, scipy,
pandas, and scikit-learn support development, analysis and
in development visualization of stock forecasts.

File Structure
============
Key files in the application hierarchy.
* daily-stock-forecast-gae
  * dailystockquant.py (App Engine homepage script)
    * indexOld.html (Current website homepage, loaded into above script w/ Jinja2 templating)
    * index.html (In development Polymer version of site)
  * forecast.py (Cron job to launch a compute engine instance to perform the forecast automatically each business day) 
* daily-stock-forecast-gce3
  * DailyForecast.py (Download historical data, runs the forecast, publishes to datastore)

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

* Python 2.7
* numpy 
* pandas
* pytz
* sk-learn
* googledatastore
* Polymer (0.5.2+)

Usage
------------
To run the forecast you must configure gcloud to the correct project and then you can run:,<br />
python DailyForecast.py # use 1 cpu,<br />
python -m scoop DailyForecast.py # Use all cpu,<br />
python -m scoop -n 16 DailyForecast.py # Use only 16 cpu,<br />

Credits
============

Daily Stock Forecast was developed by Derek M Tishler,<br />
<https://www.linkedin.com/profile/view?id=263507105>

Contact
=======

For other questions, please contact <dmtishler@gmail.com>.
