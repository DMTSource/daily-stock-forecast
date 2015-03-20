
from zipline.api import order_target, record, symbol, history, add_history, \
                        set_slippage, set_commission, commission, slippage,\
                        symbol, get_open_orders, order_target, get_order, \
                        order_target_percent
from sklearn.svm import SVR
import numpy as np

#inline run
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_from_yahoo
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
from zipline.finance import trading

def initialize(context):

    # Let's set a look up date inside our backtest to ensure we grab the correct security
    #set_symbol_lookup_date('2015-01-01')
    

    # Use a very liquid set of stocks for quick order fills
    context.symbol = symbol('SPY')
    #context.stocks = symbols(['TWX','AIG','PSX','EMC','YHOO','MDY','TNA','CHK','FXI',
    #                            'PEP','SBUX','VZ','VWO','TWC','HAL','MDLZ','CAT','TSLA',
    #                            'MU','PM','WYNN','MET',NOV BRK_B SNDK ESRX YELP])
    #set_universe(universe.DollarVolumeUniverse(99.5, 100))
    #set_benchmark(symbol('SPY'))
    
    # set a more realistic commission for IB, remove both this and slippage when live trading in IB
    set_commission(commission.PerShare(cost=0.014, min_trade_cost=1.4))
    
    # Default slippage values, but here to mess with for fun.
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.25, price_impact=0.1))
        
    # Use dicts to store items for plotting or comparison
    context.next_pred_price = {} # Current cycles prediction
    
    #Change us!
    context.history_len              = 500    # How many days in price history for training set
    context.out_of_sameple_bin_size  = 2
    context.score_filter             = -1000.0
    context.action_to_move_percent   = 0.0

    # Register 2 histories that track daily prices,
    # one with a 100 window and one with a 300 day window
    add_history(context.history_len, '1d', 'price')
    context.i = 0
    
def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < context.history_len:
        return

    context.exchange_time = algo.get_datetime().strftime('%Y-%m-%d')

    fire_sale(context,data)

    svr_trading(context, data)
    
    """
    # Save values for later inspection
    record(AAPL=data[sym].price,
           short_mavg=short_mavg[sym],
           long_mavg=long_mavg[sym])
    """
def svr_trading(context, data):

    # Historical data, lets get the past days close prices for
    pastPrice = history(bar_count=context.history_len, frequency='1d', field='price')

    # Make predictions on universe
    for stock in data:
        # Make sure this stock has no existing orders or positions to simplify our portfolio handling.
        if check_if_no_conflicting_orders(stock) and context.portfolio.positions[stock].amount == 0:
            
            #This is a scoring system for our model, we only trade when confident our model is wicked awesome 
            full_series = np.array(pastPrice[stock].values)
            l           = context.out_of_sameple_bin_size
            power = 1 #N where X^n for weight function
            
            # Create bins of X len to hold as out of sample data, average score(error) of these is a decent measure of fit.
            prediction_history = []
            for i in np.arange(context.history_len/context.out_of_sameple_bin_size):
                #Index of current in same, and out of sample data.
                # 3 cases of this slicing
                if   i == 0:
                    #First run, only two bins to work with(First OOSD bin, and the rest of the data)
                    ISD = full_series[l:]
                    OOSD = full_series[:l]
                    X = np.arange(l,len(full_series))

                    # use a variable weight (~0 - 1.0)
                    weight_training = np.power(np.arange(l,len(full_series),dtype=float), power)[::-1]/np.power(np.arange(l,len(full_series),dtype=float), power)[::-1].max()
                    # use a variable weight, focus on next day prediction (~0 - 1.0 - ~0)
                    weight_score = np.concatenate((np.power(np.arange(1,l+1,dtype=float), power)/np.power(np.arange(1,l+1,dtype=float), power).max(),
                                                   np.power(np.arange(l+1,len(full_series)+1,dtype=float), power)[::-1]/np.power(np.arange(l+1,len(full_series)+2,dtype=float), power)[::-1].max()))
                    """print len (weight_training)
                    print weight_training
                    print len (weight_score)
                    print weight_score
                    print exit()"""
                elif i == context.history_len/context.out_of_sameple_bin_size - 1:
                    #Last run, only two bins to work with(Last OOSD bin, and the rest of the data)
                    ISD = full_series[:-l]
                    OOSD = full_series[-l:]
                    X = np.arange(0,len(full_series)-l)

                    # use a variable weight (~0 - 1.0)
                    weight_training = np.power(np.arange(l,len(full_series),dtype=float)+1, power)/np.power(np.arange(l,len(full_series),dtype=float)+1, power).max()
                    # use a variable weight, focus on next day prediction (~0 - 1.0 - ~0)
                    weight_score = np.concatenate((np.power(np.arange(1,len(full_series)-l+1,dtype=float), power)/np.power(np.arange(1,len(full_series)-l+2,dtype=float), power).max(),
                                                   np.power(np.arange(1,l+1,dtype=float), power)[::-1]/np.power(np.arange(1,l+1,dtype=float), power)[::-1].max()))
                    """print len (weight_training)
                    print weight_training
                    print len (weight_score)
                    print weight_score
                    print exit()"""
                else:
                    #Any other run, we have a sandwhich of OOSD in the middle of two ISD sets so we need to aggregate.
                    ISD = np.concatenate((full_series[:(l*i)], full_series[l*(i+1):]))
                    OOSD = full_series[l*i:l*(i+1)]
                    X = np.concatenate(( np.arange(0,(l*i)), np.arange(l*(i+1),len(full_series)) ))

                    # use a variable weight (~0 - 1.0)
                    weight_training = np.concatenate(( np.power(np.arange(1, l*i+1, dtype=float), power)/np.power(np.arange(1, l*i+1, dtype=float), power).max(),
                                                       np.power(np.arange(l*(i+1), len(full_series), dtype=float), power)[::-1]/np.power(np.arange(l*(i+1), len(full_series),dtype=float), power)[::-1].max() ))
                    # use a variable weight, focus on next day prediction (~0 - 1.0 - ~0)
                    weight_score = np.concatenate(( np.power(np.arange(1, l*(i+1)+1, dtype=float), power)/np.power(np.arange(1, l*(i+1)+1, dtype=float), power).max(),
                                                    np.power(np.arange(l*(i+1), len(full_series), dtype=float), power)[::-1]/np.power(np.arange(l*(i+1), len(full_series)+1, dtype=float), power)[::-1].max() ))
                    """print len (weight_training)
                    print weight_training
                    print len (weight_score)
                    print weight_score
                    exit()"""
                
                # Domain and range of training data
                #X = np.arange(len(ISD))
                X = np.atleast_2d(X).T
                y = ISD

                # Domain of prediction set
                #x = np.atleast_2d(np.linspace(0, len(ISD)+len(OOSD)-1, len(ISD)+len(OOSD))).T
                #x = np.atleast_2d(np.linspace(len(ISD) ,len(ISD)+len(OOSD)-1, len(OOSD))).T
                x = np.atleast_2d(np.linspace(0, len(full_series)-1, len(full_series))).T
                
                # epsilon-Support Vector Regression using scikit-learn
                # Read more here: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
                SVR_model = SVR(kernel='rbf', C=100, gamma=.01)
                SVR_model.fit(X,y, weight_training)
                y_predSVR = SVR_model.predict(x)
                
                if np.isnan(full_series).any() or np.isinf(full_series).any():
                    print(stock + " Failed due to data INF or NAN")
                    y_score = 0
                    break
                else:
                    y_score = SVR_model.score(x, full_series)#, sample_weight=weight_score) #y_predSVR[-len(OOSD):] np.atleast_2d(y_predSVR).T

                    #log.debug(y_score)
                
                prediction_history.append(y_score)
                
            score = np.mean(y_score)

            # If we are studying one stock, lets plot its correlation regression results
            if len(data) == 1:
                record(Ideal=1.0, Score=score) #Slope=slope, R_value=r

            
            # Store the prediction for comparison with the rest of the universe
            #   Measure accuracy as the mean of the distance to the ideal value of 
            #   the r2 and slope from past vs predicted price correlation regression
            if score >= context.score_filter:
                
                #The model was accepted, make a forecast
                
                #form domain and range of test data(we leave no out of sameple data out since we already scored the model)
                X = np.arange(context.history_len)
                X = np.atleast_2d(X).T
                y = np.array(pastPrice[stock].values)

                # Domain of predection set. We only need to predict the next close price.
                x = np.atleast_2d(np.linspace(len(y), len(y), 1)).T
                """log.debug(X)
                log.debug(len(X))
                log.debug(x)
                log.debug(len(x))
                exit()"""
                
                # use a linearly peaking weight, focus on next day prediction (~0 - 1.0 - ~0)
                #weight_training = np.power(np.arange(1,context.history_len+1, dtype=float), power)/np.power(np.arange(1,context.history_len+1, dtype=float), power).max()
                #weight_training = np.exp(np.arange(1,context.history_len+1, dtype=float))/np.exp(np.arange(1,context.history_len+1, dtype=float)).max()
                
                # epsilon-Support Vector Regression using scikit-learn
                # Read more here: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
                SVR_model = SVR(kernel='rbf', C=100, gamma=.01)
                SVR_model.fit(X, y)#, weight_training)
                y_predSVR = SVR_model.predict(x)
                
                context.next_pred_price[stock] = y_predSVR[-1]
            else:
                #Case where stock is left in dict and we dont want to use it, so remove it.
                if stock in context.next_pred_price:
                    del context.next_pred_price[stock]
            

    # Count number of trades so we can split the availible cash properly
    number_of_trades_today = 0
    for stock in data:
        # Make sure this stock has no existing orders or positions to simplify our portfolio handling
        # Also check that we have a prediction stored in the dict
        if check_if_no_conflicting_orders(stock) and \
           context.portfolio.positions[stock].amount == 0 and \
           stock in context.next_pred_price:
            # If we plan to move on this stock, take count of it(explained more in actual buy statement below)(Make sure these match both buy statements.
            if (percent_change(context.next_pred_price[stock], pastPrice[stock][-1])  >= context.action_to_move_percent and \
               percent_change(context.next_pred_price[stock], data[stock]['price'])   >= context.action_to_move_percent) or \
               (percent_change(context.next_pred_price[stock], pastPrice[stock][-1])  <= -context.action_to_move_percent and \
                 percent_change(context.next_pred_price[stock], data[stock]['price']) <= -context.action_to_move_percent):
                number_of_trades_today += 1
    #

    #Lets use record to plot how  many securities are traded on each day.       
    if len(data) >= 2:
        record(number_of_stocks_traded=number_of_trades_today)

    #Make buys and shorts if the predicted close change is bigger than our tollerance, same with current price to avoid opening gaps.
    for stock in data:
        # Make sure this stock has no existing orders or positions to simplify our portfolio handling
        # Also check that we have a prediction stored in the dict
        if check_if_no_conflicting_orders(stock) and context.portfolio.positions[stock].amount == 0 and stock in context.next_pred_price:

            #Go long if we predict the close price will change more(upward) than our tollerance, 
            # apply same filter against current price vs predicted close in case of gap up/down.
            if percent_change(context.next_pred_price[stock], pastPrice[stock][-1]) >= context.action_to_move_percent and \
               percent_change(context.next_pred_price[stock], data[stock]['price']) >= context.action_to_move_percent:

                # Place an order, and store the ID to fetch order info
                orderId    = order_target_percent(stock, 1.0/number_of_trades_today)
                # How many shares did we just order, since we used target percent of availible cash to place order not share count.
                shareCount = get_order(orderId).amount

                # We can add a timeout time on the order.
                #context.duration[orderId] = exchange_time + timedelta(minutes=5)

                # We need to calculate our own inter cycle portfolio snapshot as its not updated till next cycle.
                value_of_open_orders(context, data)
                availibleCash = context.portfolio.cash-context.cashCommitedToBuy-context.cashCommitedToSell

                print("+ BUY {0:,d} of {1:s} at ${2:,.2f} for ${3:,.2f} / ${4:,.2f} @ {5:s}"\
                         .format(shareCount,
                                 stock,data[stock]['price'],
                                 data[stock]['price']*shareCount, 
                                 availibleCash,
                                 context.exchange_time))

            #Go short if we predict the close price will change more(downward) than our tollerance, 
            # apply same filter against current price vs predicted close incase of gap up/down.
            elif percent_change(context.next_pred_price[stock], pastPrice[stock][-1]) <= -context.action_to_move_percent and \
                 percent_change(context.next_pred_price[stock], data[stock]['price']) <= -context.action_to_move_percent:

                #orderId    = order_target_percent(stock, -1.0/len(data))
                orderId    = order_target_percent(stock, -1.0/number_of_trades_today)
                # How many shares did we just order, since we used target percent of availible cash to place order not share count.
                shareCount = get_order(orderId).amount

                # We can add a timeout time on the order.
                #context.duration[orderId] = exchange_time + timedelta(minutes=5)

                # We need to calculate our own inter cycle portfolio snapshot as its not updated till next cycle.
                value_of_open_orders(context, data)
                availibleCash = context.portfolio.cash-context.cashCommitedToBuy+context.cashCommitedToSell

                print("- SHORT {0:,d} of {1:s} at ${2:,.2f} for ${3:,.2f} / ${4:,.2f} @ {5:s}"\
                         .format(shareCount,
                                 stock,data[stock]['price'],
                                 data[stock]['price']*shareCount, 
                                 availibleCash,
                                 context.exchange_time))

#################################################################################################################################################

# Helper functions, allot of which is taken from the Quantopian documentation and forums(thanks Quantopian team for the great examples).
# Thread on these helper functions here: https://www.quantopian.com/posts/helper-functions-getting-started-on-quantopian
# Omg like why didnt I use lambdas? You should! Check the thread above!

def check_if_no_conflicting_orders(stock):
    # Check that we are not already trying to move this stock
    open_orders = get_open_orders()
    safeToMove  = True
    if open_orders:
        for security, orders in open_orders.iteritems():
            for oo in orders:
                if oo.sid == stock:
                    if oo.amount != 0:
                        safeToMove = False
    return safeToMove
    #

def check_invalid_positions(context, securities):
    # Check that the portfolio does not contain any broken positions
    # or external securities
    for sid, position in context.portfolio.positions.iteritems():
        if sid not in securities and position.amount != 0:
            errmsg = \
                "Invalid position found: {sid} amount = {amt} on {date}"\
                .format(sid=position.sid,
                        amt=position.amount,
                        date=get_datetime())
            raise Exception(errmsg)
            
def end_of_day(context, data):
    # cancle any order at the end of day. Do it ourselves so we can see slow moving stocks.
    open_orders = get_open_orders()
    
    if open_orders or context.portfolio.positions_value > 0.:
        #log.info("")
        print("*** EOD: Stoping Orders & Printing Held ***")

    # Print what positions we are holding overnight
    for stock in data:
        if context.portfolio.positions[stock].amount != 0:
            print("{0:s} has remaining {1:,d} Positions worth ${2:,.2f}"\
                     .format(stock,
                             context.portfolio.positions[stock].amount,
                             context.portfolio.positions[stock].cost_basis\
                             *context.portfolio.positions[stock].amount))
    # Cancle any open orders ourselves(In live trading this would be done for us, soon in backtest too)
    if open_orders:  
        # Cancle any open orders ourselves(In live trading this would be done for us, soon in backtest too)
        for security, orders in open_orders.iteritems():
            for oo in orders:
                print("X CANCLED {0:s} with {1:,d} / {2:,d} filled"\
                                     .format(stock,
                                             oo.filled,
                                             oo.amount))
                cancel_order(oo)
    #

def fire_sale(context, data):
    # Sell everything in the portfolio, at market price
    show_spacer = False
    for stock in data:
        if context.portfolio.positions[stock].amount != 0:
            order_target(stock, 0.0)
            value_of_open_orders(context, data)
            availibleCash = context.portfolio.cash-context.cashCommitedToBuy-context.cashCommitedToSell
            print("  * Exit {0:,d} of {1:s} at ${2:,.2f} for ${3:,.2f} / ${4:,.2f}  @ {5:s}"\
                         .format(int(context.portfolio.positions[stock].amount),
                                 stock,
                                 data[stock]['price'],
                                 data[stock]['price']*context.portfolio.positions[stock].amount,
                                 availibleCash,
                                 context.exchange_time))
            show_spacer = True
    if show_spacer:
        print('') #This just gives us a space to make reading the 'daily' log sections more easily 
    # 

def percent_change(new, old):
    return ((new-old)/old)*100.0
    
def value_of_open_orders(context, data):
    # Current cash commited to open orders, bit of an estimation for logging only
    context.currentCash = context.portfolio.cash
    open_orders = get_open_orders()
    context.cashCommitedToBuy  = 0.0
    context.cashCommitedToSell = 0.0
    if open_orders:
        for security, orders in open_orders.iteritems():
            for oo in orders:
                # Estimate value of existing order with current price, best to use order conditons?
                if(oo.amount>0):
                    context.cashCommitedToBuy  += oo.amount * data[oo.sid]['price']
                elif(oo.amount<0):
                    context.cashCommitedToSell += oo.amount * data[oo.sid]['price']
    #
#################################################################################################################################################

def show_results(algo, data, results):
    br = trading.environment.benchmark_returns
    bm_returns = br[(br.index >= start) & (br.index <= end)]
    results['benchmark_returns'] = (1 + bm_returns).cumprod().values
    results['algorithm_returns'] = (1 + results.returns).cumprod()
    #sharpe = [risk['sharpe'] for risk in algo.risk_report['one_month']]
    #print("Monthly Sharpe ratios: {0}".format(sharpe))

    #print("ideal netpnl: " + str(round(results.ideal[-1], 2)))
    actual = results.portfolio_value - algo.portfolio.starting_cash
    print("actual netpnl: " + str(actual[-1]))

    fig = plt.figure()#1, figsize=(8, 6))
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.0, top=0.96)

    ax1 = fig.add_subplot(311, ylabel='Cumulative Returns')
    results[['algorithm_returns', 'benchmark_returns']].plot(ax=ax1, sharex=True)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.legend(loc=0)

    ax2 = fig.add_subplot(312, ylabel='Price')
    data[algo.symbol].plot(ax=ax2, color='green')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.legend(loc=0)

    ax3 = fig.add_subplot(313, ylabel='Score')
    results.Score.plot(ax=ax3, color='red')
    results.Ideal.plot(ax=ax3, color='blue')
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.legend(loc=0)

    plt.gcf().set_size_inches(18, 8)
    plt.show()


if __name__ == '__main__':
    start   = datetime(2012, 1, 1, 0, 0, 0, 0, pytz.utc)
    end     = datetime(2015, 3, 1, 0, 0, 0, 0, pytz.utc)
    algo    = TradingAlgorithm(initialize=initialize, handle_data=handle_data, capital_base=10000)
    data    = load_from_yahoo(stocks=[algo.symbol], indexes={},
                           start=start, end=end).dropna()
    results = algo.run(data)
    show_results(algo, data, results)
