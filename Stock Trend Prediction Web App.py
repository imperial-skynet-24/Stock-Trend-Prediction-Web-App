import streamlit as st, yfinance as yf, pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from plotly import graph_objs as go, subplots
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title = 'Stock Forecast Web Application', page_icon = '📈', layout = 'wide', initial_sidebar_state = 'auto')

with st.sidebar :
	st.markdown("# Introduction :")
	st.markdown(f"""&nbsp; &nbsp; &nbsp; &nbsp; *Greetings Trader!* &nbsp; Thank-you for using our [stock trend forecasting application.](#stock-trend-forecast-web-app) Kindly refer to the guidelines
					given below for effective usage	of the application.
					\n&nbsp; &nbsp; &nbsp; &nbsp; *We hope you find your desired results and wish you success for a safe and proifitable trading!*""")
	st.text('')

	st.markdown("## Important : ")
	st.markdown("""> The amount of data selected *between start and end date* is the data which the forecasting model uses for training and extracting trends. Hence it is advised that the duration
					of the training data be at least **one year,** if a forecast is being made for the next **six to eight months,** or if the forecast is for the next **few weeks,** the duration of
					training data should preferrably be a **few months.** This is done in order to avoid **underfitting** *(very less data)* or **overfitting** *(excess of data),* which can compromise
					forecast accuracy.""")

	st.text('')
	st.text('')

	st.markdown("### General guidelines :")
	st.markdown(f"""\n1) Select the ticker symbol of desired stock from the *dropdown box,* or kindly enter the ticker symbol manually into the *adjacent text-box* if not available in dropdown and
			  				press the **Submit stock data** button.

					\n2) If you wish to view the data of a new stock that is available in the dropdown options, kindly *clear any data you may have entered data manually* into the adjacent text-box,
						as the data entered in the textbox is *submitted by default* upon clicking the **Submit stock data** button.

					\n3) Enter the desired date-range for the stock data using the *calendar widgets.* *(the stock data for today, i.e. {datetime.today().strftime('%Y-%m-%d')} will be unavailable until the
	 					market has closed and the closing price is available.)*
		  				\n &nbsp; &nbsp; &nbsp; &nbsp; ***Tip :** Select at least 2 month time-span to ensure sufficient data for the analytical engines.*

					\n4) Select any one of the *5 types of charts* available in the dropdown in [data overview], make use of **Plotly's** in-built chart tools, to *zoom in/out, pan, autoscale, reset axes*
	 					and *save the plot* as a *.png format image.* The attribute selection option is available only for line and scatter charts.

					\n5) Expand and view the *raw tabular data* within the application that also supports searching and column-wise sorting, alternatively the data may even be downloaded in CSV format.

					\n6) Adjust the forecast to a desired duration using the *sliders* provided, the *dropdown* can enable selection of forecast attribute, whilst *radio buttons* can adjust forecast
	 					parameters. Similar to stock data, forecast data too can be downloaded as a CSV file.

					\n7) The **Forecast Analytics** section has two parts, namely :
	 				\n &nbsp; &nbsp; &nbsp; &nbsp; i. **Performance Metrics :**
		 			\n &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Consists of a table that provides the several metrics over &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; the specifed time horizon along with a plot
			  			having user-defined metric parameter.

					\n &nbsp; &nbsp; &nbsp; &nbsp; ii. **Plot Components :**  Consists of four visualisations, namely :
	 						\n &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; *a. General Trend*
		   					\n &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; *b. Weekly Trend*
							\n &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; *c. Yearly Trend*
				   			\n &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; *d. Daily Trend*

					\n8) In [Performance Test,](#performance-test) select a suitable duration between start and end dates for training and testing data as specified in the **Important** guideline.
						Parametric selection nd adjustment is similar to the *forecast* section. Important error metrics will reflect the results of the performance test at the bottom.

					\n9) In the [Advanced Analytics](#advanced-analytics) section, we have,
		  					\n***i. Data Description :*** *Summary statistics (downloadable)*
							\n***ii. Stationarity Tests :*** *KPSS Test*
			   				\n***iii. Auto-correlation and partial auto-correlation :*** *Dynamic user-defined parameters with downloadable data
				   			\n***iv. Seasonal Decomposition :*** *Observed, Trend, Seasonal and Residual plots with dynamic parameters*""")

st.markdown("<h1 style='text-align: center;'>Stock Trend Forecast Web App</h1>", unsafe_allow_html = True)

st.text('')
st.text('')

with st.container() :
	with st.form(key = 'stock_submission') :
		stock_dropdown, stock_input = st.columns(2, gap = 'medium')

		stocks = ('TCS.NS', 'TATAMOTORS.NS', 'HDB', 'ADANIENT.NS', 'ICICIBANK.NS', 'ITC.NS', 'APOLLOHOSP.NS', 'BHARTIARTL.BO', 'KOTAKBANK.NS', 'LICI.NS', 'BAJFINANCE.NS', 'RELIANCE.NS', 'INFY', 'WIT', 'JIORX')
		selected_stock = stock_dropdown.selectbox('**Select stock ticker symbol for prediction :**', stocks, key = 'dropdown_stock')
		manual_stock = stock_input.text_input('**Or enter stock ticker symbol here if not available in drop-down :**', placeholder = 'TICKER', key = 'manual_stock')

		submit = st.form_submit_button(f"Submit stock data")

		if manual_stock : selected_stock = manual_stock

		if submit : st.markdown(f"You have submitted data request for **{selected_stock}** data!")

st.text('')

with st.container() :
	@st.cache_data
	def load_data(ticker, start_date = None, end_date = None):
		data = yf.download(ticker, start = start_date, end = end_date)
		data.reset_index(inplace = True)
		return data

	whole_data = load_data(selected_stock)
	ipo_launch = pd.Timestamp(whole_data.Date.min()).date()

	with st.container() :
		@st.cache_data(experimental_allow_widgets = True)
		def date_input() :
			start_date_column, end_date_column = st.columns(2, gap = 'medium')

			start_date = start_date_column.date_input(label = '**Enter start date :**',
													value = (date.today() - relativedelta(months = 2)) - timedelta(weeks = 1),
													min_value = ipo_launch, max_value = (date.today() - timedelta(days = 1)) - timedelta(weeks = 1),
													key = 'input_start_date', help = 'Enter the start date to view the stock from')

			end_date = end_date_column.date_input(label = '**Enter end date :**', value = date.today() - relativedelta(days = 1), min_value = start_date + timedelta(weeks = 1),
												max_value = date.today() - timedelta(days = 1), key = 'end_date_input',
												help = 'End date to view the stock to *(can be a day later than start date)*')

			return start_date, end_date

		START, END = date_input()

	st.text('')

	data_load_state = st.text(f'Loading {selected_stock} data...')
	data = whole_data[(whole_data['Date'] >= pd.to_datetime(START)) & (whole_data['Date'] <= pd.to_datetime(END))]
	data_load_state.write(f'**{selected_stock}** data from **{START}** to **{END}** containing **{data.shape[0]} records** has been successfully loaded!')

st.text('')
st.text('')

st.subheader(f'Overview - {selected_stock} data from *{START}* to *{END}* :')
with st.container() :
	data_vis, raw_data = st.columns([3, 1], gap = 'medium')
 
	def line_chart(attribute_1, attribute_2) :
		line_chart = go.Figure(layout = {'yaxis' : {'autorange' : True, 'fixedrange' : False}})
		line_chart.add_trace(go.Scatter(x = data['Date'], y = data[attribute_1], name = attribute_1))
		line_chart.add_trace(go.Scatter(x = data['Date'], y = data[attribute_2], name = attribute_2))
		line_chart.layout.update(title_text='Time Series Line Chart', xaxis_title = 'Date', yaxis_title = 'Open & Close', xaxis_rangeslider_visible=True)
		data_vis.plotly_chart(line_chart, use_container_width = True)

	def scatter_plot(attribute_1, attribute_2) :
		scatter_plot = go.Figure(layout = {'yaxis' : {'autorange' : True, 'fixedrange' : False}})
		scatter_plot.add_trace(go.Scatter(x = data['Date'], y = data[attribute_1], name = attribute_1, mode = 'markers', marker = dict(size = 3, color = 'green')))
		scatter_plot.add_trace(go.Scatter(x = data['Date'], y = data[attribute_2], name = attribute_2, mode = 'markers', marker = dict(size = 3, color = 'red')))
		scatter_plot.layout.update(title_text = 'Time Series Scatter Plot', xaxis_title = 'Date', yaxis_title = f'{selected_stock} Prices', xaxis_rangeslider_visible = True, yaxis = dict(fixedrange = False))
		data_vis.plotly_chart(scatter_plot, use_container_width = True)

	def waterfall_chart() :
		data['Change'] = data['Close'] - data['Open']
		colors = ['green' if value > 0 else 'red' for value in data['Change']]
		waterfall_chart = go.Figure(data = go.Waterfall(x = data['Date'], measure = ['relative'] + ['delta']*(len(data) - 2) + ['total'],
														y = data['Change'], base = 0,
														decreasing = {'marker' : {'color' : 'red'}},
														increasing = {'marker' : {'color' : 'green'}},
														totals = {'marker' : {'color' : 'blue'}}),
									layout = go.Layout(title = 'Waterfall Chart', xaxis_rangeslider_visible = True, xaxis_title = 'Date',
														yaxis = dict(title = 'Price Change', fixedrange = False)))
		data_vis.plotly_chart(waterfall_chart, use_container_width = True)

	def candlestick_chart() :
		candlestick_chart = go.Figure(data = go.Candlestick(x = data['Date'], open = data['Open'], high = data['High'], low = data['Low'], close = data['Close'],
															increasing = dict(line=dict(color='green')),
															decreasing = dict(line=dict(color='red'))))

		candlestick_chart.layout.update(title_text='Time Series Candlestick Chart', xaxis_title = 'Date', yaxis_title = f'{selected_stock} Prices',
										xaxis_rangeslider_visible = True, yaxis = dict(fixedrange = False))
		data_vis.plotly_chart(candlestick_chart, use_container_width = True)

	def ohlc_chart() :
		ohlc_chart = go.Figure(data = go.Ohlc(x = data['Date'], open = data['Open'], high = data['High'], low = data['Low'], close = data['Close']),
							layout = go.Layout(title = 'OHLC Chart', xaxis = dict(title = 'Date'), yaxis = dict(title = 'Stock Price')))
		ohlc_chart.layout.update(title_text = 'OHLC Chart', xaxis_title = 'Date', xaxis_rangeslider_visible = True,
								yaxis = dict(title = f'{selected_stock} Prices', fixedrange = False))
		data_vis.plotly_chart(ohlc_chart, use_container_width = True)

	def heiken_ashi_chart() :
		data['HA_Close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
		data['HA_Open'] = (data['Open'].shift(1) + data['Close'].shift(1)) / 2
		data['HA_High'] = data[['High', 'HA_Open', 'HA_Close']].max(axis=1)
		data['HA_Low'] = data[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

		heiken_ashi_chart = go.Figure(data = go.Candlestick(x = data['Date'], open = data['HA_Open'], high = data['HA_High'], low = data['HA_Low'], close = data['HA_Close'],
									increasing = dict(line=dict(color='green')),
									decreasing = dict(line=dict(color='red'))))
		heiken_ashi_chart.update_layout(title = 'Heikin Ashi Chart', xaxis_title='Date', yaxis = dict(title = f'{selected_stock} Prices', fixedrange = False), hovermode='x', xaxis_rangeslider_visible=True)

		data_vis.plotly_chart(heiken_ashi_chart, use_container_width = True)

	st.text('')

	chart_type = data_vis.selectbox(label = '**Select your desired chart type :**', options = ('Line Chart', 'Scatter Plot', 'Waterfall Chart', 'OHLC', 'Candlestick', 'Heiken Ashi'))

	if chart_type == 'Line Chart' :
		attribute_1, attribute_2 = data_vis.columns(2, gap = 'large')
		attribute_1, attribute_2 = attribute_1.selectbox(label = 'Select attribute 1 :', options = data.columns[1:]), attribute_2.selectbox(label = 'Select attribute 2 :', options = data.columns[1:])
		line_chart(attribute_1, attribute_2)
	elif chart_type == 'Scatter Plot' :
		attribute_1, attribute_2 = data_vis.columns(2, gap = 'large')
		attribute_1, attribute_2 = attribute_1.selectbox(label = 'Select attribute 1 :', options = data.columns[1:]), attribute_2.selectbox(label = 'Select attribute 2 :', options = data.columns[1:])
		scatter_plot(attribute_1, attribute_2)
	elif chart_type == 'Candlestick' : candlestick_chart()
	elif chart_type == 'OHLC': ohlc_chart()
	elif chart_type == 'Waterfall Chart' : waterfall_chart()
	elif chart_type == 'Heiken Ashi' : heiken_ashi_chart()

	raw_data.text('')
	raw_data.text('')

	raw_data.expander(label = 'Expand to view raw data :').dataframe(data.set_index('Date'), use_container_width = True)
	raw_data.download_button(label = 'Download Data', data = data.to_csv().encode('utf-8'), file_name = (f'{selected_stock} - {START} to {END}.csv'))

st.subheader(f'Data Forecast of {selected_stock} data from *{START}* to *{END}* :')
with st.container() :
	forecast_durations, forecast_dates_data = st.columns(2, gap = 'medium')
	day_week, month_year = forecast_durations.columns(2, gap = 'medium')
	forecast_chart, forecast_params = st.columns([3, 1], gap = 'medium')

	forecast_days = day_week.slider('Days of prediction: ', 0, 6)
	forecast_weeks = day_week.slider('Weeks of prediction: ', 0, 3)
	forecast_months = month_year.slider('Months of prediction:', 0, 11)
	forecast_years = month_year.slider('Years of prediction:', 0, 10)
	period = forecast_years * 365 + forecast_months * 30 + forecast_weeks * 4 + forecast_days * 7

	forecast_params.text('')
	forecast_params.text('')
	forecast_params.text('')

	data_attribute = forecast_params.selectbox(label = 'Select attribute to be forecasted :', options = data.columns[1:])
	forecast_params.text('')
	forecast_params.text('')
	include_history = forecast_params.radio("Include historical data :", [True, False], horizontal = True, help = f'Inclusion historical data lying between {START} and {END}')
	forecast_params.text('')
	daily_seasonality = forecast_params.radio("Select daily seasonality :", [True, False, 'auto'], horizontal = True, help = 'Set daily seasonality for the forecast model')
	forecast_params.text('')
	weekly_seasonality = forecast_params.radio("Select weekly seasonality :", [True, False, 'auto'], horizontal = True, help = 'Set monthly seasonality for the forecast model')
	forecast_params.text('')
	yearly_seasonality = forecast_params.radio("Select yearly seasonality :", [True, False, 'auto'], horizontal = True, help = 'Set yearly seasonality for the forecast model')

	@st.cache_data
	def prophet_modelling(data, data_attribute, daily_seasonality, weekly_seasonality, yearly_seasonality, period, include_history) :
		data_train = data[['Date', data_attribute]]
		data_train = data_train.rename(columns = {'Date': 'ds', data_attribute : 'y'})
		model = Prophet(daily_seasonality = daily_seasonality, weekly_seasonality = weekly_seasonality, yearly_seasonality = yearly_seasonality)
		model.fit(data_train)
		future = model.make_future_dataframe(periods = period, include_history = include_history)
		forecast = model.predict(future)

		return model, forecast

	model, forecast = prophet_modelling(data, data_attribute, daily_seasonality, weekly_seasonality, yearly_seasonality, period, include_history)

	# Show and plot forecast
	forecast_dates_data.expander(label = 'Expand to view tabular forecasted data :').dataframe(forecast.set_index('ds'), use_container_width = True)
	forecast_dates_data.download_button(label = 'Download Data', data = forecast.to_csv().encode('utf-8'), file_name = (f'Forecast of {selected_stock} - {START} to {END}.csv'))

	forecast_chart.text('')
	forecast_chart.text('')

	def model_forecast(data_model, data_forecast) :
		forecast_chart.markdown(f"<h6 style = 'text-align: left;'> Forecast plot for {selected_stock} from {END} to {END + timedelta(days = forecast_days) + timedelta(weeks = forecast_weeks) + relativedelta(months = forecast_months)  + relativedelta(years = forecast_years)} :</h6>", unsafe_allow_html = True)
		forecast_chart.plotly_chart(plot_plotly(data_model, data_forecast), use_container_width = True)

	model_forecast(model, forecast)


st.subheader('Forecast analytics :')
with st.container() :
	with st.expander(label = 'Expand to view :') :
		performance_metric, plot_components = st.columns(2, gap = 'large')
  
		def forecast_analytics() :
			performance_metric.markdown("<h2 style='text-align: center;'> Performance Metrics </h2>", unsafe_allow_html = True)
			performance_metric.text('')

			performance_metric.subheader('Statistics :')
			performance_metric.text('')

			initial = f'{data.shape[0]} days'
			horizon = f'{data.shape[0] // 3} days'
			performance_data = cross_validation(model = model, initial = initial, horizon = horizon, parallel = 'threads')
			performance_metric_data = performance_metrics(performance_data)

			performance_metric_data['horizon'] = performance_metrics(performance_data)['horizon'].astype('string')
			performance_metric.dataframe(performance_metric_data.set_index('horizon'), use_container_width = True)

			performance_metric.download_button(label = 'Download Data', data = performance_metric_data.set_index('horizon').to_csv().encode('utf-8'),
											file_name = (f'Performance Metric - {selected_stock} - {START} to {END}.csv'), key = 'performance_metric')

			performance_metric.text('')
			performance_metric.text('')

			cross_val_metric = performance_metric.selectbox(label = '**Select metric type :**', options = ('mse', 'rmse', 'mae', 'mape', 'mdape', 'smape', 'coverage'))
			performance_metric.text('')
			performance_metric.pyplot(plot_cross_validation_metric(performance_data, metric = cross_val_metric, color = 'red', point_color = 'blue'))


			plot_components.markdown("<h2 style='text-align: center;'> Plot Components </h2>", unsafe_allow_html = True)
			plot_components.text('')
			plot_components.text('')
			plot_components.write(model.plot_components(forecast))

		forecast_analytics()


st.text('')
st.text('')
st.text('')

st.subheader('Performance Test :')
with st.expander(label = 'Expand to view') :
	st.text('')

	start_date_column, end_date_column = st.columns(2, gap = 'medium')

	train_start_date = start_date_column.date_input(label = '**Enter start date for training data** :',
													value = datetime.strptime('2022-01-01', '%Y-%m-%d'),
													min_value = ipo_launch, max_value = datetime.today() - timedelta(days = 1),
													key = 'train_start_date', help = 'Enter the start date to train the model from')

	train_end_date = end_date_column.date_input(label = '**Enter end date for training data** :', value = datetime.strptime('2022-12-31', '%Y-%m-%d'),
												min_value = train_start_date + timedelta(days = 1), max_value = datetime.today() - timedelta(days = 1),
												key = 'train_end_date', help = 'End date to train the model to *(can be a few months later than start date)*')

	start_date_column.text('')
	end_date_column.text('')

	test_start_date = start_date_column.date_input(label = '**Enter start date for testing data** :',
												value = datetime.strptime('2023-01-01', '%Y-%m-%d'),
												min_value = train_end_date + timedelta(days = 1), max_value = datetime.today() - timedelta(days = 1),
												key = 'test_start_date', help = 'Enter the start date to view the stock from')

	test_end_date = end_date_column.date_input(label = '**Enter end date for testing data** :', value = datetime.today() - timedelta(days = 1),
											min_value = train_start_date + timedelta(days = 1), max_value = datetime.today() - timedelta(days = 1), key = 'test_end_date',
											help = 'End date to view the stock to *(can be a day later than start date)*')

	performance_train_data, performance_test_data = whole_data[(whole_data['Date'] >= pd.to_datetime(train_start_date)) & (whole_data['Date'] <= pd.to_datetime(train_end_date))], whole_data[(whole_data['Date'] >= pd.to_datetime(test_start_date)) & (whole_data['Date'] <= pd.to_datetime(test_end_date))]
	performance_attribute = start_date_column.selectbox(label = '**Select attribute to be forecasted :**', options = performance_train_data.columns[1:], key = 'performance_attributes')

	daily_seasonality, weekly_seasonality, yearly_seasonality = end_date_column.columns(3, gap = 'small')
	performance_daily_seasonality = daily_seasonality.radio("Select daily seasonality :", [True, False, 'auto'], horizontal = False, key = 'performance_daily_seasonality', help = 'Set daily seasonality for the forecast model')
	forecast_params.text('')
	performance_weekly_seasonality = weekly_seasonality.radio("Select weekly seasonality :", [True, False, 'auto'], horizontal = False, key = 'performance_weekly_seasonality', help = 'Set monthly seasonality for the forecast model')
	forecast_params.text('')
	performance_yearly_seasonality = yearly_seasonality.radio("Select yearly seasonality :", [True, False, 'auto'], horizontal = False, key = 'performance_yearly_seasonality', help = 'Set yearly seasonality for the forecast model')

	performance_model, performance_forecast = prophet_modelling(performance_train_data, performance_attribute, performance_daily_seasonality, performance_weekly_seasonality, performance_yearly_seasonality, (test_end_date - test_start_date).days, include_history = False)

	st.text('')

	forecast_plot = go.Scatter(x = performance_forecast['ds'], y = performance_forecast['yhat'], mode = 'lines', name = 'Forecasted Data', marker = {'color' : 'red'})
	actual_plot = go.Scatter(x = performance_test_data['Date'], y = performance_test_data['High'], mode = 'markers', name = 'Actual Data', marker = {'color' : 'blue'})
	performance_plot = go.Figure(data = [actual_plot, forecast_plot], layout = go.Layout(title = 'Prophet Forecast Vs. Actual Data', xaxis = dict(title = 'Date', rangeslider = dict(visible = True)), yaxis = dict(title = f'{performance_attribute}'), showlegend = True))

	st.plotly_chart(performance_plot, use_container_width = True)

	if performance_test_data.index.name != 'Date' : performance_test_data.set_index(['Date'], inplace = True)
	if performance_forecast.index.name != 'ds' : performance_forecast.set_index(['ds'], inplace = True)

	for date in performance_forecast.index :
		if not date in list(performance_test_data.index) :
			performance_forecast.drop(performance_forecast.loc[performance_forecast.index == date].index, inplace = True)

	for date in performance_test_data.index :
		if not date in performance_forecast.index :
			performance_test_data.drop(performance_test_data.loc[performance_test_data.index == date].index, axis = 'index', inplace = True)

	mae = mean_absolute_error(performance_test_data[performance_attribute], performance_forecast['yhat'])
	mse = mean_squared_error(performance_test_data[performance_attribute], performance_forecast['yhat'])
	rmse = np.sqrt(mse)
	mape = np.mean(np.abs((performance_test_data[performance_attribute] - performance_forecast['yhat']) / performance_test_data[performance_attribute])) * 100
	smape = np.mean(2 * np.abs(performance_forecast['yhat'] - performance_test_data[performance_attribute]) / (np.abs(performance_test_data[performance_attribute]) + np.abs(performance_forecast['yhat']))) * 100

	performance_metric.text('')
	performance_metric.text('')

	mae_, mse_, rmse_, mape_, smape_ = st.columns(5, gap = 'small')
	mae_.metric(label = '**Mean Absolute Error**', value = round(mae, 2))
	mse_.metric(label = '**Mean Squared Error**', value = round(mse, 2))
	rmse_.metric(label = '**Root Mean Squared Error**', value = round(rmse, 2))
	mape_.metric(label = '**Mean Absolute Percentage Error**', value = round(mape, 2))
	smape_.metric(label = '**Symmetric Mean Absolute Percentage Error**', value = round(smape, 2))


st.text('')
st.text('')
st.text('')


st.subheader('Advanced analytics :')
with st.expander(label = 'Expand to view') :
	stock_data, forecast_data = st.columns(2, gap = 'large')
	stock_data.markdown("<h2 style='text-align: center;'> Stock Data </h2>", unsafe_allow_html=True)
	forecast_data.markdown("<h2 style='text-align: center;'> Forecast Data </h2>", unsafe_allow_html=True)
	with st.container() :
		def data_description() :
			stock_data.text('')
			forecast_data.text('')
			stock_data.text('')
			forecast_data.text('')

			stock_data.subheader('Description :')
			stock_data.dataframe(data = data.describe())
			stock_data.download_button(label = 'Download Data Description', data = data.describe().to_csv().encode('utf-8'), file_name = (f'Description - {selected_stock} - {START} to {END}.csv'), key = 'stock_description')

			forecast_data.subheader('Description :')
			forecast_data.dataframe(data = forecast.describe())
			forecast_data.download_button(label = 'Download Data Description', data = data.describe().to_csv().encode('utf-8'), file_name = (f'Description - {selected_stock} - {START} to {END}.csv'), key = 'forecast_description')

			stock_data.text('')
			forecast_data.text('')
			stock_data.text('')
			forecast_data.text('')


		stock_data.subheader('Stationarity Tests')
		forecast_data.subheader('Stationarity Tests')

		stationarity_data_attribute_input, lag_input, regression_input = stock_data.columns(3, gap = 'small')
		stationarity_data_attribute = stationarity_data_attribute_input.selectbox(label = 'Select data attribute :', options = data.columns[1:], key = 'stock_data_stationarity')
		lag = lag_input.number_input(label = 'Enter number of lags', min_value = 1, value = 1, step = 1, format = '%d', key = 'stock_data_lag')
		regression = regression_input.selectbox(label = 'Enter regression type :', options = ('c', 'ct'), key = 'stock_data_regression')

		kpssresult = kpss(data[stationarity_data_attribute], nlags = lag, regression = regression)

		if (kpssresult[1] > 0.05) : stock_data.markdown(f"""The data is **:green[STATIONARY]**

			1) Test Statistic : {kpssresult[0]}
		2) p-value : {kpssresult[1]}
		3) Critical Value : {kpssresult[3]}""")

		elif (kpssresult[1] < 0.05) : stock_data.markdown(f"""The data is **:red[NON-STATIONARY]**

				1) Test Statistic : {kpssresult[0]}
		2) p-value : {kpssresult[1]}
		3) Critical Value : {kpssresult[3]}""")

		stationarity_forecast_attribute_input, forecast_lag_input, forecast_regression_input = forecast_data.columns(3, gap = 'small')
		stationarity_forecast_attribute = stationarity_forecast_attribute_input.selectbox(label = 'Select data attribute :', options = forecast.columns[1:], key = 'forecast_stationarity')
		lag = forecast_lag_input.number_input(label = 'Enter number of lags', min_value = 1, value = 1, step = 1, format = '%d', key = 'forecast_lag')
		regression = forecast_regression_input.selectbox(label = 'Enter regression type :', options = ('c', 'ct'), key = 'forecast_regression')

		kpssresult = kpss(forecast[stationarity_forecast_attribute], nlags = lag, regression = regression)

		if (kpssresult[1] > 0.05) : forecast_data.markdown(f"""The data is **:green[STATIONARY]**

			1) Test Statistic : {kpssresult[0]}
		2) p-value : {kpssresult[1]}
		3) Critical Value : {kpssresult[3]}""")

		elif (kpssresult[1] < 0.05) : forecast_data.markdown(f"""The data is **:red[NON-STATIONARY]**

			1) Test Statistic : {kpssresult[0]}
		2) p-value : {kpssresult[1]}
		3) Critical Value : {kpssresult[3]}""")


		stock_data.text('')
		forecast_data.text('')
		stock_data.text('')
		forecast_data.text('')


	stock_data.subheader("Auto-correlation :")
	forecast_data.subheader("Auto-correlation :")
	with st.container() :
		def auto_correlation(data, pacf_method, lags) :
			mpl.rc("figure", figsize=(15, 4))
			acf_values = acf(data, nlags = lags)
			pacf_values = pacf(data, nlags = lags, method = pacf_method)

			acf_pacf_fig, (axis_1, axis_2) = plt.subplots(2, 1, figsize = (10, 8))
			acf_pacf_fig.tight_layout(h_pad = 5)

			plot_acf(data, lags = lags, ax = axis_1)
			plot_pacf(data, lags = lags, method = pacf_method, ax = axis_2)

			return acf_values, pacf_values, acf_pacf_fig


		data_attribute_input, lags_input, pacf_method_input = stock_data.columns(3, gap = 'small')
		acf_table, pacf_table = stock_data.columns(2, gap = 'large')

		data_attribute = data_attribute_input.selectbox(label = 'Select data attribute :', options = data.columns[1:], key = 'stock_attribute_autocorrelaton')
		lags = lags_input.number_input(label = 'Enter the number of lags :', min_value = 1, value = 1, key = 'stock_lags_autocorrelation')
		pacf_method = pacf_method_input.selectbox(label = 'Partial auto-correlation method :', options = ('ywadjusted', 'ywmle', 'ols', 'ols-inefficient', 'old-adjusted', 'ldadjusted', 'ldbiased'), key = 'stock_pacf_autocorrelation')

		acf_vals, pacf_vals, autocorr_plot = auto_correlation(data[data_attribute], pacf_method, lags)

		acf_table.text('')
		pacf_table.text('')

		acf_table.write("Auto-correlation values :")
		acf_values = pd.DataFrame({'ACF' : acf_vals[:-1], 'Lag' : list(range(1, lags+1))}).set_index('Lag')
		acf_table.dataframe(data = acf_values, use_container_width = True)
		acf_table.download_button(label = 'Download ACF Data', data = acf_values.to_csv().encode('utf-8'), file_name = (f'ACF Values - {selected_stock} - {START} to {END}.csv'), key = 'stock_acf_download')


		pacf_table.write("Partial auto-correlation values :")
		pacf_values = pd.DataFrame({'PACF' : pacf_vals[:-1], 'Lag' : list(range(1, lags+1))}).set_index('Lag')
		pacf_table.dataframe(data = pacf_values, use_container_width = True)
		pacf_table.download_button(label = 'Download PACF Data', data = pacf_values.to_csv().encode('utf-8'), file_name = (f'PACF Values - {selected_stock} - {START} to {END}.csv'), key = 'stock_pacf_download')

		stock_data.text('')
		stock_data.text('')

		stock_data.pyplot(autocorr_plot)

		data_attribute_input, lags_input, pacf_method_input = forecast_data.columns(3, gap = 'small')
		acf_table, pacf_table = forecast_data.columns(2, gap = 'large')

		data_attribute = data_attribute_input.selectbox(label = 'Select data attribute :', options = forecast.columns[1:], key = 'forecast_attribute_autocorrelaton')
		lags = lags_input.number_input(label = 'Enter the number of lags :', min_value = 1, value = 1, key = 'forecast_lags_autocorrelation')
		pacf_method = pacf_method_input.selectbox(label = 'Partial auto-correlation method :', options = ('ywadjusted', 'ywmle', 'ols', 'ols-inefficient', 'old-adjusted', 'ldadjusted', 'ldbiased'), key = 'forecast_pacf_autocorrelation')

		acf_vals, pacf_vals, autocorr_plot = auto_correlation(forecast[data_attribute], pacf_method, lags)

		acf_table.text('')
		pacf_table.text('')

		acf_table.write("Auto-correlation values :")
		acf_values = pd.DataFrame({'ACF' : acf_vals[:-1], 'Lag' : list(range(1, lags + 1))}).set_index('Lag')
		acf_table.dataframe(data = acf_values, use_container_width = True)
		acf_table.download_button(label = 'Download ACF Data', data = acf_values.to_csv().encode('utf-8'), file_name = (f'Forecast ACF Values - {selected_stock} - {START} to {END}.csv'), key = 'forecast_acf_download')

		pacf_table.write("Partial auto-correlation values :")
		pacf_values = pd.DataFrame({'PACF' : pacf_vals[:-1], 'Lag' : list(range(1, lags + 1))}).set_index('Lag')
		pacf_table.dataframe(data = pacf_values, use_container_width = True)
		pacf_table.download_button(label = 'Download PACF Data', data = pacf_values.to_csv().encode('utf-8'), file_name = (f'Forecast PACF Values - {selected_stock} - {START} to {END}.csv'), key = 'forecast_pacf_download')

		forecast_data.text('')
		forecast_data.text('')

		forecast_data.pyplot(autocorr_plot)


	stock_data.text('')
	forecast_data.text('')
	stock_data.text('')
	forecast_data.text('')


	stock_data.subheader('Decomposition plots :')
	forecast_data.subheader('Decomposition plots :')
	with st.container() :
		@st.cache_data(experimental_allow_widgets = True)
		def decomposition(data, attribute, model, period, extrapolate_trend) :

			result = seasonal_decompose(x = data[attribute], model = model, period = period, extrapolate_trend = extrapolate_trend)

			decomposition_figure = subplots.make_subplots(rows = 4, cols = 1, subplot_titles = ['Observed', 'Trend', 'Seasonal', 'Residuals'])
			decomposition_figure.add_trace(go.Scatter(x = data['Date'], y = result.observed, mode = 'lines', name = 'Observed'), row = 1, col = 1)
			decomposition_figure.add_trace(go.Scatter(x = data['Date'], y = result.trend, mode = 'lines', name = 'Trend'), row = 2, col = 1)
			decomposition_figure.add_trace(go.Scatter(x = data['Date'], y = result.seasonal, mode = 'lines', name = 'Seasonal'), row = 3, col = 1)
			decomposition_figure.add_trace(go.Scatter(x = data['Date'], y = result.resid, mode = 'lines', name = 'Residual'), row = 4, col = 1)
			decomposition_figure.update_layout(height = 900, title = f'<b>Decomposition Plot</b>', margin={'t':100}, title_x = 0.5, showlegend = True)

			return result.observed, result.trend, result.seasonal, result.resid, decomposition_figure

		attribute_input, model_input, period_input, extrapolate_trend_input = stock_data.columns(4, gap = 'small')

		attribute = attribute_input.selectbox(label = 'Select data attribute :', options = data.columns[1:], key = 'stock_attribute_decomposition')
		model = model_input.selectbox(label = 'Select model type :', options = ('additive', 'multiplicative'), key = 'stock_model_decomp')
		period = period_input.number_input(label = 'Enter periods :', min_value = 1, value = 1, key = 'stock_period_decomposition')
		extrapolate_trend = extrapolate_trend_input.selectbox(label = 'Extrapolate trend :', options = (True, False), key = 'stock_trend_extrapolation_decomposition')

		observed, trend, seasonal, resid, decomposition_plot = decomposition(data, attribute, model, period, extrapolate_trend)

		attribute_input.text('')
		model_input.text('')
		period_input.text('')
		extrapolate_trend_input.text('')

		attribute_input.dataframe(data = pd.DataFrame({'observed' : observed}), use_container_width = True)
		model_input.dataframe(data = trend, use_container_width = True)
		period_input.dataframe(data = seasonal, use_container_width = True)
		extrapolate_trend_input.dataframe(data = resid, use_container_width = True)

		stock_data.plotly_chart(decomposition_plot)


		attribute_input, model_input, period_input, extrapolate_trend_input = forecast_data.columns(4, gap = 'small')

		attribute = attribute_input.selectbox(label = 'Select data attribute :', options = forecast.columns[1:], key = 'forecast_attribute_decomposition')
		model = model_input.selectbox(label = 'Select model type :', options = ('additive', 'multiplicative'), key = 'forecast_model_decomp')
		period = period_input.number_input(label = 'Enter periods :', min_value = 1, value = 1, key = 'forecast_period_decomposition')
		extrapolate_trend = extrapolate_trend_input.selectbox(label = 'Extrapolate trend :', options = (True, False), key = 'forecast_trend_extrapolation_decomposition')

		observed, trend, seasonal, resid, decomposition_plot = decomposition(forecast.rename(columns = {'ds' : 'Date'}), attribute, model, period, extrapolate_trend)

		attribute_input.text('')
		model_input.text('')
		period_input.text('')
		extrapolate_trend_input.text('')

		attribute_input.dataframe(data = pd.DataFrame({'observed' : observed}), use_container_width = True)
		model_input.dataframe(data = trend, use_container_width = True)
		period_input.dataframe(data = seasonal, use_container_width = True)
		extrapolate_trend_input.dataframe(data = resid, use_container_width = True)

		forecast_data.plotly_chart(decomposition_plot)