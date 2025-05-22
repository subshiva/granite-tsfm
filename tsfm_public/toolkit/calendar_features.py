#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from datetime import date
import holidays


def generate_calendar_features(start_time, horizon, freq_str, country_code, 
                               start_time_idx=None):
    #inputs
    
    #start_time is a calendar start-date such as '2024-01-01'
    # horizon is the integer number of time periods generated based on the input frequency given by freq_str
    # freq_str indicates frequency. values include H/D/W/M/Y etc.
    # country_code = 2-letter code of country for which to extract holidays, e.g. 'US'
    # start_time_idx is used to create a global time index feature to capture a global trend
    
    #outputs
    #df with the features generated over the time range defined by the input params

    
    #first add seasonality for all frequencies from hourly to yearly, then add holidays for country_name_list[0]

    if start_time_idx is None:
        start_time_idx = 1
        
    # Create a time index
    time_index = pd.date_range(start=start_time, periods=horizon, freq=freq_str)
    df = pd.DataFrame(index=time_index)
    df.index.name = "timestamp"

    # start adding temporal features    
    df["running_index"] = range(start_time_idx, start_time_idx + horizon)

    timestamp_values = df.index.get_level_values("timestamp")
    
    df["year"] = timestamp_values.year
    calendar_features = {
        "hour_of_day": (timestamp_values.hour, 24),
        "day_of_week": (timestamp_values.dayofweek, 7),
        "day_of_month": (timestamp_values.day, 30.5),
        "day_of_year": (timestamp_values.dayofyear, 365),
        #"week_of_year": (timestamp_values.isocalendar().week, 52),
        "week_of_year": (timestamp_values.isocalendar().week.to_numpy(), 52),
        "month_of_year": (timestamp_values.month, 12),
    }

    for feature_name, (values, seasonality) in calendar_features.items():
        values = values.astype(np.int32)
        df[f"{feature_name}_sin"] = np.sin(2 * np.pi * values / (seasonality - 1))
        df[f"{feature_name}_cos"] = np.cos(2 * np.pi * values / (seasonality - 1))


    #next, add holidays for the input country_code for hourly or daily data
    #for weekly or lower frequency data, holidays are not currently detected
    
    years = df["year"].unique().tolist()
    df['country'] = country_code
    df['holiday_name'] = 'noholiday'  # default

    #print(country_code)
    mask = df['country'] == country_code
    cal = holidays.CountryHoliday(country_code, years=years)

    # Extract just the date for holiday matching
    dates = df.index.get_level_values("timestamp")[mask].date
    holiday_names = pd.Series(dates).map(lambda d: cal.get(d, 'noholiday')).values

    # Assign the results back efficiently
    df.loc[mask, 'holiday_name'] = holiday_names

    # Test to confirm Independence Day is detected
    #july_4th_check = df.loc[df.index.get_level_values("timestamp").date == date(2024, 7, 4)]
    #print("\nCheck for July 4th entry:")
    #print(july_4th_check[['holiday_name']])

    return df.reset_index(drop=True)


#test temporal feature generation on some example data. currently it is done one country at a time.
# example country
#country = 'US' # ['United States'] #, 'DE', 'FR'
#test_df = generate_calendar_features(start_time='2024-07-01', horizon=28, freq_str='W', country_code=country, start_time_idx=None)
#test_df = generate_calendar_features(start_time='2024-07-01', horizon=120, freq_str='H', country_code=country, start_time_idx=None)





