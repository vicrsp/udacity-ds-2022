import pandas as pd
from datetime import datetime

def pre_process_performances():
    raw_data = pd.read_csv('performances_wattbike.csv', parse_dates=['dateStamp'])
    raw_data = raw_data[(raw_data.dateStamp >= '2021')&(raw_data.ID == 7)]
    raw_data.drop(columns=['ID', 'Age', 'Gender', 'aveHR', 'units'], inplace=True)
    raw_data.to_csv('performances_wattbike_self.csv',index=False)

def pre_process_weight():
    raw_weight = pd.read_json('weight_fit.json')
    raw_weight['Weight'] = raw_weight['Data Points'].apply(lambda x: x['fitValue'][0]['value']['fpVal'])
    raw_weight['Timestamp'] = raw_weight['Data Points'].apply(lambda x: datetime.fromtimestamp(x['startTimeNanos']/1e9))

    raw_weight = raw_weight[raw_weight.Timestamp >= '2021']
    raw_weight.drop(columns=['Data Source','Data Points']).to_csv('weight_self.csv', index=False)