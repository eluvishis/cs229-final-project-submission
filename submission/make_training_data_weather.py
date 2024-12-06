import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np


from preprocessing import *

temp_precip = pd.read_csv('./data/weather_precipitation_data_since_2016.csv')
dates = []
for date in temp_precip['DATE']:
    dates.append(datetime.datetime.strptime(str(date).split()[0], '%Y%m%d'))
temp_precip['DATE'] = dates
years = []
for date in dates:
    years.append(date.year)
temp_precip['YEAR'] = years
temp_precip['VALUE_precip'] = temp_precip['VALUE_precip'].apply(pd.to_numeric, errors='coerce')
total_precip = temp_precip.groupby('YEAR')['VALUE_precip'].sum()
d = total_precip.to_dict()
prior_dict = {}
for key, val in d.items():
    prior_dict[key+1] = val
prior_dict['2016'] = None

temp_precip['ANNUAL_precip'] = temp_precip['YEAR'].map(total_precip)
temp_precip['PRIOR_precip'] = temp_precip['YEAR'].map(prior_dict)
metadata = pd.read_csv('./data/station_metadata.csv')
firedata = pd.read_csv('./data/fire_data_11042024.csv')

calmac_locs = get_calmac_data_locations('./data/weather/all_weather_data_processed')
firedata = fire_preprocess(firedata)
station = []
calmacfile = []
for row in firedata.iterrows():
    distances = []
    date = str(row[1]['incident_date_created'])
    stas = temp_precip[temp_precip['DATE'] == date.split()[0]]['STATION_ID'].to_list()
    for r in metadata.iterrows():
        lat = row[1]['incident_latitude']
        lon = row[1]['incident_longitude']
        lt = r[1]['Latitude']
        ln = r[1]['Longitude']
        distances.append(np.sqrt((lat-lt)**2 + (lon - ln)**2))
    checked = False
    while not checked:
        sta = metadata['STATION_ID'][distances.index(min(distances))]
        if sta not in stas:
            distances.pop(distances.index(min(distances)))
        else:
            checked = True
    station.append(sta)
    d = []
    for v in calmac_locs.values():
        lat = row[1]['incident_latitude']
        lon = row[1]['incident_longitude']
        lt = v['lat']
        ln = v['lon']
        d.append(np.sqrt((lat-lt)**2 + (lon - ln)**2))
    calmacfile.append(list(calmac_locs.keys())[d.index(min(d))])

firedata['calmac_file'] = calmacfile
firedata['station'] = station
firedata = firedata.sort_values('incident_date_created')
files = os.listdir('./data/weather/all_weather_data_processed')
training_df = pd.DataFrame(columns=['ID', 'DATE', 'Lat', 'Lon', 'VALUE_temp', 'VALUE_precip', 'PRIOR_precip', 'DPT (F)',  'DBT (F)', 'Wind Speed (m/s)', 'Wind Dir', 'Fire'])
print(len(firedata.index))

multiples = {}
for name, group in firedata.groupby('incident_county'):
    if name not in multiples.keys():
        multiples[name] = []
        diffs = group['incident_date_created'].diff()
        for i, diff in enumerate(diffs):
            if isinstance(diff, datetime.timedelta) & (diff >= datetime.timedelta(31*6)): # conservative estimate
                multiples[name].append(group.index[i])

print(multiples)

j = 0
counter = 0
for incident in firedata.iterrows():
    print("Incident ", incident[0])
    file = incident[1]['calmac_file']
    station = incident[1]['station']
    yr = str(incident[1]['incident_year'])[-2:]
    date = str(incident[1]['incident_date_created']).split()[0]
    day = date[-2:].lstrip("0")
    month = date[-5:-3].lstrip("0")
    duration = incident[1]['incident_duration']
    county = incident[1]['incident_county']
    latitude = incident[1]['incident_latitude']
    longitude = incident[1]['incident_longitude']

    if datetime.datetime.strptime(date, "%Y-%m-%d").date() > datetime.date(2016, 6, 1):
        # get calmac data
        filenames = []
        for f in files:
            if (file in f):
                filenames.append(f)
        for i, file in enumerate(filenames):
            if i < 1:
                cal_df = pd.read_csv('./data/weather/all_weather_data_processed/' + file,  header=2, sep='\\s+')
                if all(name in cal_df.columns for name in ['Year', 'Mo', 'Dy']):
                    calmac_file = calmac_preprocess(cal_df)
                else:
                    continue
            else:
                cal_df = pd.read_csv('./data/weather/all_weather_data_processed/' + file,  header=2, sep='\\s+')
                if all(name in cal_df.columns for name in ['Year', 'Mo', 'Dy']):
                    df = calmac_preprocess(cal_df)
                else:
                    continue
                calmac_file = pd.concat([calmac_file, df])
        
        # get previous 6 mos data from station
        temp_precip_previous = temp_precip[(temp_precip['STATION_ID'] == station) & 
                                           (temp_precip['DATE'].dt.date < datetime.datetime.strptime(date, "%Y-%m-%d").date()) & 
                                           (temp_precip['DATE'].dt.date >= datetime.datetime.strptime(date, "%Y-%m-%d").date() - relativedelta(months=6))]
        temp_precip_previous = temp_precip_previous[['DATE', 'VALUE_temp', 'VALUE_precip', 'PRIOR_precip']].copy().reset_index()
        temp_precip_previous['Lat'] = [latitude]*len(temp_precip_previous)
        temp_precip_previous['Lon'] = [longitude]*len(temp_precip_previous)
        temp_precip_previous['ID'] = [incident[0]]*len(temp_precip_previous)
        print(temp_precip_previous.head())
        temp_precip_previous.drop(columns=['index'], inplace=True)
        
        # get previous 6 mos days of calmac data
        grp = calmac_file.groupby('DATE', as_index=False).agg({'DPT (F)': 'mean', 'DBT (F)': 'mean', 'Wind Speed (m/s)': 'mean', 'Wind Dir': 'mean'})
        previous = grp[
        (grp['DATE'].dt.date >= datetime.datetime.strptime(date, "%Y-%m-%d").date() - relativedelta(months=6)) &
        (grp['DATE'].dt.date < datetime.datetime.strptime(date, "%Y-%m-%d").date())
        ].copy().reset_index()
        previous['Fire'] = [0]*len(previous.index)
        previous.drop(columns=['index', 'DATE'], inplace=True)
        previous = pd.concat([temp_precip_previous, previous], axis=1)

        # Drop days where mult fires in 1 county in 6 mos
        if county in multiples.keys():
            ranges = []
            for idx in multiples[county]:
                start = str(firedata.iloc[idx]['incident_date_created']).split()[0]
                end = datetime.datetime.strptime(str(firedata.iloc[idx]['incident_date_created']).split()[0], "%Y-%m-%d").date() + datetime.timedelta(days=int(firedata.iloc[idx]['incident_duration']))
                ranges.append(pd.date_range(start, end))
            previous.set_index('DATE', inplace=True)
            for range in ranges:
                for d in range:
                    if d in previous.index:
                        previous.drop(d, inplace=True)
            previous.reset_index(inplace=True)
        training_df = pd.concat([training_df, previous])
        print(training_df.head())

        # get days of incident sta
        temp_precip_days = temp_precip[(temp_precip['STATION_ID'] == station) & 
                                           (temp_precip['DATE'].dt.date >= datetime.datetime.strptime(date, "%Y-%m-%d").date()) & 
                                           (temp_precip['DATE'].dt.date <= datetime.datetime.strptime(date, "%Y-%m-%d").date() + pd.Timedelta(days=duration))]
        temp_precip_days = temp_precip_days[['DATE', 'VALUE_temp', 'VALUE_precip', 'PRIOR_precip']].copy().reset_index()
        temp_precip_days['Lat'] = [latitude]*len(temp_precip_days)
        temp_precip_days['Lon'] = [longitude]*len(temp_precip_days)
        temp_precip_days['ID'] = [incident[0]]*len(temp_precip_days)
        temp_precip_days.drop(columns=['index'], inplace=True)

        # get days of incident calmac
        inc_calmac = grp[
        (grp['DATE'].dt.date <= datetime.datetime.strptime(date, "%Y-%m-%d").date() + pd.Timedelta(days=duration)) &
        (grp['DATE'].dt.date >= datetime.datetime.strptime(date, "%Y-%m-%d").date())
        ].copy().reset_index()
        inc_calmac.drop(columns=['index', 'DATE'], inplace=True)
        inc_calmac['Fire'] = [1]*len(inc_calmac.index)

        inc = pd.concat([temp_precip_days, inc_calmac], axis=1)
        training_df = pd.concat([training_df, inc])
training_df.reset_index(inplace=True)

training_df.to_csv('./all_weather_data.csv')
training_data = training_df.iloc[:int(len(training_df)*.7)]
validation_data = training_df.iloc[int(len(training_df)*.7):]
training_data.to_csv('./weather_training_data.csv')
validation_data.to_csv('./weather_val_data.csv')