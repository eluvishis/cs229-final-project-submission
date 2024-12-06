import numpy as np
import os
import pandas as pd
import shutil
import zipfile

def extract(sourcedir, extraction_dir_final):
    """Handling nested zipfiles.
    
    Parameters
    ----------
    sourcedir: where the zipfile lives
    extraction_dir_final: Where you want unzipped data to go
    
    Makes folder of extracted data
    """
    for file in os.listdir(sourcedir):
        extraction_dir = sourcedir + '/temp'
        os.makedirs(extraction_dir, exist_ok=True)

        if file.endswith('.zip'):
            with zipfile.ZipFile(sourcedir + file, 'r') as zip:
                zip.extractall(extraction_dir)
    
    for file in os.listdir(extraction_dir):
        os.makedirs(extraction_dir, exist_ok=True)
        if file.endswith('.zip'):
                with zipfile.ZipFile(extraction_dir + '/' + file, 'r') as zip:
                    for file_info in zip.infolist():
                        if 'FIN4' in file_info.filename:
                            zip.extract(file_info, extraction_dir_final)
    shutil.rmtree(extraction_dir)

def get_calmac_data_locations(path):
    """Getting lat/lons to map to precipitation data.
    
    Parameters
    ----------
    path: Filepath to CALMAC data
    
    Returns
    -------
    locations: dict, location of station as keys with a dict for lat and lon values.
    """
    files = os.listdir(path)
    locations = {}
    for file in files[:-1]:
        with open(path + '/' + file, 'r') as f:
            lines = f.readlines()[0].strip()
            data = lines.split()
            locations[data[0].replace('CA_', '')] = {'lat': float(data[2]), 'lon': float(data[3])}

    return locations

def calmac_preprocess(df):
    """Renaming columns to include units.
    
    Parameters
    ----------
    df: Pandas DataFrame of extracted CALMAC data
    
    Returns
    -------
    df: Cleaned DF
    """
    rename = {'(C)': 'DBT (C)', '(C).1': 'DPT (C)', '(mb)': 'Press (mb)', '(inHg)': 'Altim (inHg)', 'Cov': 'Sky Cov', 'Cov.1': 'Sky Cov (1)', 'QLW': 'Sky QLW', '(m/s)': 'Wind Speed (m/s)', 'Dir': 'Wind Dir', '(W/m2)': 'SatGHI (W/m2)', '(W/m2).1': 'SatDNI (W/m2)', 'Wth': 'Pressure Wth'}
    df.rename(columns=rename, inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(str).str.extract("(\\d+\\.\\d+|\\d+)", expand=False)
        df[col] = df[col].apply(pd.to_numeric, errors='coerce')
    if (df['Wind Speed (m/s)'] > 100).any():
        df.columns = df.columns[:df.columns.get_loc('Altim (inHg)')].to_list() + df.columns[df.columns.get_loc('Altim (inHg)'):].to_list()[1:] + [df.columns[df.columns.get_loc('Altim (inHg)')]]
    df['DPT (F)'] = df['DPT (C)']*9/5 + 32
    df['DBT (F)'] = df['DBT (C)']*9/5 + 32
    df['DATE'] = pd.to_datetime(
    df[['Year', 'Mo', 'Dy']].astype(str).agg('-'.join, axis=1), 
    format='%Y-%m-%d'
)
    return df

def fire_preprocess(df):
    df['incident_date_last_update'] = pd.to_datetime(df['incident_date_last_update'], utc=True)
    df['incident_date_created'] = pd.to_datetime(df['incident_date_created'])
    df['incident_date_extinguished'] = pd.to_datetime(df['incident_date_extinguished'])
    df['incident_year'] = df['incident_date_created'].dt.year
    df.dropna(subset=['incident_date_extinguished', 'incident_date_created'], inplace=True)
    duration = []
    delta = df['incident_date_extinguished'] - df['incident_date_created']
    for delta in delta:
        duration.append(int(delta.total_seconds()/86400)+1) #add 1 to make sure day is captured
    df['incident_duration'] = duration
    df.drop(['incident_dateonly_extinguished', 'incident_dateonly_created'], axis=1, inplace=True)
    corrected_df = df[(((df['incident_latitude'] > 32) | (df['incident_latitude'] < 42)) & (df['incident_longitude'] < -113) | (df['incident_longitude'] > -125)) & (df['incident_year'] > 2015)] #TODO: Fix to be better
    corrected_df.reset_index(inplace=True)
    return corrected_df