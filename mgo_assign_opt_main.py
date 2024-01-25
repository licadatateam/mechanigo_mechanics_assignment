# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 07:39:25 2024

@author: carlo
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from datetime import datetime as dt
import regex as re
import logging
import json
from fuzzywuzzy import fuzz, process

import streamlit as st

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(message)s',
                    datefmt = '%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)

def check_streamlit():
    """
    Function to check whether python code is run within streamlit

    Returns
    -------
    use_streamlit : boolean
        True if code is run within streamlit, else False
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
            use_streamlit = False
        else:
            use_streamlit = True
    except ModuleNotFoundError:
        use_streamlit = False
    return use_streamlit

def load_config(config_file = 'config'):
    '''
    
    Load configuration data and secrets
    
    google sheet link:
    https://docs.google.com/spreadsheets/d/1C4XVFaYPCMPtRFE5X_sHO_a89QCmU2bVzmnMTWwrkN8/edit#gid=0
    
    Args:
    -----
    - config_file : str
            configuration filename (no extension)
    
    Returns:
    --------
    - 
    
    '''
    # load configuration and secrets
    config_file = f'{config_file}.json'
    with open(config_file) as json_file:
        keys = json.load(json_file)
    
    # gmaps api key
    maps_key = keys['mapsApiKey']
    # redash api key
    redash_key = keys['redashApiKey']
    # barangays data from google sheets 
    # TODO: convert to redash query
    barangays = {'sheet_id': keys['barangays_sheet_id'],
                 'tab': keys['barangays_tab']}
    orders = {'sheet_id': keys['orders_sheet_id'],
                 'tab': keys['orders_tab']}
    
    return {'maps' : maps_key,
            'redash' : redash_key,
            'barangays': barangays,
            'orders' : orders}
    

def initialize() -> dict:
    """
    Initialize date values.

    Returns:
    --------
        - init_dates: A dict containing the minimum date, default value, and maximum date.
        
    """
    min_date = date(2023, 7, 1)
    value = date.today() #- timedelta(days=2)
    max_date = date.today()
    return {'min_date' : min_date,
            'value' : value,
            'max_date' : max_date}

def extract_hub(entry: str):
    """
    Extract hub information from the given entry.

    Args:
    -----
        - entry : str 
            The entry to process.

    Returns:
    --------
        - res : str 
            The extracted hub information, or None if not found.
    """
    hubs = ['makati_hub', 'sucat_hub', 'starosa_hub']
    
    try:
        _ = process.extractOne(entry, hubs, 
                           scorer = fuzz.partial_ratio,
                           score_cutoff = 75)
    except Exception as e:
        raise e
    
    # if result satisfying matching cutoff is found
    if _ is not None:
        res = _[0]
    # if no match is found
    else:
        res = None
    
    return res

@st.cache_data
def gather_data(key: str, 
                selected_date: date) -> pd.DataFrame:
    """
    Gather appointment data from the specified key and selected date.

    Args:
    ------
        - key : str
            The key to access appointment data (redash_key)
        - selected_date : date 
            The selected date for filtering appointments.

    Returns:
    --------
        - appointments : pd.DataFrame 
            The gathered cleaned appointment data.
    """
    
    try:
        appointments = pd.read_csv(key)
        logging.debug('Importing appointments data from redash.')
    except:
        logging.exception('FAILED: Importing appointments data from redash.')
    
    try:
        appointments['date'] = pd.to_datetime(appointments['date'], format = '%m/%d/%y')
        appointments['time'] = pd.to_datetime(appointments['time'], format = '%H:%M:%S')
        # TODO: adjustable service duration
        appointments['time_end'] = appointments['time'].apply(lambda x: x+timedelta(hours=2, minutes=15)).dt.time
        appointments[['time','time_end']] = appointments[['time','time_end']].map(lambda x: x.strftime('%H:%M'))
        # TODO: clean province from redash
        appointments['province'] = appointments['province'].str.title()
        # filter appointments on date
        appointments = appointments.loc[appointments['date']==selected_date.strftime('%Y-%m-%d')].drop(columns = ['date'])
        # drop duplicates
        appointments = appointments.drop_duplicates(subset = ['appointment_id']).reset_index(drop = True)
        appointments['address'] = appointments['address'].fillna('')
        # if hub is explicitly stated, else None meaning need to be solved
        appointments['hub_solver'] = appointments['pin_address'].apply(extract_hub)
        logging.debug('Cleaning appointments data.')
        
    except Exception as e:
        raise e
        
    return appointments

def no_st_main():
    # configuration settings
    config_dict = load_config()
    
    # initialize date
    init_dates = initialize()
    
    # select date
    #selected_date = init_dates['value']
    selected_date = date(2024, 1, 24)
    
    # Load Redash query result
    appointments = gather_data(config_dict['redash'], selected_date)
    
    # check if there are appointments scheduled
    if len(appointments) == 0:
        logging.error(f'Found 0 scheduled appointments for {selected_date}.')
    else:
        logging.debug(f'Found {len(appointments)} scheduled appointments for {selected_date}.')
    
    return appointments
    
def st_main():
    # page config
    st.set_page_config(layout="wide")
    
    # configuration settings
    config_dict = load_config()
    
    # initialize date
    init_dates = initialize()
    
    # select date
    with st.sidebar:
        selected_date = st.date_input('Select date:',
                                      value = init_dates['value'], 
                                      min_value = init_dates['min_date'], 
                                      max_value = init_dates['max_date'])
        selected_date = date(2024, 1, 24).strftime('%Y-%m-%d')
    
    # Define tabs for Streamlit app    
    tab_eda, tab_raw, tab_geocode, tab_clustered, tab_timeline, tab_dm, tab_final = st.tabs(
        ['EDA', 'Raw Data', 'Geocoded Appointments', 'Clustered Appointments', 
         'Timelines', 'Distance Matrix', 'Assignment'])
    
    # Load Redash query result
    appointments = gather_data(config_dict['redash'], selected_date)
    
    # check if there are appointments scheduled
    if len(appointments) == 0:
        st.error(f'Found 0 scheduled appointments for {selected_date}.')
        st.stop()
    else:
        # Display Redash query result
        with tab_raw:
            st.header("Imported cleaned appointments data.")
            st.write(f'Found {len(appointments)} scheduled appointments for {selected_date}.')
            st.write(appointments)
    
    
    return appointments

if __name__ == "__main__":
    # start program
    start = dt.now()
    logging.debug('Program start.')
    
    # check if running in streamlit
    st_running = check_streamlit()
    
    if st_running:
        res = st_main()
    else:
        res = no_st_main()

    # end program
    end = dt.now()
    logging.debug('Program finished! Runtime: {0:.2f}s'.format((end-start).total_seconds()))
