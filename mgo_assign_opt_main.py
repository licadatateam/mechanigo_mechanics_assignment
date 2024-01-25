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
import folium

# custom modules
from barangay_processing import geocode_by_barangay

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(message)s',
                    datefmt = '%d-%b-%y %H:%M:%S',
                    level=logging.WARNING)

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

def monkeypatch_get_storage_manager():
    if st.runtime.exists():
        return st.runtime.get_instance().cache_storage_manager
    else:
        # When running in "raw mode", we can't access the CacheStorageManager,
        # so we're falling back to InMemoryCache.
        # _LOGGER.warning("No runtime found, using MemoryCacheStorageManager")
        return st.runtime.caching.storage.dummy_cache_storage.MemoryCacheStorageManager()

st.runtime.caching._data_caches.get_storage_manager = monkeypatch_get_storage_manager

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
                selected_date: date or str) -> pd.DataFrame:
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
        try:
            # assume type str, if date, trigger exception
            appointments = appointments.loc[appointments['date']==selected_date.strftime('%Y-%m-%d')].drop(columns = ['date'])
        except:
            appointments = appointments.loc[appointments['date']==selected_date].drop(columns = ['date'])
        # drop duplicates
        appointments = appointments.drop_duplicates(subset = ['appointment_id']).reset_index(drop = True)
        appointments['address'] = appointments['address'].fillna('')
        # if hub is explicitly stated, else None meaning need to be solved
        appointments['hub_solver'] = appointments['pin_address'].apply(extract_hub)
        logging.debug('Cleaning appointments data.')
        
    except Exception as e:
        raise e
        
    return appointments

@st.cache_data
def clean_address(row: pd.Series, 
                  address_col: str = 'pin_address', 
                  barangay_col: str = 'barangay',
                  municipality_col: str = 'municipality', 
                  province_col: str = 'province') -> tuple[str, str, str, str, str]:
    """
    Clean and format the address information in a given row of a DataFrame.

    Args:
    -----
        - row : pd.Series
            The row containing address-related columns.
        - address_col : str (Optional)
            Column name for the address. Defaults to 'pin_address'.
        - barangay_col : str (Optional)
            Column name for the barangay. Defaults to 'barangay'.
        - municipality_col : str (Optional)
            Column name for the municipality. Defaults to 'municipality'.
        - province_col : str (Optional)
            Column name for the province. Defaults to 'province'.

    Returns:
        - Tuple[str, str, str, str, str]: A tuple containing the cleaned street address, street name, cleaned full address,
        full address with street name, and full address with street name and barangay.

    Examples:
        >>> import pandas as pd
        >>> data = {'pin_address': '123 Main St, Brgy. Example, Municipality A, Province X', 'barangay': 'Example', 'municipality': 'Municipality A', 'province': 'Province X'}
        >>> row = pd.Series(data)
        >>> clean_address(row)
        ('123 Main St', 'Main St', '123 Main St in Example, Municipality A, Province X in Philippines', 'Main St in Brgy. Example, Municipality A, Province X in Philippines', 'Main St in Example, Municipality A, Province X in Philippines')
    """
    address = row[address_col] if row[address_col] else ''
    barangay = row[barangay_col] if row[barangay_col] else ''
    municipality = row[municipality_col] if row[municipality_col] else ''
    province = row[province_col] if row[province_col] else ''

    # Additional logic to handle specific cases
    if province.lower() == 'metro manila' and municipality.lower() == 'quezon':
        municipality = 'quezon city'

    remove_pattern = re.compile(r'b\.?f\.? homes')

    if len(str(address)) != 0:
        address = re.sub(remove_pattern, '', str(address))
    
    try:
        if barangay in address:
            street_address = address.split(barangay)[0].strip()
        else:
            if municipality in address:
                street_address = address.split(municipality)[0].strip()
            else:
                if province in address:
                    street_address = address.split(municipality)[0].strip()
                else:
                    street_address = address
                    
        cleaned_address_ = street_address
        if street_address != address:
            cleaned_address = ' '.join([street_address, 'in', barangay + ',', municipality + ',', province, 'in Philippines'])
        else:
            cleaned_address = street_address + ' in Philippines'
    except:
        street_address = address
        cleaned_address_ = address
        cleaned_address = address

    street = ''
    # Street pattern
    st_pattern = r'(\b\w+\s+(st\.?|street)\b)'
    match = re.search(st_pattern, street_address, re.IGNORECASE)
    if match:
        street = match.group(1)
    cleaned_address_ = re.sub(street, '', cleaned_address_).strip()

    try:
        street_address_ = ' '.join([street, 'brgy.', barangay + ',', municipality + ',', province, 'in Philippines'])
    except:
        street_address_ = street_address

    return street_address, street, cleaned_address, cleaned_address_, street_address_

@st.cache_data
def clean_address_appts(appointments : pd.DataFrame) -> pd.DataFrame:
    '''
    Wrapper function for clean_address
    
    Args:
    -----
        - appointments : pd.DataFrame
            appointments data
    
    Returns:
    --------
        - _appointments : pd.DataFrame
            copy of appointments data with cleaned address (if no errors)
    '''
    _appointments = appointments.copy()
    
    try:
        _appointments[['street_address', 'street', 'address_query', 'partial_address',
                      'street_address_query']] = _appointments.apply(clean_address, 
                                                                    axis=1, 
                                                                    result_type='expand')
        logging.debug('Cleaning appointments addresses.')
    except Exception as e:
        logging.exception(e)
    
    return _appointments

@st.cache_data
def geocode(barangays: dict, 
            appointments: pd.DataFrame) -> dict:
    """
    Perform geocoding based on barangays and appointments data.

    Args:
    -----
        - barangays : dict
            Dictionary containing information about barangays, typically loaded from configuration.
        - appointments : pd.DataFrame
            DataFrame containing appointment data.

    Returns:
    --------
        - geo_dict : dict {pd.DataFrame, pd.DataFrame, Optional[str]}
            A dict containing geocoded appointments DataFrame, a review DataFrame,
        and an error message if any.
    """
    try:
        _ = geocode_by_barangay(barangays, appointments)
        geo_dict = {'appointments' : _[0],
                    'review' : _[1],
                    'error' : _[2]}
        
        logging.debug('Geocoding addresses.')
        return geo_dict
    
    except Exception as e:
        if check_streamlit():
            st.error('FAILED: Geocoding addresses.')
        else:
            raise e

@st.cache_data
def generate_eda(appointments: pd.DataFrame) -> dict:
    """
    Generate exploratory data analysis (EDA) metrics from the given appointments DataFrame.

    Args:
    -----
        - appointments : pd.DataFrame
            The DataFrame containing appointment data.

    Returns:
    --------
        dict : [int, int, pd.DataFrame] 
            A dictcontaining total appointments, total services, and a DataFrame with location information.
    """
    # Calculate total number of appointments and services
    total_appointments = appointments['appointment_count'].sum()
    total_services = appointments['services_count'].sum()
    
    # Generate DataFrame with location information
    df_location = appointments.groupby('time')['province'].value_counts().unstack()
    df_location['total'] = df_location.sum(axis=1).astype(int)
    
    return {'total_appointments' : total_appointments,
            'total_services' : total_services,
            'location' : df_location}

@st.cache_data
def display_tab_eda(appointments : pd.DataFrame):
    
    st.header("Geocoded Addresses")
    
    eda_dict =  generate_eda(appointments)
    
    a, b = st.columns([1,3])
    
    with a:
        st.metric('Total Appointments',
                  value = eda_dict['total_appointments'],
                  delta = 'Services: '+ str(eda_dict['total_services']),
                  delta_color ='off')
    
    with b:
        st.bar_chart(eda_dict['location'].drop(columns='total'))
    
    c, d = st.columns([2,2])
    
    # construct timeslots
    timeslots = appointments['time'].sort_values(ascending=True).unique()
    
    with c:
        time_filter = st.multiselect('Select timeslots:', 
                                     timeslots,
                                     default = timeslots)
        
    display_appointments = appointments.loc[appointments['time'].isin(time_filter)]
    # display appointments
    st.write(display_appointments)
    
    map_lat = display_appointments['lat'].mean()
    map_long = display_appointments['long'].mean()
    m = folium.Map(location=[map_lat, map_long], zoom_start=10)
    
    if len(display_appointments)>0:
        for index, row in display_appointments.iterrows():
            lat_row = row['lat']
            long_row = row['long']
            tooltip = 'Appointment ID: ' +str(row['appointment_id'])
            popup = '\n'.join([row['time'],row['fullname'],f"Service: {row['service_category']} Address: {row['pin_address']}"])
            
            folium.Marker(
                location=[lat_row, long_row],
                popup=folium.Popup(popup, parse_html=True),
                tooltip=tooltip,
            ).add_to(m)
            # custom_icon = folium.CustomIcon(golden_icon, icon_size=(35, 35))
            # folium.Marker(location=coordinates, icon=custom_icon,popup=popup)
           
    folium_data = st_folium(m)
    with d:
        st.write(display_appointments)
    
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
    
    # clean address
    appointments = clean_address_appts(appointments)

    # geocoding
    geo_dict = geocode(config_dict['barangays'], appointments)
    appointments = geo_dict['appointments']
    
    # generate eda
    eda_dict =  generate_eda(appointments)
    
    # construct timeslots
    timeslots = appointments['time'].sort_values(ascending=True).unique()
    
    
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
        
        # clean address
        appointments = clean_address_appts(appointments)
    
        # geocoding
        geo_dict = geocode(config_dict['barangays'], appointments)
        appointments = geo_dict['appointments']
    
        with tab_eda:
            display_tab_eda(appointments)
            
            
    return appointments

if __name__ == "__main__":
    # start program
    start = dt.now()
    logging.debug('Program start.')
    
    # check if running in streamlit
    st_running = check_streamlit()
    
    if st_running:
        logging.debug('Running script in streamlit.')
        res = st_main()
    else:
        #res = no_st_main()
        pass

    # end program
    end = dt.now()
    logging.debug('Program finished! Runtime: {0:.2f}s'.format((end-start).total_seconds()))
