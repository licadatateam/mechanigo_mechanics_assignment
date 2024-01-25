# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:16:14 2023

@author: Arvin Jay
"""

from gmaps_maps import Geocoder
import pandas as pd



def url(sheet_id,tab):
    """
    Generate a Google Sheets URL for a given sheet_id and tab.

    Args:
        sheet_id (str): The Google Sheets document's ID.
        tab (str): The specific sheet's ID (gid).

    Returns:
        str: The URL to download the sheet as a CSV.
    """
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?gid={tab}&format=csv"

def load_sheet(sheet_info):
    """
    Load data from a Google Sheets document.

    Args:
        sheet_info (dict): A dictionary containing 'sheet_id' and 'tab'.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    return pd.read_csv(url(sheet_info['sheet_id'],sheet_info['tab']))

def geocode_df(df):
    """
    Geocode addresses in a DataFrame using the Geocoder library.

    Args:
        df (pd.DataFrame): A DataFrame containing addresses to geocode.

    Returns:
        pd.DataFrame: The input DataFrame with additional geocoding columns.
        pd.DataFrame: A DataFrame containing geocoding review information.
    """
    gmaps = Geocoder()   
    gmaps.initialize_driver()
    df, df_review = gmaps.analyze_set(df,'address','address')
    gmaps.close_driver()
    return df, df_review

def gather_data(barangays, orders):
    """
    Load data for barangays and orders from Google Sheets.

    Args:
        barangays (dict): A dictionary containing 'sheet_id' and 'tab' for barangays data.
        orders (dict or None): A dictionary containing 'sheet_id' and 'tab' for orders data, or None.

    Returns:
        pd.DataFrame: A DataFrame containing barangay data.
        pd.DataFrame or None: A DataFrame containing orders data, or None if orders is not provided.
    """
    
    # barangays_sheed_id = "1C4XVFaYPCMPtRFE5X_sHO_a89QCmU2bVzmnMTWwrkN8"
    # barangays_tab = '1054857011'
    # barangays = {'sheet_id': barangays_sheed_id,
    #              'tab': barangays_tab}
    # orders_sheet_id = "1C4XVFaYPCMPtRFE5X_sHO_a89QCmU2bVzmnMTWwrkN8"
    # orders_tab = '1417590793'
    # orders = {'sheet_id': orders_sheet_id,
    #           'tab': orders_tab}
    
    df_barangays = load_sheet(barangays)
    try:
        df_orders = load_sheet(orders)
    except:
        df_orders = orders
    return df_barangays,df_orders

def get_barangay_coords(df_barangays,df_orders):
    """
    Merge barangays and orders DataFrames based on 'barangay' and 'municipality' columns.

    Args:
        df_barangays (pd.DataFrame): A DataFrame containing barangays data.
        df_orders (pd.DataFrame): A DataFrame containing orders data.

    Returns:
        pd.DataFrame: A DataFrame with coordinates and merged data.
    """
    barangay_columns = ['barangay','municipality','lat','long']
    df = df_orders.merge(df_barangays[barangay_columns], on=['barangay','municipality'],how='left')
    return df

def fill_table(df):
    """
    Fill missing 'lat' and 'long' values in a DataFrame by geocoding missing addresses.

    Args:
        df (pd.DataFrame): A DataFrame with 'lat' and 'long' columns.

    Returns:
        pd.DataFrame: The updated DataFrame with filled coordinates.
        pd.DataFrame: A DataFrame containing geocoding review information.
        str: An error message if an error occurs during geocoding.
    """
    error_message = ''
    df_review = pd.DataFrame()
    try:
        df_search,df_review = geocode_df(df.loc[df['lat'].isnull()])
        print("### problem solved ###")
        df.update(df_search,overwrite=False)
    except Exception as e:
        print("### problem NOT solved ###")
        error_message = f"Cannot complete table: {str(e)}"
    return df, df_review, error_message
        
def geocode_by_barangay(barangays,orders=None):
    """
    Perform geocoding for barangays and orders data.

    Args:
        barangays (dict): A dictionary containing 'sheet_id' and 'tab' for barangays data.
        orders (dict or None): A dictionary containing 'sheet_id' and 'tab' for orders data, or None.

    Returns:
        pd.DataFrame: A DataFrame with coordinates and merged data.
        pd.DataFrame: A DataFrame containing geocoding review information.
        str: An error message if an error occurs during geocoding.
    """
    df,df_review,error_message = pd.DataFrame(), pd.DataFrame(), ''
    df_barangays,df_orders = gather_data(barangays, orders)
    df = get_barangay_coords(df_barangays,df_orders)
    if len(df['lat'].isnull()):
        df,df_review,error_message = fill_table(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df,df_review,error_message

# """
# Usage:
# df,df_review,error_message = geocode_by_barangay(barangays,orders)
# """
