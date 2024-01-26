# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:11:27 2023

@author: Arvin Jay
"""

import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
from datetime import timedelta
from datetime import datetime as dt



# Function to generate hub data from a JSON configuration file
def generate_hub_data(file : (str, dict) = 'config.json') -> tuple[pd.DataFrame, int]:
    """
    Generates hub data from a JSON configuration file.

    Args:
    -----
        - file (str, dict): 
            Path to the JSON configuration file or dict of config (default is 'config.json').

    Returns:
    --------
        - hub_data : pd.DataFrame
            DataFrame containing hub data.
        - len(hubs) : int
            Number of hubs in the generated hub data.
    """
    try:
        if isinstance(file, str):
            # if file is local filename
            with open(file) as json_file:
                config_file = json.load(json_file)
                
        elif isinstance(file, dict):
            # if file is loaded dict from config
            config_file = file
        
        else:
            raise Exception('Incorrect config file argument.')
            
    except:
        raise Exception('Incorrect config file argument.')
    
    # list of dicts
    hubs = [config_file[key] for key in config_file.keys() if key.endswith("_hub")]
    
    hub_data = pd.DataFrame(hubs)
    hub_data[['lat_rad', 'long_rad']] = hub_data[['lat', 'long']].map(lambda x: np.radians(x))
    hub_data.columns = ['hub', 'lat', 'long', 'lat_rad', 'long_rad']
    
    return hub_data

# Function to gather information from a CSV file or DataFrame
def gather_info(source : (str, pd.DataFrame) ='sample_coords.csv') -> tuple[pd.DataFrame, 
                                                                            pd.DataFrame]:
    """
    Gathers information from a CSV file or DataFrame.

    Args:
        source (str or pd.DataFrame): Path to the CSV file or DataFrame with information.

    Returns:
        - df : pd.DataFrame
            DataFrame containing gathered information with 'appointment_id', 'lat_rad', and 'long_rad' columns.
        - df_hub_service : pd.DataFrame
            Filtered dataFrame containing appointments with nonnull hub_solver field
    """
    try:
        if isinstance(source, str):
            df = pd.read_csv(source)#, usecols=['appointment_id', 'lat_rad', 'long_rad']
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise Exception(f'Incorrect source type {type(source)}')
    except:
        raise Exception(f'Incorrect source type {type(source)}')
        
    df_hub_service = pd.DataFrame()
    if 'hub_solver' in df.columns:
        df_hub_service = df[df['hub_solver'].notnull()]
        df = df[df['hub_solver'].isnull()]
    else:
        df['hub_solver'] = None
        
    if 'lat_rad' not in df.columns:
        df[['lat_rad', 'long_rad']]= df[['lat', 'long']].map(lambda x: np.radians(x))

    if 'duration' not in df.columns:
        df['duration'] = 2.25
        
    elif len(df.loc[df['duration'].isnull()])>0:
        # number of rows with null duration is greater than 0
        df.loc[df['duration'].isnull()]['duration'] = 2.25
        
    if 'time_end' not in df.columns:
        df['time_end'] = df[['time','duration']].apply(lambda row: (dt.strptime(row['time'],"%H:%M")+timedelta(hours=row['duration'])).strftime('%H:%M'),axis=1)#['time'].apply(lambda row: (datetime.strptime(row,"%H:%M")+timedelta(hours=2)).strftime('%H:%M'))
    
    df = df.drop_duplicates(subset = ['appointment_id'])
    return df, df_hub_service
    

# Function to write a dataset by concatenating hub data and current data
def write_dataset(hub_data, current_data):
    """
    Writes a dataset by concatenating hub data and current data.

    Args:
        hub_data (pd.DataFrame): DataFrame containing hub data.
        current_data (pd.DataFrame): DataFrame containing current data.

    Returns:
        df (pd.DataFrame): Concatenated DataFrame.
    """
    df = pd.concat([hub_data, current_data], axis=0, ignore_index=True)
    return df

# Function for recursive DBSCAN clustering
def recursive_dbscan(df, hub_data):
    """
    Performs recursive DBSCAN clustering.

    Args:
    -----
        - df : pd.DataFrame 
            DataFrame with data to cluster (appointments)
        - hub_data : pd.DataFrame
            DataFrame with hub data.

    Returns:
        labels : list
            Cluster labels assigned to data points.
        status : int
            1 if DBSCAN was successful, 0 if it failed.
    """
    hubs = len(hub_data)
    # Earth's mean radius = 6371.0088
    kms_per_radian = 6371.0088
    coords = df[['lat_rad', 'long_rad']]
    
    eps = 5 / kms_per_radian
    min_samples = 1
    n_clusters = 30
    status = 0
    i = 0
    while n_clusters > hubs and i < 1000:
        curr_radius = eps + eps * i / 1000
        
        db = DBSCAN(eps=curr_radius, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords)
        if i == 0:
            labels = db.labels_
        n_clusters_ = len(set(db.labels_))
        if n_clusters_ != n_clusters:
            n_clusters = n_clusters_
            if len(set(db.labels_[0:hubs])) == hubs:
                labels = db.labels_
            else:
                break
        i += 1
        if n_clusters == hubs:
            status = 1
        if i == 1000:
            status = 0
    return labels, status

# Function to gather DBSCAN clustering results
def gather_dbscan(df, hub_data):
    """
    Gathers DBSCAN clustering results and adds them to the DataFrame.

    Args:
    -----
        - df : pd.DataFrame
            DataFrame with appointments data.
        - hub_data : pd.DataFrame
            DataFrame with hub data.

    Returns:
        - df : pd.DataFrame 
            DataFrame with DBSCAN cluster labels.
    """
    status = ''
    try:
        
        labels, status = recursive_dbscan(df, hub_data)
        df['dbscan'] = labels
        
    except:
        status = 'error'

    return df,status

# Function to calculate Haversine distances and assign clusters
def gather_hd(df, hub_data):
    """
    Calculates Haversine distances and assigns clusters to the DataFrame.

    Args:
    -----
        - df : pd.DataFrame
            DataFrame with appointments data.
        - hub_data : pd.DataFrame 
            DataFrame with hub data.

    Returns:
    -------
        - df : pd.DataFrame 
            DataFrame with 'hd' column containing cluster assignments based on Haversine distances.
    """
    status = ''
    try:
        cluster_centers = hub_data[['lat_rad', 'long_rad']]
        # TODO: option for vincenty distance
        distances = haversine_distances(df[['lat_rad', 'long_rad']], 
                                        cluster_centers)
        # clustering based on min distance from cluster/hub
        cluster_assignments = np.argmin(distances, axis=1)
        # distance-based clusters
        df['hd'] = cluster_assignments
        
    except:
        status = 'error'
        
    return df, status

# Function to perform KMeans clustering
def gather_kmeans(df, hub_data):
    """
    Performs KMeans clustering and assigns cluster labels to the DataFrame.

    Args:
    -----
        - df : pd.DataFrame
            DataFrame with data.
        - hub_data : pd.DataFrame
            DataFrame with hub data.

    Returns:
    --------
        - df : pd.DataFrame 
            DataFrame with 'kmeans' column containing KMeans cluster labels.
    """
    
    status = ''
    try: 
        cluster_centers = hub_data[['lat_rad', 'long_rad']]
        data = df[['lat_rad', 'long_rad']]
        kmeans = KMeans(n_clusters = len(hub_data), 
                        init=cluster_centers, n_init=1, )
        kmeans.fit(data)
        kmeans.predict(data)
        df['kmeans'] = kmeans.labels_
        
    except:
        status = 'error'
        
    return df, status

        
def gather_clusters(df : pd.DataFrame, 
                    hub_data : pd.DataFrame) -> pd.DataFrame:
    """
    Gathers clustering results from different methods and adds them to the DataFrame.

    Args:
    ------
        - df : pd.DataFrame
            DataFrame with data (appointments)
        - hub_data : pd.DataFrame
            DataFrame with hub data (hub_data)

    Returns:
    --------
        - df : pd.DataFrame
            DataFrame with additional columns for DBSCAN, Haversine distances, and KMeans clustering results.
    """
    # get cluster labels from recursive dbscan algo
    df, status_dbscan = gather_dbscan(df, hub_data)
    # get cluster labels based on min distance from cluster
    df, status_hd = gather_hd(df, hub_data)
    # get cluster labels from kmeans algo
    df, status_kmeans = gather_kmeans(df, hub_data)
    
    status = f"{status_dbscan}, {status_hd}, {status_kmeans}"
    return df, status

def hub_mapping(hub_data):
    """
    Creates a dictionary mapping hub indices to hub appointment IDs.

    Args:
    -----
        - hub_data : pd.DataFrame
            DataFrame containing hub data.

    Returns:
    --------
        hub_dict : dict 
            Dictionary where keys are hub indices and values are hub appointment IDs.
    """
    hub_dict = dict()
    
    for i in range(len(hub_data)):
        hub_dict[i] = hub_data.iloc[i]['hub']
    
    return hub_dict

def apply_hub_map(df, hub_data, cols=['dbscan', 'hd'], hub='hd'):#, 'kmeans'
    """
    Applies hub mapping to the DataFrame by creating new columns for each clustering method.
    *Label to hub
    
    Args:
    -----
        - df : pd.DataFrame
            DataFrame with data and clustering results.
        - hub_data : pd.DataFrame
            DataFrame with hub data.
        - cols :  list
            List of columns containing clustering results (default is ['dbscan', 'hd', 'kmeans']).
        - hub : str 
            Chosen clustering method for hub mapping (default is 'hd').

    Returns:
        - df : pd.DataFrame
            DataFrame with additional columns for hub mapping based on the specified clustering method.
    """
    hub_dict = hub_mapping(hub_data)
    cols_ = [col+'_hub' for col in cols]
    df[cols_] = df[cols].map(lambda x: hub_dict[x] if x in hub_dict.keys() else '')
    return df


import streamlit as st
from ortools.linear_solver import pywraplp


def time_range(start, end, delta):
    """
    Generates a sequence of time values within a specified range.

    Args:
    -----
        - start : datetime
            Start time of the range.
        - end : datetime
            End time of the range.
        - delta : timedelta
            Time interval between consecutive values.

    Yields:
    --------
        - current : datetime 
            Yields successive datetime values within the specified range.
    """
    current = start
    while current < end:
        yield current
        current += delta
        

def assign_hub(i, mechanics_per_hub, index):
    """
    Assigns a hub to a mechanic based on the specified number of mechanics per hub.

    Args:
        i (int): Index representing the mechanic.
        mechanics_per_hub (dict): Dictionary specifying the number of mechanics for each hub.
        index (list): List of hubs.

    Returns:
        str: Hub assigned to the mechanic based on the specified index and number of mechanics per hub.
    """
    l = i
    for key in index:
      #st.info(key)
      if l < mechanics_per_hub[key]:
          hub = key
          #st.info('Yey')
          break
      else:
          #st.write(mechanics_per_hub[key])
          l -= mechanics_per_hub[key]
          #st.write(l)
          hub = ''
    return hub


def gather_variables(mechanics_count_redash : str, 
                     appointments_source : (str, pd.DataFrame) =' sample_appointments.csv') -> tuple:
    """
    Gathers variables including hub data, hub mechanics, and appointments from different sources.

    Args:
    -----
        - mechanics_count_redash : str
            URL or file path for the mechanics count data in Redash.
        - appointments_source : str or pd.DataFrame
            Path or DataFrame containing information about appointments (default is 'sample_appointments.csv').

    Returns:
    --------
        tuple: A tuple containing hub data, the number of hubs, appointments, hub mechanics, and hub service data.
        - hub_data : pd.DataFrame
            Dataframe containing lat/long data of hubs
        - len(hub_data) : int
            number of hubs/number of rows in hub_data
        - appointments : pd.DataFrame
            dataframe of appointments data
        - hub_mechanics : pd.DataFrame
            dataframe of hub-mechanics data (avg appts, mechanics, util rate)
        - df_hub_service : pd.DataFrame
            dataframe of appointments with non-null hub_solver
            
            
    """
    hub_data = generate_hub_data()
    hub_mechanics = pd.read_csv(mechanics_count_redash)
    appointments, df_hub_service = gather_info(appointments_source)
    
    return hub_data, len(hub_data), appointments, hub_mechanics, df_hub_service


# create objects for hubs
def calculate_appointment_distances(hub_data, appointments):
    """
    Calculates Haversine distances between appointments and hub locations.

    Args:
    -----
        - hub_data : pd.DataFrame 
            DataFrame containing hub data (lat, long, lat_rad, long_rad)
        - appointments : pd.DataFrame
            DataFrame containing information about appointments.

    Returns:
    --------
        tuple: A tuple containing a transposed DataFrame of appointment distances and an updated DataFrame with distance information.
        - appointment_distances : pd.DataFrame
            Dataframe with distances between hubs (rows) and appointment_id (columns)
        - appointments : pd.DataFrame
            Dataframe with appointments data with added distance from hubs
    """
    # set hub coords as cluster centers
    cluster_centers = hub_data[['lat_rad', 'long_rad']]
    # TODO: implement choice between haversine and vincenty
    distances = haversine_distances(appointments[['lat_rad', 'long_rad']], 
                                    cluster_centers)
    distances = pd.DataFrame(distances, 
                             columns = hub_data['hub'].tolist())
    # convert radians to km via Earth's radius (r*theta)
    distances = distances.map(lambda x: x*6371.0088)
    appointments = pd.concat([appointments.reset_index(drop=True),
                              distances], axis=1)
    
    appointment_distances = distances.T
    # hubs x appointment_id
    appointment_distances.columns = appointments.dropna(how='all')['appointment_id'].tolist()
    
    return appointment_distances, appointments

def initialize_mechanics_dict(hub_mechanics, column='unique_mechanics'):
    """
    Initializes a dictionary with the count of mechanics per hub from a DataFrame.

    Args:
    -----
        - hub_mechanics : pd.DataFrame
            DataFrame containing information about mechanics per hub.
        - column : str
            Name of the column containing the count of mechanics per hub (default is 'unique_mechanics').

    Returns:
        - mechanics_per_hub : dict
            A dictionary with service locations/hub as keys and the count of mechanics as values.
    """
    hub_mechanics = hub_mechanics.set_index('service_location').to_dict()
    mechanics_per_hub = hub_mechanics[column]
    return mechanics_per_hub

def generate_mechanics_dict(mechanics_per_hub, init_mechanics_dict):
    """
    Generates a dictionary with the count of mechanics per hub, using an initialization dictionary.

    Args:
    -----
        - mechanics_per_hub :  dict
            A dictionary with service locations as keys and the count of mechanics as values.
        - init_mechanics_dict : dict 
            A dictionary with service locations as keys and initial count of mechanics as values.

    Returns:
    --------
        - init_mechanics: dict 
            A dictionary with service locations as keys and the count of mechanics. If the initialization count
              is None, it uses the count from the original mechanics_per_hub dictionary.
    """
    init_mechanics = dict()
    
    for key in init_mechanics_dict.keys():
        if init_mechanics_dict[key] == None:
            init_mechanics[key] = mechanics_per_hub[key]
        else:
            init_mechanics[key] = init_mechanics_dict[key]
    return init_mechanics

def add_actual_assignment(appointments_reference_redash, appointments):
    """
    Adds actual assignment information to the appointments DataFrame based on a reference DataFrame.

    Args:
    -----
        - appointments_reference_redash : str
            Path to the reference DataFrame containing actual assignment information.
        - appointments : pd.DataFrame
            DataFrame containing appointment information, including appointment_id.

    Returns:
    --------
        - appointments : pd.DataFrame
            Modified appointments DataFrame with actual assignment information added.
    """
    df_actual = pd.read_csv(appointments_reference_redash, usecols=['appointment_id',
                                                                    'actual_hub',
                                                                    'pin_address'])
    # df_actual['appointment_id']=df_actual['appointment_id'].astype(str)
    df_actual = df_actual.loc[df_actual['appointment_id'].isin(appointments['appointment_id'].to_list())]#+['21429', '21439','21390', '21392']
    appointments = pd.concat([appointments.set_index('appointment_id'),df_actual.set_index('appointment_id')],axis=1)
    return appointments.reset_index()

def find_current_appointments(appointments, current_time):
    """
    Filters appointments that are ongoing (overlap with the current time) based on the provided current time.

    Args:
    -----
        - appointments : pd.DataFrame 
            DataFrame containing appointment data.
        - current_time : datetime 
            The reference time for filtering.

    Returns:
    --------
        - appointments : pd.DataFrame
            Filtered DataFrame containing ongoing appointments.
    """
    appointments = appointments.loc[appointments.apply(lambda row: dt.strptime(row['time'],"%H:%M")+timedelta(hours=row['duration']),axis=1)>current_time]
    appointments = appointments.loc[appointments.apply(lambda row: dt.strptime(row['time'],"%H:%M")-timedelta(hours=row['duration']),axis=1)<current_time]
    
    # appointments = appointments.loc[appointments['time'].apply(lambda row:datetime.strptime(row,"%H:%M"))>=current_time]
    # appointments = appointments.loc[appointments.apply(lambda row:datetime.strptime(row['time'],"%H:%M")-timedelta(hours=row['duration']),axis=1)<current_time]
    return appointments

def appointments_to_assign(appointments, current_time):
    """
    Filters appointments that are eligible for assignment based on the provided current time.

    Args:
    -----
        - appointments : pd.DataFrame 
            DataFrame containing appointment data.
        - current_time : datetime
            The reference time for filtering.

    Returns:
    --------
        - _appts : pd.DataFrame 
            Filtered DataFrame containing appointments eligible for assignment.
    """
    _appts = appointments.copy()
    # get all current and future available appointments
    _appts = _appts.loc[_appts['time'].apply(lambda row: dt.strptime(row,"%H:%M"))>=current_time]
    # filter all appointments which start/occur witin service duration of start time
    _appts = _appts.loc[_appts.apply(lambda row: dt.strptime(row['time'],"%H:%M")-timedelta(hours=row['duration']),axis=1)<current_time]
    return _appts

def get_free_mechanics(mechanics_per_hub, occupied_mechanics):
    """
    Calculates the number of free mechanics for each hub.

    Args:
        mechanics_per_hub (dict): Dictionary specifying the total number of mechanics for each hub.
        occupied_mechanics (dict): Dictionary specifying the number of occupied mechanics for each hub.

    Returns:
        dict: A dictionary containing the number of free mechanics for each hub.
    """
    free_mechanics = dict()
    for hub in mechanics_per_hub.keys():
        try:
            free_mechanics[hub] = mechanics_per_hub[hub] - occupied_mechanics[hub]
        except:
            free_mechanics[hub] = mechanics_per_hub[hub]
    return free_mechanics

def generate_data_list(mechanics_per_hub, appointments_slice):
    """
    Generates a list of data for each hub based on the specified number of mechanics per hub.

    Args:
    -----
        - mechanics_per_hub : dict
            Dictionary specifying the number of mechanics for each hub.
        - appointments_slice : pd.DataFrame
            DataFrame containing appointment data.

    Returns:
    --------
        - data_list : list 
            List of data where each entry corresponds to an appointment in the appointments_slice.
    """
    data_list = list()
    for hub in appointments_slice.index:
        for i in range(mechanics_per_hub[hub]):
            data_list.append(appointments_slice.loc[hub].tolist())
    return data_list


def or_tools_solver(data_list, appointments_list, mechanics_per_hub, index):
    """
    Solves the assignment problem using Google OR-Tools MIP solver.

    Args:
    -----
        - data_list : list
            List containing the cost matrix for the assignment problem.
        - appointments_list :  list
            List of appointment IDs corresponding to the columns of the cost matrix (ongoing appointments)
        - mechanics_per_hub : dict
            Dictionary specifying the number of FREE mechanics for each hub (free_mechanics)
        - index : list
            List of hubs (indices of appointments_slice)

    Returns:
    --------
        - solution_list: list
            A list of dictionaries representing the assignment solution with appointment IDs and assigned hubs.
            Number of elements in list corresponds to number of appointments in slice
    """
    # Data
    solution_list = list()
    costs = data_list
    num_workers = len(costs)
    num_tasks = len(costs[0])

    # Solver
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return

    # Variables
    # x[i, j] is an array of 0-1 variables, which will be 1
    # if worker i is assigned to task j.
    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[i, j] = solver.IntVar(0, 1, "")

    # Constraints
    # Each worker is assigned to at most 1 task.
    for i in range(num_workers):
        solver.Add(solver.Sum([x[i, j] for j in range(num_tasks)]) <= 1)

    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        solver.Add(solver.Sum([x[i, j] for i in range(num_workers)]) == 1)

    # Objective
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(costs[i][j] * x[i, j])
    solver.Minimize(solver.Sum(objective_terms))

    # Solve
    status = solver.Solve()

    # Print solution.
    if status == pywraplp.Solver.OPTIMAL:# or status == pywraplp.Solver.FEASIBLE
        # st.write(f"Total cost = {solver.Objective().Value()}\n")
        for i in range(num_workers):
            for j in range(num_tasks):
                # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                if x[i, j].solution_value() > 0.75:
                    hub = assign_hub(i, mechanics_per_hub,index)
                    solution_list.append({'appointment_id':appointments_list[j],'hub_solver':hub})
                    # st.write(f"Worker {i} assigned to task {j}." + f" Cost: {costs[i][j]}")
        return solution_list
    else:
        # st.write("No solution found.")
        return solution_list

def main_algorithm(appointments, ddf, mechanics_per_hub, time_interval):
    """
    Main algorithm for assigning hubs to appointments based on availability of mechanics.

    Args:
    -----
        - appointments : pd.DataFrame
            DataFrame containing appointment information, including appointment_id and time.
        - ddf : pd.DataFrame
            DataFrame with distance of appointments to hubs (hubs x appointment_id)
        - mechanics_per_hub : dict
            Dictionary specifying the number of mechanics per hub.
        - time_interval : float
            Time interval for scheduling appointments.

    Returns:
    --------
        - _appointments : pd.DataFrame 
            Modified appointments DataFrame with assigned hub information.
    """
    # get earliest appointment start time
    min_time = dt.strptime(appointments['time'].min(),"%H:%M")
    # get latest appointment start time
    max_time = dt.strptime(appointments['time'].max(),"%H:%M")
    # does not include max_time
    dts = [_dt for _dt in time_range(min_time, max_time, 
                                   timedelta(hours = time_interval))]
    
    # bookings_per_time = appointments['time'].value_counts().reset_index()
    # time_bookings_list = bookings_per_time['index'].to_list()
    _appointments = appointments.copy()
    for current_time in dts:#time_bookings_list
        if isinstance(current_time, str):
            # if element are in str format, convert to datetime
            current_time = dt.strptime(current_time,'%H:%M')
        
        # determine number of appointments at this time
        appointments_list = find_current_appointments(_appointments, current_time)
        # determine number of unavailable mechanics based on number of appointments
        occupied_mechanics = appointments_list['hub_solver'].value_counts().to_dict()
        # unoccupied mechanics
        free_mechanics = get_free_mechanics(mechanics_per_hub, occupied_mechanics)
        
        appointments_ongoing = appointments_to_assign(_appointments, current_time)
        
        appointments_list = appointments_ongoing.loc[appointments_ongoing['hub_solver'].isnull(),'appointment_id'].tolist()
        
        appointments_slice = ddf[appointments_list]
        # st.write(free_mechanics)
        # st.write(appointments_ongoing)
        data_list = generate_data_list(free_mechanics, appointments_slice)
        # list of dict - keys (appointment_id, hub_solver)
        solution_list = or_tools_solver(data_list, appointments_list,
                                        free_mechanics, appointments_slice.index)
        for item in solution_list:
            _appointments.loc[_appointments['appointment_id']==item['appointment_id'], 'hub_solver'] = item['hub_solver']
    
    return _appointments

def find_similarity_score(appointments):
    """
    Calculate the similarity score between the actual and assigned hubs for appointments.
    'actual_hub' vs 'hub_solver'

    Args:
    -----
        - appointments : pd.DataFrame
            DataFrame containing appointment information with actual and assigned hub columns.

    Returns:
    --------
        similarity : float
            Similarity score as a percentage.
    """
    appointments['is_the_same'] = appointments['actual_hub'].astype(str) == appointments['hub_solver'].astype(str)
    similarity = np.floor(appointments['is_the_same'].sum()/len(appointments)*100)
    # st.metric(label = 'Similarity',value=similarity)
    return similarity

def show_similarity(similarity):
    """
    Display the similarity score using a metric visualization.

    Args:
        similarity (float): Similarity score as a percentage.

    Returns:
        None
    """
    st.metric(label = 'Similarity',value=str(similarity)+'%')
    
def cluster_appointments(appointments_source, config, init_mechanics_dict):
    """
    Cluster and assign appointments to service hubs based on various algorithms.

    Args:
        appointments_source (str): Path to the source file containing appointment data.
        config (dict): Loaded dictionary of configuration file in JSON format.
        init_mechanics_dict (dict): Initial mechanics count per hub.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with clustered and assigned appointments.
            - float: Similarity score between actual and assigned hubs.
            - dict: Final mechanics count per hub after assignment.
            - pd.DataFrame: DataFrame with hub service information.
    """
    # with open(config_file) as json_file:
    #     config = json.load(json_file)
    mechanics_count_redash = config['mechanics_count_redash']
    time_interval = config['time_interval']
    mechanic_col = config['redash_mechanic_col']
    
    hub_data, hubs, appointments, hub_mechanics, df_hub_service = gather_variables(mechanics_count_redash,appointments_source)
    ## var = gather_variables(config['mechanics_count_redash'], appointments_source)
    
    # ddf - appointment_distances (hub x appointment_id)
    ddf, appointments = calculate_appointment_distances(hub_data, appointments)
    ## ddf, appointments = calculate_apppointment_distances(var['hub_data'], appointments)
    # gather clustering labels from various algo
    appointments, status = gather_clusters(appointments, hub_data)
    ## appointments, status = gather_clusters(appointments, var['hub_data'])
    
    if status == ',,':
        return appointments,status,np.nan,np.nan
    
    # mapping cluster labels (for each method) to hubs
    appointments = apply_hub_map(appointments, hub_data)
    
    ## mechanics_per_hub = initialize_mechanics_dict(var['hub_mechanics'], config['mechanic_col'])
    mechanics_per_hub = initialize_mechanics_dict(hub_mechanics, mechanic_col)
    mechanics_per_hub = generate_mechanics_dict(mechanics_per_hub, init_mechanics_dict)
    
    # assign values of hub_solver
    appointments = main_algorithm(appointments, ddf, 
                                  mechanics_per_hub, time_interval)
    
    # assign hub for each appointment with null "hub_solver"
    appointments.loc[appointments['hub_solver'].isnull(),'hub_solver'] = appointments.loc[appointments['hub_solver'].isnull(),'hd_hub']
    
    appointments['type'] = 'Home Service'
    df_hub_service['type'] = 'Hub Service'
    
    appointments = pd.concat([appointments, df_hub_service],ignore_index=True)
    appointments = appointments.sort_values(by='time')
    
    similarity = find_similarity_score(appointments)
    
    return appointments,similarity,mechanics_per_hub,df_hub_service

time_interval = 2 #hours
mechanics_count_redash='http://app.redash.licagroup.ph/api/queries/237/results.csv?api_key=YuHtFpGduQNyxubodM6t0Lonpr6obCiszR0Vh5vd'
appointments_reference_redash = "http://app.redash.licagroup.ph/api/queries/225/results.csv?api_key=CNrNQGAhqA3BXIa5N9nvMBzGxs1frIMUY63fFMvi"
show_columns = ['appointment_id','time','time_end','duration','lat','long','pin_address','makati_hub','sucat_hub','starosa_hub','hub_solver','actual_hub','is_the_same']

# if __name__ == '__main__':
#     hub_data, hubs,appointments,hub_mechanics = gather_variables(mechanics_count_redash,'config.json','sample_appointments.csv')
#     appointments = add_actual_assignment(appointments_reference_redash,appointments)
#     distances,ddf,appointments = calculate_appointment_distances(hub_data,appointments)
#     appointments = gather_clusters(appointments, hub_data)
#     appointments = apply_hub_map(appointments,hub_data)
#     mechanics_per_hub = initialize_mechanics_dict(hub_mechanics,'unique_mechanics')
#     mechanics_per_hub = generate_mechanics_dict(mechanics_per_hub)
#     appointments = main_algorithm(appointments,ddf,mechanics_per_hub,time_interval)
#     similarity = find_similarity_score(appointments)
#     show_similarity(similarity)
#     st.write(distances)
#     st.write(appointments[show_columns])
#     #st.write(mechanics_per_hub)


