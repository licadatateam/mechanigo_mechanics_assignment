# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:28:00 2023

@author: Arvin Jay
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from datetime import timedelta, date
from datetime import datetime as dt
import pandas as pd
import json
import os
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(message)s',
                    datefmt = '%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)


req_cols = ['time_str','fullname','pin_address','appointment_id','lat','long','service_category','duration','time','time_end']

def load_hub_details(config_file, hub):
    """
    Load hub details from a JSON configuration file.

    Args:
        config_file (str): Path to the JSON configuration file.
        hub (str): Hub identifier.

    Returns:
        dict: Start details for Mechanigo.
        dict: End details for Mechanigo.
    """
    with open(config_file) as json_file:
        config_file = json.load(json_file)
    mechanigo_start = {'time_str':'Duty start/end',
                    'fullname':'Rapide Makati',
                    'pin_address': hub,
                    'lat': config_file[hub]['lat'],
                    'long': config_file[hub]['long'],
                    'service_category':'Depart from hub',
                    'appointment_id':'0',
                    'time': '05:00',
                    }
    
    mechanigo_end = {'time_str':'Duty end',
                    'fullname':'Mechanigo hub',
                    'pin_address': hub,
                    'lat': config_file[hub]['lat'],
                    'long': config_file[hub]['long'],
                    'service_category':'Return to hub',
                    'appointment_id':'0',
                    'time': '19:00',
                    'time_end':'19:01'
                    }
    return mechanigo_start,mechanigo_end



def process_data(df_file):
    """
    Process appointment or matrix data.

    Args:
        df_file (str or pd.DataFrame): Path to the CSV file or DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    if isinstance(df_file, str):
        matrix = pd.read_csv(df_file)
        if 'Unnamed: 0' in matrix.columns and 'appointment_id' not in matrix.columns:
            matrix = matrix.rename(columns={'Unnamed: 0':'appointment_id'})
            matrix = matrix.set_index('appointment_id')
        elif 'Unnamed: 0' in matrix.columns:
            matrix = matrix.rename(columns={'Unnamed: 0':'index'})
            matrix = matrix.set_index('index')
    else:
        matrix = df_file
    return matrix

def load_data(distance_matrix='sample_dm.csv', time_matrix='sample_tm.csv', appointments='sample_a.csv'):
    """
    Load distance matrix, time matrix, and appointments data.

    Args:
        distance_matrix (str or pd.DataFrame): Path to the distance matrix CSV file or DataFrame.
        time_matrix (str or pd.DataFrame): Path to the time matrix CSV file or DataFrame.
        appointments (str or pd.DataFrame): Path to the appointments CSV file or DataFrame.

    Returns:
        pd.DataFrame: Processed distance matrix.
        pd.DataFrame: Processed time matrix.
        pd.DataFrame: Processed appointments data.
    """
    distance_matrix = process_data(distance_matrix)
    time_matrix = process_data(time_matrix)
    appointments = process_data(appointments)
    return distance_matrix,time_matrix,appointments


def insert_start_info(timeline, mechanigo_start):
    """
    Insert start information to the timeline.

    Args:
        timeline (pd.DataFrame): Timeline DataFrame.
        mechanigo_start (dict): Start details for Mechanigo.

    Returns:
        pd.DataFrame: Updated timeline.
    """
    
    timeline = pd.concat([pd.DataFrame([mechanigo_start]),
                          timeline
                          ])
    return timeline


def insert_end_info(timeline, mechanigo_end):
    """
    Insert end information to the timeline.

    Args:
        timeline (pd.DataFrame): Timeline DataFrame.
        mechanigo_end (dict): End details for Mechanigo.

    Returns:
        pd.DataFrame: Updated timeline.
    """
    
    timeline = pd.concat([timeline,
                          pd.DataFrame([mechanigo_end])
                          ])
    return timeline

def get_transit_row(timeline, i, time_matrix):
    """
    Get a transit row for the timeline.

    Args:
        timeline (pd.DataFrame): Timeline DataFrame.
        i (int): Index in the timeline.
        time_matrix (pd.DataFrame): Time matrix.

    Returns:
        dict: Transit row details.
    """
    source_id = timeline.iloc[i]['appointment_id']
    destination_id = timeline.iloc[i+1]['appointment_id']
    duration = time_matrix.loc[source_id,destination_id]
    duration_ = int(round(duration/60,0))
    if i != len(timeline)-100:
        time = timeline.iloc[i+1]['time']
        time_ = dt.strptime(time, '%H:%M') 
        time_ = time_-timedelta(seconds = int(duration))
    else:
        time = timeline.iloc[i]['time']
        time_ = dt.strptime(time, '%H:%M') 
        time_ = time_+timedelta(hours = duration)
    time_str = dt.strftime(time_,"%I:%M %p")
    time_ = dt.strftime(time_,"%H:%M")
    fullname = 'In transit'
    # pin_address = ''
    lat = float('nan')
    long = float('nan')
    service_category = 'Travel'
    appointment_id = '0'
    
    row = {
        'time_str' : time_str,
        'fullname' : fullname,
        'pin_address' : f'Duration: Around {duration_} minutes',
        'lat':lat,
        'long':long,
        'service_category':service_category,
        'appointment_id':appointment_id,
        'time':time_,
        'duration':duration_
        }
    return row

def transit_df(timeline, time_matrix):
    """
    Create a DataFrame for transit rows in the timeline.

    Args:
        timeline (pd.DataFrame): Timeline DataFrame.
        time_matrix (pd.DataFrame): Time matrix.

    Returns:
        pd.DataFrame: DataFrame for transit rows.
    """
    transit_rows = []
    for i in range(len(timeline)-1):
        transit_row = get_transit_row(timeline,i,time_matrix)
        transit_rows.append(transit_row)
    return pd.DataFrame(transit_rows)

def insert_transit_df(current_booking, transit_rows):
    """
    Insert transit rows to the current booking.

    Args:
        current_booking (pd.DataFrame): Current booking DataFrame.
        transit_rows (pd.DataFrame): DataFrame for transit rows.

    Returns:
        pd.DataFrame: Updated DataFrame with transit rows.
    """
    temp = pd.concat([current_booking,transit_rows])
    temp_ = temp.sort_values(by='time')
    return temp_

def get_end_time(row, selected_date):
    """
    Calculate end time based on the start time and duration.

    Args:
        row (pd.Series): Row from the DataFrame.
        selected_date (datetime.date): Selected date.

    Returns:
        tuple: Start time and end time.
    """
    start_time = dt.strptime(row['time'], '%H:%M')
    #start_time = datetime.combine(selected_date,start_time.time())
    end_time = start_time + timedelta(minutes=int(row['duration']))
    return (start_time,end_time)


def get_solution(distance_matrix, mechanic_count, mechanic_capacity, time_limit):
    """
    Get the solution using the OR-Tools library.

    Args:
        distance_matrix (pd.DataFrame): Distance matrix.
        mechanic_count (int): Number of mechanics.
        mechanic_capacity (int): Capacity of each mechanic.
        time_limit (int): Time limit in seconds to find all solutions.

    Returns:
        pywrapcp.RoutingModel.SolveWithParameters: Solution.
        ortools.constraint_solver.pywrapcp.RoutingModel: Routing model.
        ortools.constraint_solver.pywrapcp.RoutingIndexManager: Routing index manager.
    """
    
    def create_data_model(distance_matrix, mechanic_count, mechanic_capacity, depot_index=0):
        """
        Create the data model for the OR-Tools solver.

        Args:
            distance_matrix (pd.DataFrame): Distance matrix.
            mechanic_count (int): Number of mechanics.
            mechanic_capacity (int): Capacity of each mechanic.
            depot_index (int): Index of the depot.

        Returns:
            dict: Data model for the OR-Tools solver.
        """
        data = dict()
        data['distance_matrix'] = distance_matrix.applymap(lambda x: int(round(x/1000.,0))).values.tolist()
        data['num_vehicles'] = mechanic_count
        data['depot'] = depot_index
        data['demands'] = [0]+[1]*(len(distance_matrix)-1)
        data['vehicle_capacities'] = [mechanic_capacity]*mechanic_count
        return data



    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """
        Returns the distance between the two nodes.

        Args:
            from_index (int): From index.
            to_index (int): To index.

        Returns:
            int: Distance between the nodes.
        """
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    # Add Capacity constraint.
    def demand_callback(from_index):
        """
        Returns the demand of the node.

        Args:
            from_index (int): From index.

        Returns:
            int: Demand of the node.
        """
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]
    
    # Instantiate the data problem
    data = create_data_model(distance_matrix,mechanic_count,mechanic_capacity)
    # st.write(data['distance_matrix'])
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
                                            demand_callback_index,
                                            0,  # null capacity slack
                                            data["vehicle_capacities"],  # vehicle maximum capacities
                                            True,  # start cumul to zero
                                            "Capacity",
                                            )
    
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(time_limit)
    
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    
    return solution, routing, manager

def return_solution(solution, routing, manager, mechanic_count, reference):
    """
    Return the solution details.

    Args:
        solution (ortools.constraint_solver.pywrapcp.RoutingModel.SolveWithParameters): Solution from the OR-Tools solver.
        routing (ortools.constraint_solver.pywrapcp.RoutingModel): Routing model.
        manager (ortools.constraint_solver.pywrapcp.RoutingIndexManager): Routing index manager.
        mechanic_count (int): Number of mechanics.
        reference (list): Reference list for distance matrix columns.

    Returns:
        list: Mechanic plan.
        float: Total distance of the plan.
    """

    index_reference = list(reference)# distance_matrix.columns
    mechanic_plan = list()
    total_distance = 0
    for vehicle_id in range(mechanic_count):
        index = routing.Start(vehicle_id)
        result = list()
        route_distance = 0
        #total_appointments = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            #total_appointments += data["demands"][node_index]
            
            apt_i = index_reference[node_index]
            result.append(apt_i)        
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
                )
        if result != ['0']:
            mechanic_plan.append(result)
        total_distance += route_distance
        
    return mechanic_plan, total_distance


def insert_transit_activity(mechanic_plan, 
                            appointments_summary, 
                            time_matrix, 
                            mechanigo_end):
    """
    Insert transit activity to the timeline.

    Args:
        mechanic_plan (list): Mechanic plan.
        appointments_summary (pd.DataFrame): Appointments summary DataFrame.
        time_matrix (pd.DataFrame): Time matrix.
        mechanigo_end (dict): End details for Mechanigo.

    Returns:
        pd.DataFrame: Timeline DataFrame.
    """
    timeline = pd.DataFrame()
    for mechanic in mechanic_plan:
        mechanic_id = mechanic_plan.index(mechanic)+1
        current_booking = pd.DataFrame()
        for booking in mechanic:
            current_booking_ = appointments_summary.loc[appointments_summary['appointment_id']==booking]
            current_booking = pd.concat([current_booking,current_booking_])
        current_booking = insert_end_info(current_booking,mechanigo_end).reset_index(drop=True)
        transit_rows = transit_df(current_booking, time_matrix)
        current_booking = insert_transit_df(current_booking,transit_rows)
        current_booking.loc[:,'mechanic'] = mechanic_id
        timeline = pd.concat([timeline,current_booking])

    return timeline

def summarize_appointments(appointments, mechanigo_start, req_cols):
    """
    Summarize appointments data.

    Args:
        appointments (pd.DataFrame): Appointments DataFrame.
        mechanigo_start (dict): Start details for Mechanigo.
        req_cols (list): Required columns.

    Returns:
        pd.DataFrame: Appointments summary DataFrame.
    """
    appointments_summary = insert_start_info(appointments,mechanigo_start)
    appointments_summary = appointments_summary[req_cols]
    appointments_summary['duration'] = appointments_summary['duration'].apply(lambda x: x*60)
    appointments_summary = appointments_summary.reset_index(drop = True)
    return appointments_summary

def calculate_time_end(row):
    """
    Calculate the end time based on the start time and duration.

    Args:
        row (pd.Series): DataFrame row.

    Returns:
        datetime: End time.
    """
    if row['time_end'] != None:
        value =row['time_end']
    else:
        value = row['time_start'] + timedelta(minutes = row['duration'])
    return value

def process_timeline(timeline, selected_date, hub, def_mins=30):
    """
    Process the timeline DataFrame.

    Args:
        timeline (pd.DataFrame): Timeline DataFrame.
        selected_date (date): Selected date.
        hub (str): Hub identifier.
        def_mins (int): Default minutes.

    Returns:
        pd.DataFrame: Processed timeline DataFrame.
    """
    
    timeline['duration'] = timeline['duration'].fillna(def_mins)#.apply(lambda x:x/60.)
    timeline['time_end'] = timeline.apply(lambda row: calculate_time_end(row),axis = 1)
    # timeline['start'] = timeline['time'].apply(lambda t: datetime.strptime(t, '%H:%M'))
    # timeline['finish'] = timeline['time_end']
    timeline[['start','finish']] = timeline.apply(lambda row: get_end_time(row,selected_date),axis=1,result_type='expand')
    timeline['mechanicActivity'] = timeline.apply(lambda row: f"{row['mechanic']} {row['service_category']}",axis = 1) #str(row['mechanic'])+' '+row['service_category']
    timeline['start'] = timeline['start'].apply(lambda x:str(x)[-8:])
    timeline['finish'] = timeline['finish'].apply(lambda x:str(x)[-8:])
    timeline['hub'] = hub
    return timeline.reset_index(drop=True)

def clean_timeline(timeline, hub):
    """
    Clean and format the timeline DataFrame for better presentation.

    Args:
        timeline (pd.DataFrame): Original timeline DataFrame.
        hub (str): Name of the hub.

    Returns:
        pd.DataFrame: Cleaned timeline DataFrame with selected columns.
    """
    timeline_temp = timeline[['start','finish','mechanic','mechanicActivity','appointment_id','service_category','pin_address','fullname']].reset_index(drop=True)
    timeline_temp['start'] = timeline_temp['start'].apply(lambda x:str(x)[-8:])
    timeline_temp['finish'] = timeline_temp['finish'].apply(lambda x:str(x)[-8:])
    timeline_temp['hub'] = hub
    return timeline_temp

if __name__ == "__main__":
    # Program Start
    start = dt.now()
    logging.debug('Program start.')
    
    hub = 'makati_hub'
    mechanic_count = 23
    os.chdir('..')
    config_file =  'config.json'
    # Load Hub Details from JSON/config file
    mechanigo_start, mechanigo_end = load_hub_details(config_file, hub)
    selected_date = date(2023,10,18)
    mechanic_capacity = 2 # appointments per day
    distance_matrix = 'sample_dm.csv'
    time_matrix = 'sample_tm.csv'
    appointments='sample_a.csv'
    time_limit = 15 #seconds to find solution
    
    # st.info('start')
    
    distance_matrix, time_matrix, appointments = load_data()
    # st.write(distance_matrix,time_matrix,appointments)
    solution, routing, manager = get_solution(distance_matrix,mechanic_count,mechanic_capacity,time_limit)
    
    mechanic_plan,total_distance = return_solution(solution, routing, manager,mechanic_count,distance_matrix.columns)
    appointments_summary = summarize_appointments(appointments,mechanigo_start,req_cols)
    timeline = insert_transit_activity(mechanic_plan,
                                       appointments_summary,
                                       time_matrix,
                                       mechanigo_end)
    timeline = process_timeline(timeline,selected_date, hub)
    # st.write(timeline)
    # st.info('Final')
    # st.write(timeline[['start','finish','mechanic','mechanicActivity','appointment_id','service_category','pin_address','fullname']])
    
    # Program End
    end = dt.now()
    logging.debug('Program finished! Runtime: {0:.2f}s'.format((end-start).total_seconds()))
