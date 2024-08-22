"""
The Fruit Distribution Problem for the PuLP Modeller

Authors: Collins Patrick Ohagwu, 1st Aug 2024
"""
# Import Dependencies
import pulp # Import PuLP modeler functions
import gradio as gr # Import Gradio for UI

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading and Inspection

def assign_demand_nodes_by_type(df_demand, df_matrix, demand_n_matrix_col_id, demand_n_matrix_col_id_category):
  matrix_demand_nodes_by_type_dict = df_matrix[[demand_n_matrix_col_id, demand_n_matrix_col_id_category]].set_index(demand_n_matrix_col_id).to_dict()[demand_n_matrix_col_id_category]
  df_demand[demand_n_matrix_col_id_category] = df_demand[demand_n_matrix_col_id].map(matrix_demand_nodes_by_type_dict)

  return df_demand

def load_dataset(supply_dir, demand_dir, matrix_dir):
  """
  Load the dataset from the given directories.
  """
  df_supply = pd.read_csv(supply_dir)
  df_demand = pd.read_csv(demand_dir)
  df_matrix = pd.read_csv(matrix_dir)

  return df_supply, df_demand, df_matrix

### Preprocessing Functions
# These functions ensure the data schema (with the currect data types) and check the nodes in `supply` and `demand` sheet against those of the `preference matrix`.

def enforce_data_schema(df_supply, supply_col_id, supply_col_val_list, df_demand, demand_col_val, df_matrix, mx_col_demand_id, mx_col_demand_id_cat):
  """
  Convert data types of pandas df columns to the correct types.
  """
  df_supply[supply_col_id] = df_supply[supply_col_id].astype(str)
  supply_node_list = df_supply[supply_col_id].unique().tolist()
  for col in supply_col_val_list:
    df_supply[col] = df_supply[col].astype(int)

  df_demand[mx_col_demand_id] = df_demand[mx_col_demand_id].astype(str)
  demand_node_list = df_demand[mx_col_demand_id].unique().tolist()
  df_demand[demand_col_val] = df_demand[demand_col_val].astype(int)
  df_demand[mx_col_demand_id_cat] = df_demand[mx_col_demand_id_cat].astype(str)

  df_matrix[mx_col_demand_id] = df_matrix[mx_col_demand_id].astype(str)
  matrix_demand_node_list = df_matrix[mx_col_demand_id].unique().tolist()
  df_matrix[mx_col_demand_id_cat] = df_matrix[mx_col_demand_id_cat].astype(str)

  # Convert all columns to float
  matrix_supply_node_list = df_matrix.drop(columns=[mx_col_demand_id, mx_col_demand_id_cat]).columns.to_list()
  for col in matrix_supply_node_list:
    df_matrix[col] = df_matrix[col].astype(float)

  return df_supply, supply_node_list, df_demand, demand_node_list, df_matrix, matrix_supply_node_list, matrix_demand_node_list

### Check Nodes in Supply and Demand Sheet Against Pref. Matrix
def check_matrix_completeness(df_supply, supply_node_list, df_demand, demand_node_list, df_matrix, matrix_supply_node_list, matrix_demand_node_list):
  """
  Check that the Farm Codes in the supply sheet are present in the preference matrix.
  Also check that the SKU's in the demand sheet are present in the preference matrix.
  Else, remove any SKU's or Farm Codes that are not present in the preference matrix.
  """
  ################################################
  # SUPPLY SHEET
  # Check that the Farm Codes in the supply sheet are present in the preference matrix
  supply_node_not_in_matrix = [node for node in supply_node_list if node not in matrix_supply_node_list]
  matrix_supply_node_not_in_supply = [node for node in matrix_supply_node_list if node not in supply_node_list]

  # Instanciate the log messages
  log_messages = []
  if len(supply_node_not_in_matrix) > 0 | len(matrix_supply_node_not_in_supply) > 0:

    log_messages.append(f'# Farm Codes in the supply sheet but not in the preference matrix: {len(supply_node_not_in_matrix)} \n Node List: {supply_node_not_in_matrix}')
    log_messages.append('WARNING! Values from the Supply Nodes above will not be assigned to any Demand Node.\n')
    log_messages.append(f'# Farm Codes in the preference matrix but not in the supply sheet: {len(matrix_supply_node_not_in_supply)} \n Node List: {matrix_supply_node_not_in_supply}')
    log_messages.append('==============================================================')
    # Exlude nodes from supply sheet
    df_supply = df_supply[~df_supply['Code'].isin(supply_node_not_in_matrix)]
    # Exclude nodes from the preference matrix
    df_matrix = df_matrix.drop(columns=matrix_supply_node_not_in_supply)

  else:
    log_messages.append('All Farm Codes in the Supply Sheet are present in the Preference Matrix.')
    log_messages.append('==============================================================')

  ################################################
  # DEMAND SHEET
  # Check that the SKU's in the demand sheet are present in the preference matrix
  demand_node_not_in_matrix = [node for node in demand_node_list if node not in matrix_demand_node_list]
  matrix_demand_node_not_in_demand = [node for node in matrix_demand_node_list if node not in demand_node_list]

  if len(demand_node_not_in_matrix) > 0 | len(matrix_demand_node_not_in_demand) > 0:

    log_messages.append(f'# SKU\'s in the demand sheet but not in the preference matrix: {len(demand_node_not_in_matrix)} \n Node List: {demand_node_not_in_matrix}')
    log_messages.append('WARNING! Values in the Demand Nodes above will not be satisfied from any Supply Node.\n')
    log_messages.append(f'# SKU\'s in the preference matrix but not in the demand sheet: {len(matrix_demand_node_not_in_demand)} \n Node List: {matrix_demand_node_not_in_demand}')
    log_messages.append('==============================================================')
    # Exlude nodes from demand sheet
    df_demand = df_demand[~df_demand['SKU'].isin(demand_node_not_in_matrix)]
    # Exclude nodes from the preference matrix
    df_matrix = df_matrix.drop(columns=matrix_demand_node_not_in_demand)

  else:
    log_messages.append('All SKU\'s in the Demand Sheet are present in the Preference Matrix.')
    log_messages.append('==============================================================')

  return log_messages, df_supply, df_demand, df_matrix

# #########################################
# LP Model Functions
'''
  Functions to handle the creation of linear programming **descision variables, update the variables to include dummy demand nodes
  (to handle excess supply) and dummy supply nodes (to handle excess demand), create and solve the LP problem, and interpret and output
  the results as pandas dataframes**.
'''

### Descision Variables
def generate_variable_dictionaries(df_supply: pd.DataFrame, supply_id: str, supply_val: str, df_demand: pd.DataFrame, demand_id: str, demand_val: str, df_matrix: pd.DataFrame):
  """
  Generate dictionaries for decision variables
  Args:
    df_supply: df with supply data
    supply_id: column name for supply id
    supply_val: column name for supply value
    df_demand: df with demand data
    demand_id:  column name for demand id
    demand_val: column name for demand value
    df_matrix:  df for the preference matrix per supply (row indeces) and demand (column names)

    Returns:
    supply_dict: dictionary for the number of units of supply for each supply node
    demand_dict: dictionary for the number of units of demand for each demand node
    preference_dict: dictionary for the preference matrix
  """
  # Initialize log messages
  log_messages = []
  log_messages.append(f'SOLVING FOR `{supply_val}` TYPE')
  log_messages.append('\n==============================================================\n')

  # Dictionary for the number of units of supply for each supply node
  supply_dict = {row[supply_id]: row[supply_val] for _, row in df_supply.iterrows()}

  # Dictionary for the number of units of demand for each demand node
  demand_dict = {row[demand_id]: row[demand_val] for _, row in df_demand.iterrows()}

  # Dictionary of dictionary for the preference matrix
  df_matrix = df_matrix.transpose() # Transpose and use the first row as header
  new_header = df_matrix.iloc[0] #grab the first row for the header
  df_matrix = df_matrix[1:] #take the data less the header row
  df_matrix.columns = new_header #set the header row as the df header
  preference_dict = df_matrix.to_dict('index')

  # Print the number of supply and demand nodes
  log_messages.append('Before Dummy Update')
  log_messages.append(f'# Supply Nodes: {len(supply_dict)}')
  log_messages.append(f'# Demand Nodes: {len(demand_dict)}')
  log_messages.append(f'Preference df Shape: {df_matrix.shape[0]} Supply Rows | {df_matrix.shape[1]} Demand Columns')

  # Calculate the total supply and demand
  log_messages.append(f'Total Supply: {int(sum(supply_dict.values())):,}')
  log_messages.append(f'Total Demand: {int(sum(demand_dict.values())):,}')
  log_messages.append('\n==============================================================\n')

  return log_messages, supply_dict, demand_dict, preference_dict

def update_vd_for_dummy_nodes(log_messages, supply_dict: dict, demand_dict: dict, preference_dict: dict):
  """
  Update supply and demand dictionaries with dummy nodes for excess or deficit supply conditions
  """
  # Calculate the total supply and demand
  total_supply = sum(supply_dict.values())
  total_demand = sum(demand_dict.values())

  # Update demand nodes with Dummy nodes
  dummy_demand_node = 'DummyDemandNode'
  supply_excess = total_supply - total_demand

  # Update demand dict with Dummy EXCESS demand (to cover excess supply)
  if supply_excess > 0:
    demand_dict[dummy_demand_node] = abs(supply_excess)
    log_messages.append(f'Dummy Excess Demand: {abs(supply_excess):,}')

  # Update supply nodes with Dummy nodes
  dummy_supply_node = 'DummySupplyNode'

  # Update supply dict with Dummy EXCESS supply (to cover excess demand)
  if supply_excess < 0:
    supply_dict[dummy_supply_node] = abs(supply_excess)
    log_messages.append(f'Dummy Excess Supply: {abs(supply_excess):,}')

  # Update preference dict with Dummy Supply and Demand nodes
  preference_df = pd.DataFrame(preference_dict).T # as pandas dataframe for updates
  # Add new dummy supply to the preference matrix if present in the supply dict keys (nodes)
  if dummy_supply_node in supply_dict.keys():
    preference_df.loc[dummy_supply_node] = 0
  # Add new dummy demand to the preference matrix if present in the demand dict keys (nodes)
  if dummy_demand_node in demand_dict.keys():
    preference_df[dummy_demand_node] = 0
  # Convert back to dictionary
  preference_dict = preference_df.to_dict('index')

  # Print the number of supply and demand nodes after updating
  log_messages.append('After Dummy Update')
  log_messages.append(f'# Supply Nodes: {len(supply_dict)}')
  log_messages.append(f'# Demand Nodes: {len(demand_dict)}')
  log_messages.append(f'Preference df Shape: {preference_df.shape[0]} Supply Rows | {preference_df.shape[1]} Demand Columns')

  # Calculate the total supply and demand after updating
  log_messages.append(f'Total Supply: {int(sum(supply_dict.values())):,}')
  log_messages.append(f'Total Demand: {int(sum(demand_dict.values())):,}')
  log_messages.append('\n==============================================================\n')

  return log_messages, supply_dict, demand_dict, preference_dict

### Problem Statements/ Objective Functions/ Constraints

def create_and_solve_lp_problem(log_messages, supply_dict: dict, demand_dict: dict, preference_dict: dict):

  """
  Create and solve the LP problem
  """

  # Creates the 'prob' variable to contain the problem data
  prob = pulp.LpProblem("Fruit_Distribution_Problem", pulp.LpMaximize)

  # Creates a list of tuples containing all the possible routes for transport
  routes_tup = [(s, d) for s in supply_dict.keys() for d in demand_dict.keys()]

  # A dictionary called 'Vars' is created to contain the referenced variables(the routes)
  vars = pulp.LpVariable.dicts("Route", (supply_dict.keys(), demand_dict.keys()), 0, None, pulp.LpInteger)

  # The objective function is added to 'prob' first
  prob += (
      pulp.lpSum([vars[s][d] * preference_dict[s][d] for (s, d) in routes_tup]),
      "Sum_of_Preference_Matrix",
      )

  # The supply maximum constraints are added to prob for each supply node
  for s in supply_dict.keys():
      prob += (
          pulp.lpSum([vars[s][d] for d in demand_dict.keys()]) <= supply_dict[s],
          f"Sum_of_Fruits_out_of_Farm_{s}",
      )

  # The demand minimum constraints are added to prob for each demand node
  for d in demand_dict.keys():
      prob += (
          pulp.lpSum([vars[s][d] for s in supply_dict.keys()]) >= demand_dict[d],
          f"Sum_of_Fruits_into_SKU_{d}",
      )

  # The problem is solved using PuLP's choice of Solver
  prob.solve() # takes no argument, uses default solver

  # The status of the solution is printed to the screen
  log_messages.append(f"Solution Status: {pulp.LpStatus[prob.status]}")
  log_messages.append('\n==============================================================\n')

  return log_messages, prob

def get_results(prob: pulp.LpProblem):
  """
  Get results in matrix and direct form
  """

  # Process matrix results, (all values in a grid form)
  results = {}
  # Each of the variables is printed with it's resolved optimum value
  for route in prob.variables():
    # print(f'{route.name}: {route.varValue}')
    route_name = route.name.split("_")
    supply = route_name[1]
    demand = route_name[2]

    if supply not in results:
      results[supply] = {}

    results[supply][demand] = int(route.varValue)

  df_matrix_results = pd.DataFrame(results)


  # Process direct results, (only none zero values)
  results = []
  # Each of the variables is printed with it's resolved optimum value
  for route in prob.variables():
    # Only non zero values
    if route.varValue > 0:

      route_name = route.name.split("_")
      supply = route_name[1]
      demand = route_name[2]

      results_route = []
      results_route.append(supply)
      results_route.append(demand)
      results_route.append(int(route.varValue))

      results.append(results_route)

    df_direct_results = pd.DataFrame(results, columns=['Supply (Farm Codes)', 'Demand (SKU)', 'Units (Boxes)'])

  return df_matrix_results, df_direct_results

### Call Linear Programming Functions with Functions

def run_lp_model(df_supply: pd.DataFrame, supply_id: str, supply_col_val_list: list, df_demand: pd.DataFrame, demand_id: str, demand_val: str, df_matrix: pd.DataFrame):
  """
  Run the LP model
  """
  # Initialize model list
  models = []
  # Initialize log messages
  log_messages = []
  # Iterate through each supply value
  for supply_val in supply_col_val_list:
    # Generate dictionaries of decision variables
    logs, supply_dict, demand_dict, preference_dict = generate_variable_dictionaries(df_supply, supply_id, supply_val, df_demand, demand_id, demand_val, df_matrix)
    # Update supply and demand dictionaries with dummy nodes for excess or deficit supply conditions
    logs, supply_dict, demand_dict, preference_dict = update_vd_for_dummy_nodes(logs, supply_dict, demand_dict, preference_dict)
    # Create and solve lp problem
    logs, model = create_and_solve_lp_problem(logs, supply_dict, demand_dict, preference_dict)
    # Append model to list
    models.append(model)
    # Append logs to list
    log_messages.append(logs)
  return log_messages, models

def process_model_results(models: list):
  """
  Process the model results
  """
  # Initialize results list
  df_matrix_results = []
  df_direct_results = []

  # Initialize log messages
  log_messages = []
  log_messages.append('==============================================================')
  log_messages.append('PROCESSING MODEL RESULTS')
  log_messages.append('==============================================================')

  count = 1

  # Iterate through each model
  for model in models:
    # Get results in matrix and direct form
    if model.status == 1:
      df_matrix_result, df_direct_result = get_results(model)
      df_matrix_results.append(df_matrix_result)
      df_direct_results.append(df_direct_result)
      log_messages.append(f'Problem {count} solved!')
      # Write lp problem to file
      model_name = f"Fruit_Distribution_Problem_{count}.lp"
      model.writeLP(model_name)
      log_messages.append(f"The problem is written to {model_name}")
      log_messages.append('==============================================================')

    else:
      log_messages.append('Problem not solved!')
      log_messages.append('==============================================================')

    df_matrix_results_concat = pd.concat(df_matrix_results, axis=0)
    df_direct_results_concat = pd.concat(df_direct_results, axis=0)

    count += 1

  return log_messages, df_matrix_results_concat, df_direct_results_concat

# Gradio App

def flatten_list(nested_list):
  # Check if the list is nested
  if any(isinstance(i, list) for i in nested_list):
    # Flatten a nested list
    return [item for sublist in nested_list for item in sublist]
  else:
    return nested_list

def display_list(list_items):
    # Convert list to a string with each item on a new line
    list_string = "\n".join(list_items)
    return list_string

# In run_lp_model, use the state variables
def gr_run_lp_model(df_supply, supply_col_id, supply_col_val_list, df_demand, demand_n_matrix_col_id, demand_col_val, df_matrix):

    import ast
    # Convert string to list
    supply_col_val_list = ast.literal_eval(supply_col_val_list)

    # Run LP model
    mod_logs, models = run_lp_model(df_supply, supply_col_id, supply_col_val_list, df_demand, demand_n_matrix_col_id, demand_col_val, df_matrix)

    # Process model results
    results_logs, df_matrix_results, df_direct_results = process_model_results(models)

    # Flatten logs
    logs = []
    logs.append(flatten_list(mod_logs))
    logs.append(flatten_list(results_logs))
    logs = flatten_list(logs)

    # Save the processed CSV for download
    matrix_results_csv_path = '/matrix_results.csv'
    direct_results_csv_path = '/direct_results.csv'
    df_matrix_results.to_csv(matrix_results_csv_path, index=True)
    df_direct_results.to_csv(direct_results_csv_path, index=True)

    return display_list(logs), df_direct_results, matrix_results_csv_path, direct_results_csv_path

def gr_process_files(supply_file, demand_file, matrix_file, supply_col_id, supply_col_val_list):
  """
  Process the files
  """
  # The following variables are defined based on domain knowledge of the dataset.
  supply_col_id = 'Code' # This is the farm code from production.
  supply_col_val_list = ['Regular', 'Pre Weighted'] # Also the fruit type:  `Regular`, `Pre Pesado`, etc.

  demand_n_matrix_col_id = 'SKU' # This is the SKU from logistics.
  demand_col_val = 'Volume' # This is the volume requested per SKU.
  demand_n_matrix_col_id_category = 'Type' # This is the sub classification found on the `supply_col_val_list`.


  # Load the files
  df_supply, df_demand, df_matrix = load_dataset(supply_file.name, demand_file.name, matrix_file.name)
  df_demand = assign_demand_nodes_by_type(df_demand, df_matrix, demand_n_matrix_col_id, demand_n_matrix_col_id_category)

  # Filter fruit distribution by week and drop `Week` column
  # Normally, this should not be needed since the dataset is refreshed each week.
  WEEK = 'WK29'
  df_supply = df_supply[df_supply['Week'] == WEEK].drop(columns=['Week']).fillna(0)
  df_demand = df_demand[df_demand['Week'] == WEEK].drop(columns=['Week']).fillna(0)

  df_supply, supply_node_list, df_demand, demand_node_list, df_matrix, matrix_supply_node_list, matrix_demand_node_list = enforce_data_schema(df_supply, supply_col_id, supply_col_val_list, df_demand, demand_col_val, df_matrix, demand_n_matrix_col_id, demand_n_matrix_col_id_category)
  logs, df_supply, df_demand, df_matrix = check_matrix_completeness(df_supply, supply_node_list, df_demand, demand_node_list, df_matrix, matrix_supply_node_list, matrix_demand_node_list)

  return display_list(logs), supply_col_id,  supply_col_val_list, df_supply, demand_n_matrix_col_id, demand_col_val, demand_n_matrix_col_id_category, df_demand, df_matrix

# Gradio Interface
with gr.Blocks() as app:
    # Logo
    img_url="https://upload.wikimedia.org/wikipedia/en/thumb/d/d5/Dole_Foods_Logo_Green_Leaf.svg/1280px-Dole_Foods_Logo_Green_Leaf.svg.png"
    gr.Image(value=img_url, width=100, height=100)

    gr.Markdown("# Fruit Distribution App")

    ###############################################
    gr.Markdown("## File Procesor")
    # CSV File Uploads
    supply = gr.File(label="Upload Supply CSV File")
    demand = gr.File(label="Upload Demand CSV File")
    matrix = gr.File(label="Upload Matrix CSV File")

    # Process Button
    process_button = gr.Button("Process Files")

    # Outputs
    gr_logs = gr.Textbox(label="Data Processing Logs")
    gr_supply_col_id = gr.Textbox(label="Supply Column ID")
    gr_supply_col_val_list = gr.Textbox(label="Supply Column Value List")
    gr_df_supply = gr.Dataframe(label="Supply Dataframe")

    gr_demand_n_matrix_col_id = gr.Textbox(label="Demand N Matrix Column ID")
    gr_demand_col_val = gr.Textbox(label="Demand Column Value")
    gr_demand_n_matrix_col_id_category = gr.Textbox(label="Demand N Matrix Column ID Category")
    gr_df_demand = gr.Dataframe(label="Demand Dataframe")

    gr_df_matrix = gr.Dataframe(label="Matrix Dataframe")

    # Link function to button
    process_button.click(fn=gr_process_files,
                         inputs=[supply, demand, matrix],
                         outputs=[gr_logs,
                                  gr_supply_col_id, gr_supply_col_val_list, gr_df_supply,
                                  gr_demand_n_matrix_col_id, gr_demand_col_val, gr_demand_n_matrix_col_id_category, gr_df_demand,
                                  gr_df_matrix])


    ###############################################
    gr.Markdown("## Model Builder")
    # Run Model Button
    run_model_button = gr.Button("Run Model")

    # Outputs
    out_logs = gr.Textbox(label="Model Building Logs")
    out_df_direct_results = gr.Dataframe(label="Direct Results Dataframe")

    download_matrix_results = gr.File(label="Download Matrix Results")
    download_direct_results = gr.File(label="Download Direct Results")

    # Link function to button
    run_model_button.click(fn=gr_run_lp_model,
                           inputs=[gr_df_supply, gr_supply_col_id, gr_supply_col_val_list,
                                   gr_df_demand, gr_demand_n_matrix_col_id, gr_demand_col_val,
                                   gr_df_matrix],
                           outputs=[out_logs,
                                    out_df_direct_results,
                                    download_matrix_results, download_direct_results])

# Launch the app
app.launch(share=True)
