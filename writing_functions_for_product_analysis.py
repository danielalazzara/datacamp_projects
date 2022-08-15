# Import pandas with the usual alias
import pandas as pd

# Write a function that matches the docstring
def convert_csv_to_df(csv_name, source_type):
    """ Converts an NPS CSV into a DataFrame with a column for the source. 

    Args:
        csv_name (str): The name of the NPS CSV file.
        source_type (str): The source of the NPS responses.

    Returns:
        A DataFrame with the CSV data and a column, source.
    """   
    df = pd.read_csv(csv_name)
    df['source'] = source_type
    return df
    
    return df

# Test the function on the mobile data 
convert_csv_to_df("datasets/2020Q4_nps_mobile.csv", "mobile")

# Define a function check_csv which takes csv_name
def check_csv(csv_name):
    """ Checks if a CSV has three columns: response_date, user_id, nps_rating

    Args:
        csv_name (str): The name of the CSV file.

    Returns:
        Boolean: True if the CSV is valid, False otherwise 
    """
    # Open csv_name as f using with open
    with open(csv_name) as f:
        first_line = f.readline()
        # Return true if the CSV has the three specified columns
        if first_line == "response_date,user_id,nps_rating\n":
            return True
        # Otherwise, return false    
        else:
            return False

# Test the function on a corrupted NPS file
check_csv('datasets/corrupted.csv')

# Write a function combine_nps_csvs() with the arg csvs_dict 
def combine_nps_csvs(csvs_dict):
    # Define combine as an empty DataFrame
    combined = pd.DataFrame()
    # Iterate over csvs_dict to get the name and source of the CSVs
    for csv_name, source_type in csvs_dict.items():
        # Check if the csv is valid using check_csv()
        if check_csv(csv_name):
            # Convert the CSV using convert_csv_to_df() and assign it to temp
            temp = convert_csv_to_df(csv_name, source_type)
            # Concatenate combined and temp and assign it to combined
            combined = pd.concat([combined, temp])
        # If the file is not valid, print a message with the CSV's name
        else:
            print(csv_name + " is not a valid file and will not be added.")
    # Return the combined DataFrame
    return combined

my_files = {
  "datasets/2020Q4_nps_email.csv": "email",
  "datasets/2020Q4_nps_web.csv": "web",
  "datasets/2020Q4_nps_mobile.csv": "mobile",
  "datasets/corrupted.csv": "social_media"
}

# Test the function on the my_files dictionary
combine_nps_csvs(my_files)

def categorize_nps(x):
    """ 
    Takes a NPS rating and outputs whether it is a "promoter", 
    "passive", "detractor", or "invalid" rating. "invalid" is
    returned when the rating is not between 0-10.

    Args:
        x: The NPS rating

    Returns:
        String: the NPS category or "invalid".
    """
    # Write the rest of the function to match the docstring
    if x == 9 or x == 10:
        return 'promoter'
    elif x == 7 or x == 8:
        return 'passive'
    elif x >= 0 and x <= 6:
        return 'detractor'
    else: 
        return "invalid"

# Test the function 
categorize_nps(8)

def convert_csv_to_df(csv_name, source_type):    
    """ Convert an NPS CSV into a DataFrame with columns for the source and NPS group.

    Args:
        csv_name (str): The name of the NPS CSV file.
        source_type (str): The source of the NPS responses.

    Returns:
         A DataFrame with the CSV data and columns: source and nps_group.
    """
    df = pd.read_csv(csv_name)
    df['source'] = source_type
    # Define a new column nps_group which applies categorize_nps to nps_rating
    df['nps_group'] = df['nps_rating'].apply(categorize_nps)
    return df

# Test the updated function with mobile data
convert_csv_to_df("datasets/2020Q4_nps_mobile.csv", "mobile")

# Define a function calculate_nps that takes a DataFrame
def calculate_nps(nps_df):
    # Calculate the NPS score using the nps_group column 
    counts = nps_df['nps_group'].value_counts()
    detractor = counts['detractor']
    promotor = counts['promoter']
    total = counts.sum()
    # Return the NPS Score
    return ((promotor-detractor)/ total) * 100

my_files = {
  "datasets/2020Q4_nps_email.csv": "email",
  "datasets/2020Q4_nps_web.csv": "web",
  "datasets/2020Q4_nps_mobile.csv": "mobile",
}

# Test the function on the my_files dictionary
q4_nps = combine_nps_csvs(my_files)
calculate_nps(q4_nps)

# Define a function calculate_nps_by_source that takes a DataFrame
def calculate_nps_by_source(nps_df):
    # Group the DataFrame by source and apply calculate_nps()
    x = nps_df.groupby(['source']).apply(calculate_nps)
    # Return a Series with the NPS scores broken by source
    return x


my_files = {
  "datasets/2020Q4_nps_email.csv": "email",
  "datasets/2020Q4_nps_web.csv": "web",
  "datasets/2020Q4_nps_mobile.csv": "mobile",
}

# Test the function on the my_files dictionary
q4_nps = combine_nps_csvs(my_files)
calculate_nps_by_source(q4_nps)

# Copy and paste your code for the function from Task 3
def combine_nps_csvs(csvs_dict):
    """ Combine an NPS CSV into a DataFrame with a column for the source. 

    Args:
        csv_name (str): The name of the NPS CSV file.
        source_type (str): The source of the NPS responses.

    Returns:
        A DataFrame with the CSV data and a column, source.
    """ 
    # Define combine as an empty DataFrame
    combined = pd.DataFrame()
    # Iterate over csvs_dict to get the name and source of the CSVs
    for csv_name, source_type in csvs_dict.items():
        # Check if the csv is valid using check_csv()
        if check_csv(csv_name):
            # Convert the CSV using convert_csv_to_df() and assign it to temp
            temp = convert_csv_to_df(csv_name, source_type)
            # Concatenate combined and temp and assign it to combined
            combined = pd.concat([combined, temp])
        # If the file is not valid, print a message with the CSV's name
        else:
            print(csv_name + " is not a valid file and will not be added.")
    # Return the combined DataFrame
    return combined
    # Add docstring

# Copy and paste your code for the function from Task 6
def calculate_nps(nps_df):
    """ Calculate an NPS CSV into a DataFrame with a column for the source. 

    Args:
        csv_name (str): The name of the NPS CSV file.
        source_type (str): The source of the NPS responses.

    Returns:
        A DataFrame with the CSV data and a column, source.
    """ 
    # Calculate the NPS score using the nps_group column 
    counts = nps_df['nps_group'].value_counts()
    detractor = counts['detractor']
    promotor = counts['promoter']
    total = counts.sum()
    # Return the NPS Score
    return ((promotor-detractor)/ total) * 100

# Copy and paste your code for the function from Task 7
def calculate_nps_by_source(nps_df):
    """ Calculate an NPS CSV into a DataFrame with a column for the source. 

    Args:
        csv_name (str): The name of the NPS CSV file.
        source_type (str): The source of the NPS responses.

    Returns:
        A DataFrame with the CSV data and a column, source.
    """ 
    # Group the DataFrame by source and apply calculate_nps()
    x = nps_df.groupby(['source']).apply(calculate_nps)
    # Return a Series with the NPS scores broken by source
    return x

