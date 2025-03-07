# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:24:50 2025

@author: ConnorChristensen
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

#full lightcast data pull
file_path = "C:/Users/ConnorChristensen/OneDrive - Wyoming Business Council/Documents/Analysis/lc_data24.csv"  
df = pd.read_csv(file_path)

#does it look alright?
print(df.head())

#we can get rid of some of these columns
df = df.drop(['Area Bucket', 'Industry Bucket'], axis=1)

#we can drop non-county data
df = df[~df['Area Name'].str.contains("county not reported", case=False, na=False)]

#get that crosswalk in there
file_path_cw = "C:/Users/ConnorChristensen/OneDrive - Wyoming Business Council/Documents/GitHub/Economic-Complexity/county20.csv"  
cw = pd.read_csv(file_path_cw)

#merge it up
df = df.merge(cw, left_on="Area", right_on="GEOID", how="left")

#drop the GEOID, it's already represented with "Area"
df = df.drop(['GEOID'], axis=1)

# Ensure 'Jobs' column is numeric
df['Jobs'] = pd.to_numeric(df['Jobs'], errors='coerce')

# Compute total employment for each industry within each commuting zone
df['Total_Employment'] = df.groupby(['CZ20', 'Industry'])['Jobs'].transform('sum')

# Compute total employment in each commuting zone across all industries
df['Total_CZ_Employment'] = df.groupby('CZ20')['Jobs'].transform('sum')

# Calculate industry's share of total employment in the commuting zone
df['Employment_Share'] = df['Total_Employment'] / df['Total_CZ_Employment']

#collapsed data
df = df[['CZ20', 'Area', 'Area Name', 'Industry', 'Industry Name', 'Jobs', 'Total_CZ_Employment', 'Employment_Share']].drop_duplicates()

# Compute log transformations
df['Log_Total_CZ_Employment'] = np.log(df['Total_CZ_Employment'])
df['Log_Total_CZ_Employment_Squared'] = df['Log_Total_CZ_Employment'] ** 2

def run_ppml_te(df, naics_code):
    """
    Runs a Pseudo-Poisson Maximum Likelihood (PPML) regression for a specific industry
    identified by its 6-digit NAICS code across all commuting zones (CZ20) 
    and generates a single plot.
    
    Parameters:
    df (DataFrame): The dataset containing employment data.
    naics_code (int or str): The 6-digit NAICS code of the industry.
    """

    # Convert NAICS code to string (in case it's stored as int)
    naics_code = str(naics_code)

    # Retrieve the industry name corresponding to the NAICS code
    if naics_code not in df['Industry'].astype(str).values:
        print(f"No data available for NAICS code {naics_code}.")
        return

    industry_name = df[df['Industry'].astype(str) == naics_code]['Industry Name'].iloc[0]

    # Filter the dataset for the given NAICS industry
    df_industry = df[df['Industry'].astype(str) == naics_code].copy()

    # Drop missing or infinite values
    df_industry = df_industry.replace([np.inf, -np.inf], np.nan).dropna()

    # Define the regression formula
    formula = "Jobs ~ Log_Total_CZ_Employment + Log_Total_CZ_Employment_Squared"

    # Run PPML regression across all zones
    model = smf.glm(formula=formula, data=df_industry, family=sm.families.Poisson()).fit()

    # Print regression summary
    print(model.summary())

    # Plot Jobs vs Log Total Employment for all zones
    plt.figure(figsize=(8, 5))
    plt.scatter(df_industry['Log_Total_CZ_Employment'], np.log1p(df_industry['Jobs']), alpha=0.5, label='Data')

    # Generate predicted values
    df_industry['Predicted_Jobs'] = model.predict(df_industry)

    # Sort values for smooth line
    sorted_df = df_industry.sort_values('Log_Total_CZ_Employment')

    # Plot regression line
    plt.plot(sorted_df['Log_Total_CZ_Employment'], np.log(sorted_df['Predicted_Jobs']), color='red', label='PPML Fit')

    # Labels and title
    plt.xlabel("Log Total Employment in CZ")
    plt.ylabel("Log Employment in Industry in CZ")
    plt.title(f"PPML Regression for {industry_name} (NAICS {naics_code})")
    plt.legend()
    plt.grid(True)
    
    # Prevent y-axis from dropping below zero
    plt.ylim(bottom=0)
    
    # Show plot
    plt.show()
    
#total employment, crop production
run_ppml_te(df, naics_code=111000)
    
#total employment, software publishers
run_ppml_te(df, naics_code=513210)

#share of employment function
def run_ppml_se(df, naics_code):
    """
    Runs a Pseudo-Poisson Maximum Likelihood (PPML) regression for a specific industry
    identified by its 6-digit NAICS code across all commuting zones (CZ20) 
    and generates a single plot.
    
    Parameters:
    df (DataFrame): The dataset containing employment data.
    naics_code (int or str): The 6-digit NAICS code of the industry.
    """

    # Convert NAICS code to string (in case it's stored as int)
    naics_code = str(naics_code)

    # Retrieve the industry name corresponding to the NAICS code
    if naics_code not in df['Industry'].astype(str).values:
        print(f"No data available for NAICS code {naics_code}.")
        return

    industry_name = df[df['Industry'].astype(str) == naics_code]['Industry Name'].iloc[0]

    # Filter the dataset for the given NAICS industry
    df_industry = df[df['Industry'].astype(str) == naics_code].copy()

    # Drop missing or infinite values
    df_industry = df_industry.replace([np.inf, -np.inf], np.nan).dropna()

    # Define the regression formula
    formula = "Employment_Share ~ Log_Total_CZ_Employment + Log_Total_CZ_Employment_Squared"

    # Run PPML regression across all zones
    model = smf.glm(formula=formula, data=df_industry, family=sm.families.Poisson()).fit()

    # Print regression summary
    print(model.summary())

    # Plot Jobs vs Log Total Employment for all zones
    plt.figure(figsize=(8, 5))
    plt.scatter(np.log(df_industry['Total_CZ_Employment']), (df_industry['Employment_Share']), alpha=0.5, label='Data')

    # Generate predicted values
    df_industry['Predicted_Jobs_Sh'] = model.predict(df_industry)

    # Sort values for smooth line
    sorted_df = df_industry.sort_values('Log_Total_CZ_Employment')

    # Plot regression line
    plt.plot(sorted_df['Log_Total_CZ_Employment'], (sorted_df['Predicted_Jobs_Sh']), color='red', label='PPML Fit')

    # Labels and title
    plt.xlabel("Log Total Employment in CZ")
    plt.ylabel("Employment Share in Industry in CZ")
    plt.title(f"PPML Regression for {industry_name} (NAICS {naics_code})")
    plt.legend()
    plt.grid(True)

    
    # Show plot
    plt.show()

#share of employment, crop production
run_ppml_se(df, naics_code=111000)

#share of employment, software publishing
run_ppml_se(df, naics_code=513210)

#identify the desired commuting zones
def get_commuting_zone(df, county_name):
    """
    Returns the commuting zone (CZ20) for a given county name.
    
    Parameters:
    df (DataFrame): The dataset containing "Area Name" (county names) and "CZ20" (commuting zones).
    county_name (str): The name of the county to look up.
    
    Returns:
    array or None: The unique commuting zone(s) if found, otherwise None.
    """
    result = df[df["Area Name"].str.contains(county_name, case=False, na=False)]
    return result["CZ20"].unique() if not result.empty else None



county_name = "Larimer County, CO"                            #Fort Collins, CO
larimer = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {larimer}")

county_name = "Yellowstone County, MT"                            #Billings, MT
yellowstone = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {yellowstone}")

county_name = "Gallatin County, MT"                                #Bozeman, MT
gallatin = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {gallatin}")

county_name = "Pennington County, SD"                           #Rapid City, SD
pennington = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {pennington}")

county_name = "Scotts Bluff County, NE"                        #Scottsbluff, NE
scottsbluff = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {scottsbluff}")

county_name = "Bannock County, ID"                               #Pocatello, ID
bannock = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {bannock}")

county_name = "Cache County, UT"                                     #Logan, UT
cache = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {cache}")


# Combine all commuting zones into a set (removes duplicates)
target_czs = set()

for cz in [larimer, yellowstone, gallatin, pennington, scottsbluff, bannock, cache]:
    if cz is not None:
        if isinstance(cz, np.ndarray):  # Extract value if it's an array
            target_czs.update(cz.tolist())
        else:
            target_czs.add(cz)
            
# Convert to sorted list for readability
target_czs = sorted(target_czs)
print(f"Target Commuting Zones: {target_czs}")




# Filter dataset to only the selected commuting zones
df_filtered = df[df['CZ20'].isin(target_czs)]

# Check if filtering worked
print(df_filtered['CZ20'].unique()) 

print(df_filtered.shape)  # Should have at least some rows
print(df_filtered['Industry'].value_counts())

#sometimes this comes in handy, it's been hit or miss so far
df_filtered['Industry'] = df_filtered['Industry'].astype(str)

#regression time
scaling_coefficients = {}

for naics_code in df_filtered['Industry'].astype(str).unique():
    df_industry = df_filtered[df_filtered['Industry'] == naics_code].copy()
    
    print(f"Processing Industry: {naics_code}, Observations: {len(df_industry)}")  # Debugging print

    if len(df_industry) < 10:
        continue  # Skip industries with too few data points

    formula = "Jobs ~ Log_Total_CZ_Employment + Log_Total_CZ_Employment_Squared"

    try:
        model = smf.glm(formula=formula, data=df_industry, family=sm.families.Poisson()).fit()
        
        # Extract coefficients
        beta_0 = model.params.get("Intercept", np.nan)  
        beta_1 = model.params.get("Log_Total_CZ_Employment", np.nan)
        beta_2 = model.params.get("Log_Total_CZ_Employment_Squared", np.nan)  

        # Store all coefficients
        scaling_coefficients[naics_code] = (beta_0, beta_1, beta_2)

    except Exception as e:
        print(f"Error processing NAICS {naics_code}: {e}")

# Convert to DataFrame
df_scaling = pd.DataFrame.from_dict(scaling_coefficients, orient="index", 
                                    columns=["Intercept", "Scaling_Coefficient_Beta1", "Beta2"])
df_scaling.reset_index(inplace=True)
df_scaling.rename(columns={"index": "Industry"}, inplace=True)

#merge merge merge
df_filtered = df_filtered.merge(df_scaling, on="Industry", how="left")

# Compute expected employment using the scaling model
df_filtered['Expected_Jobs'] = np.exp(
    df_filtered['Intercept'] + 
    df_filtered['Scaling_Coefficient_Beta1'] * df_filtered['Log_Total_CZ_Employment'] + 
    df_filtered['Beta2'] * df_filtered['Log_Total_CZ_Employment_Squared']
)

# Compute employment deviation (actual - expected)
df_filtered['Employment_Deviation'] = df_filtered['Jobs'] - df_filtered['Expected_Jobs']

# Display the top industries with the highest deviations
df_filtered.sort_values(by="Employment_Deviation", ascending=False).head(10)


#it's a bit fooky, but I sort of start from scratch to run the other model of regression.
#It just ended up being easier that way, even though I know there's a better way to do it.

#full lightcast data pull
file_path = "C:/Users/ConnorChristensen/OneDrive - Wyoming Business Council/Documents/Analysis/lc_data24.csv"  
df = pd.read_csv(file_path)

#does it look alright?
print(df.head())

#we can get rid of some of these columns
df = df.drop(['Area Bucket', 'Industry Bucket'], axis=1)

#we can drop non-county data
df = df[~df['Area Name'].str.contains("county not reported", case=False, na=False)]

#get that crosswalk in there
file_path_cw = "C:/Users/ConnorChristensen/OneDrive - Wyoming Business Council/Documents/GitHub/Economic-Complexity/county20.csv"  
cw = pd.read_csv(file_path_cw)

#merge it up
df = df.merge(cw, left_on="Area", right_on="GEOID", how="left")

#drop the GEOID, it's already represented with "Area"
df = df.drop(['GEOID'], axis=1)

# Ensure 'Jobs' column is numeric
df['Jobs'] = pd.to_numeric(df['Jobs'], errors='coerce')

# Compute total employment for each industry within each commuting zone
df['Total_Employment'] = df.groupby(['CZ20', 'Industry'])['Jobs'].transform('sum')

# Compute total employment in each commuting zone across all industries
df['Total_CZ_Employment'] = df.groupby('CZ20')['Jobs'].transform('sum')

# Calculate industry's share of total employment in the commuting zone
df['Employment_Share'] = df['Total_Employment'] / df['Total_CZ_Employment']

#collapsed data
df = df[['CZ20', 'Area', 'Area Name', 'Industry', 'Industry Name', 'Jobs', 'Total_CZ_Employment', 'Employment_Share']].drop_duplicates()

# Compute log transformations
df['Log_Total_CZ_Employment'] = np.log(df['Total_CZ_Employment'])
df['Log_Total_CZ_Employment_Squared'] = df['Log_Total_CZ_Employment'] ** 2

def get_commuting_zone(df, county_name):
    """
    Returns the commuting zone (CZ20) for a given county name.
    
    Parameters:
    df (DataFrame): The dataset containing "Area Name" (county names) and "CZ20" (commuting zones).
    county_name (str): The name of the county to look up.
    
    Returns:
    array or None: The unique commuting zone(s) if found, otherwise None.
    """
    result = df[df["Area Name"].str.contains(county_name, case=False, na=False)]
    return result["CZ20"].unique() if not result.empty else None



county_name = "Larimer County, CO"                            #Fort Collins, CO
larimer = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {larimer}")

county_name = "Yellowstone County, MT"                            #Billings, MT
yellowstone = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {yellowstone}")

county_name = "Gallatin County, MT"                                #Bozeman, MT
gallatin = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {gallatin}")

county_name = "Pennington County, SD"                           #Rapid City, SD
pennington = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {pennington}")

county_name = "Scotts Bluff County, NE"                        #Scottsbluff, NE
scottsbluff = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {scottsbluff}")

county_name = "Bannock County, ID"                               #Pocatello, ID
bannock = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {bannock}")

county_name = "Cache County, UT"                                     #Logan, UT
cache = get_commuting_zone(df, county_name)
print(f"Commuting Zone for {county_name}: {cache}")

# Combine all commuting zones into a set (removes duplicates)
target_czs = set()

for cz in [larimer, yellowstone, gallatin, pennington, scottsbluff, bannock, cache]:
    if cz is not None:
        if isinstance(cz, np.ndarray):  # Extract value if it's an array
            target_czs.update(cz.tolist())
        else:
            target_czs.add(cz)
            
# Convert to sorted list for readability
target_czs = sorted(target_czs)
print(f"Target Commuting Zones: {target_czs}")




# Filter dataset to only the selected commuting zones
df_filtered = df[df['CZ20'].isin(target_czs)]

# Check if filtering worked
print(df_filtered['CZ20'].unique()) 

print(df_filtered.shape)  # Should have at least some rows
print(df_filtered['Industry'].value_counts())

df_filtered['Industry'] = df_filtered['Industry'].astype(str)

scaling_coefficients = {}

for naics_code in df_filtered['Industry'].astype(str).unique():
    df_industry = df_filtered[df_filtered['Industry'] == naics_code].copy()
    
    print(f"Processing Industry: {naics_code}, Observations: {len(df_industry)}")  # Debugging print

    if len(df_industry) < 10:
        continue  # Skip industries with too few data points

    formula = "Employment_Share ~ Log_Total_CZ_Employment + Log_Total_CZ_Employment_Squared"

    try:
        model = smf.glm(formula=formula, data=df_industry, family=sm.families.Poisson()).fit()
        
        # Extract coefficients
        beta_0 = model.params.get("Intercept", np.nan)  
        beta_1 = model.params.get("Log_Total_CZ_Employment", np.nan)
        beta_2 = model.params.get("Log_Total_CZ_Employment_Squared", np.nan)  

        # Store all coefficients
        scaling_coefficients[naics_code] = (beta_0, beta_1, beta_2)

    except Exception as e:
        print(f"Error processing NAICS {naics_code}: {e}")

# Convert to DataFrame
df_scaling = pd.DataFrame.from_dict(scaling_coefficients, orient="index", 
                                    columns=["Intercept", "Scaling_Coefficient_Beta1", "Beta2"])
df_scaling.reset_index(inplace=True)
df_scaling.rename(columns={"index": "Industry"}, inplace=True)


df_filtered_se = df_filtered.merge(df_scaling, on="Industry", how="left")

# Compute expected employment using the scaling model
df_filtered_se['Expected_Jobs_Share'] = np.exp(
    df_filtered_se['Intercept'] + 
    df_filtered_se['Scaling_Coefficient_Beta1'] * df_filtered_se['Log_Total_CZ_Employment'] + 
    df_filtered_se['Beta2'] * df_filtered_se['Log_Total_CZ_Employment_Squared']
)

# Compute employment deviation (actual - expected)
df_filtered_se['Deviation_Jobs_Share'] = df_filtered_se['Employment_Share'] - df_filtered_se['Expected_Jobs_Share']

# Display the top industries with the highest deviations
df_filtered_se.sort_values(by="Deviation_Jobs_Share", ascending=False).head(10)


df_top_deviation = (
    df_filtered_se.sort_values(by=["CZ20", "Deviation_Jobs_Share"], ascending=[True, False])
    .drop_duplicates(subset=["CZ20", "Deviation_Jobs_Share"])  # Remove duplicates within each CZ20
    .groupby("CZ20")
    .head(5)  # Keep only the top 5 per CZ20
)

df_top_deviation


#time for the good good print out (for now, it just feels good good at this moment)
# Define the target commuting zones for the five cities
target_czs = {larimer[0], yellowstone[0], gallatin[0], pennington[0], scottsbluff[0], bannock[0], cache[0]}

# Filter for only the target CZs
df_filtered_cities = df_filtered_se[df_filtered_se["CZ20"].isin(target_czs)].copy()

# Convert Deviation_Jobs_Share to percentage
df_filtered_cities["Deviation_Jobs_Share"] *= 100  

# Get the top 5 unique industries per city
df_top_cities = (
    df_filtered_cities.sort_values(by=["CZ20", "Deviation_Jobs_Share"], ascending=[True, False])
    .drop_duplicates(subset=["CZ20", "Deviation_Jobs_Share"])  # Remove duplicate values
    .groupby("CZ20")
    .head(5)  # Keep top 5 per city
)

# Create a mapping of CZ20 to City Names
city_mapping = {
    larimer[0]: "Fort Collins, CO",
    yellowstone[0]: "Billings, MT",
    gallatin[0]: "Bozeman, MT",
    pennington[0]: "Rapid City, SD",
    scottsbluff[0]: "Scottsbluff, NE",
    bannock[0]: "Pocatello, ID",
    cache[0]: "Logan, UT"
}

# Print the results
for cz, city in city_mapping.items():
    print(f"\nCity Name: {city}")
    df_city = df_top_cities[df_top_cities["CZ20"] == cz]
    for _, row in df_city.iterrows():
        print(f"  - {row['Industry Name']}: {row['Deviation_Jobs_Share']:.2f}%")
        
        
###############################################################################
'''
Next steps: 
    Obtain crosswalk for tradeable/non-tradable industries
    Filter for only tradable industries
    Apply normalization model to measures of deviation from expected value
        Ratio of actual share to expected share ie [(RCA - 1) / (RCA + 1)]
        Value will be 0 if exactly at expected value.
    Create tree
        Size of nodes based on deviation from 0
        Red, Yellow, Green color scheme
        May be sufficiently complex to rate a plotly interactive program
'''
###############################################################################
