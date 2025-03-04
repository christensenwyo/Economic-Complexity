# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:24:50 2025

@author: ConnorChristensen
"""

import pandas as pd

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
file_path_cw = "C:/Users/ConnorChristensen/OneDrive - Wyoming Business Council/Documents/Analysis/county20.csv"  
cw = pd.read_csv(file_path_cw)

#merge it up
df_merged = df.merge(cw, left_on="Area", right_on="GEOID", how="left")

#drop the GEOID, it's already represented with "Area"
df_merged = df_merged.drop(['GEOID'], axis=1)

###############################################################################

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Ensure 'Jobs' column is numeric
df_merged['Jobs'] = pd.to_numeric(df['Jobs'], errors='coerce')

# Compute total employment for each industry within each commuting zone
df_merged['Total_Employment'] = df_merged.groupby(['CZ20', 'Industry'])['Jobs'].transform('sum')

# Compute total employment in each commuting zone across all industries
df_merged['Total_CZ_Employment'] = df_merged.groupby('CZ20')['Jobs'].transform('sum')

# Calculate industry's share of total employment in the commuting zone
df_merged['Employment_Share'] = df_merged['Total_Employment'] / df_merged['Total_CZ_Employment']

#collapsed data
df_cz = df_merged[['CZ20', 'Industry', 'Industry Name', 'Total_CZ_Employment', 'Employment_Share']].drop_duplicates()

# Compute log transformations
df_cz['Log_Total_CZ_Employment'] = np.log(df_cz['Total_CZ_Employment'])
df_cz['Log_Total_CZ_Employment_Squared'] = df_cz['Log_Total_CZ_Employment'] ** 2



def run_ppml(df, naics_code):
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

    # Plot Employment Share vs Log Total Employment for all zones
    plt.figure(figsize=(8, 5))
    plt.scatter(df_industry['Log_Total_CZ_Employment'], df_industry['Employment_Share'], alpha=0.5, label='Data')

    # Generate predicted values
    df_industry['Predicted_Share'] = model.predict(df_industry)

    # Sort values for smooth line
    sorted_df = df_industry.sort_values('Log_Total_CZ_Employment')

    # Plot regression line
    plt.plot(sorted_df['Log_Total_CZ_Employment'], sorted_df['Predicted_Share'], color='red', label='PPML Fit')

    # Labels and title
    plt.xlabel("Log Total Employment")
    plt.ylabel("Employment Share")
    plt.title(f"PPML Regression for {industry_name} (NAICS {naics_code})")
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.show()

# Example usage:
run_ppml(df_cz, naics_code=334220)

###############################################################################

# Create an empty dictionary to store results
scaling_coefficients = {}

# Ensure 'Industry' in df_cz is a string for consistency
df_cz['Industry'] = df_cz['Industry'].astype(str)

# Get unique NAICS industry codes
naics_codes = df_cz['Industry'].unique()

# Loop through each industry and run the PPML regression
for naics_code in naics_codes:
    # Filter the dataset for the given industry
    df_industry = df_cz[df_cz['Industry'] == naics_code].copy()

    # Ensure enough data points for regression
    if len(df_industry) < 10:
        continue  # Skip industries with too few data points

    # Define the regression formula
    formula = "Employment_Share ~ Log_Total_CZ_Employment + Log_Total_CZ_Employment_Squared"

    try:
        # Run Poisson regression
        model = smf.glm(formula=formula, data=df_industry, family=sm.families.Poisson()).fit()

        # Extract the scaling coefficient (β₁)
        beta_1 = model.params.get("Log_Total_CZ_Employment", np.nan)

        # Store the coefficient in the dictionary
        scaling_coefficients[naics_code] = beta_1

    except Exception as e:
        print(f"Error processing NAICS {naics_code}: {e}")

# Convert dictionary to DataFrame
df_scaling = pd.DataFrame(list(scaling_coefficients.items()), columns=["Industry", "Scaling_Coefficient_Beta1"])

# Ensure 'Industry' in df_scaling is also a string
df_scaling['Industry'] = df_scaling['Industry'].astype(str)

# Merge scaling coefficients back into df_cz
df_cz = df_cz.merge(df_scaling, on="Industry", how="left")

# Display the first few rows of the updated DataFrame
print(df_cz.head())

###############################################################################
import seaborn as sns
# Plot histogram and density plot
plt.figure(figsize=(10, 6))

# Histogram
sns.histplot(df_cz["Scaling_Coefficient_Beta1"], bins=30, kde=True, color="blue", edgecolor="black")

# Labels and title
plt.xlabel("Scaling Coefficient (β₁)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Distribution of Industry Scaling Coefficients", fontsize=14)

# Show grid for better readability
plt.grid(True, linestyle="--", alpha=0.6)
###############################################################################

import statsmodels.discrete.count_model as cm

def run_ppml_with_zeros(df, naics_code, zero_handling="poisson"):
    """
    Runs a Pseudo-Poisson Maximum Likelihood (PPML) regression for a specific industry
    identified by its 6-digit NAICS code across all commuting zones (CZ20),
    handling zeros properly and generating a single plot.

    Parameters:
    df (DataFrame): The dataset containing employment data.
    naics_code (int or str): The 6-digit NAICS code of the industry.
    zero_handling (str): Method to handle zeros ("poisson" for standard Poisson regression, 
                         "zip" for Zero-Inflated Poisson).
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

    # Ensure zeros are retained
    df_industry['Employment_Share'] = df_industry['Employment_Share'].fillna(0)
    df_industry['Total_Employment'] = df_industry['Total_Employment'].fillna(0)

    # Adjust log transformation to handle zeros
    df_industry['Log_Total_Employment'] = np.log(df_industry['Total_Employment'] + 1)  # Avoid log(0)
    df_industry['Log_Total_Employment_Squared'] = df_industry['Log_Total_Employment'] ** 2

    # Drop infinite values (should be none due to log(0) fix)
    df_industry = df_industry.replace([np.inf, -np.inf], np.nan).dropna()

    # Define the regression formula
    formula = "Employment_Share ~ Log_Total_Employment + Log_Total_Employment_Squared"

    # Run either Poisson or Zero-Inflated Poisson regression
    if zero_handling == "zip":
        model = cm.ZeroInflatedPoisson.from_formula(formula, df_industry).fit()
    else:  # Default to Poisson
        model = smf.glm(formula=formula, data=df_industry, family=sm.families.Poisson()).fit()

    # Print regression summary
    print(model.summary())

    # Plot Employment Share vs Log Total Employment for all zones
    plt.figure(figsize=(8, 5))
    plt.scatter(df_industry['Log_Total_Employment'], df_industry['Employment_Share'], alpha=0.5, label='Data')

    # Generate predicted values
    df_industry['Predicted_Share'] = model.predict(df_industry)

    # Sort values for smooth line
    sorted_df = df_industry.sort_values('Log_Total_Employment')

    # Plot regression line
    plt.plot(sorted_df['Log_Total_Employment'], sorted_df['Predicted_Share'], color='red', label='PPML Fit')

    # Labels and title
    plt.xlabel("Log Total Employment")
    plt.ylabel("Employment Share")
    plt.title(f"PPML Regression for {industry_name} (NAICS {naics_code})")
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.show()

# Example usage:
run_ppml_with_zeros(df_cz, naics_code=541611, zero_handling="poisson")  # Try ZIP for zero handling



###############################################################################

def find_top_ppml_fits(df, top_n=5):
    """
    Runs PPML regression for all 6-digit NAICS codes in the dataset, 
    selects the top industries with the most "interesting" fits (β₁ closest to 1),
    and plots their scaling relationships.

    Parameters:
    df (DataFrame): The dataset containing employment data.
    top_n (int): Number of industries to plot (default is 5).
    """

    # Store results for all industries
    results = []

    # Get unique NAICS codes
    naics_codes = df['Industry'].astype(str).unique()

    for naics_code in naics_codes:
        # Get industry name
        industry_name = df[df['Industry'].astype(str) == naics_code]['Industry Name'].iloc[0]

        # Filter the dataset for the given NAICS industry
        df_industry = df[df['Industry'].astype(str) == naics_code].copy()

        # Drop missing or infinite values
        df_industry = df_industry.replace([np.inf, -np.inf], np.nan).dropna()

        # Ensure sufficient data points for regression
        if len(df_industry) < 10:  # Skip if too few observations
            continue

        # Define the regression formula
        formula = "Employment_Share ~ Log_Total_Employment + Log_Total_Employment_Squared"

        # Run PPML regression
        model = smf.glm(formula=formula, data=df_industry, family=sm.families.Poisson()).fit()

        # Get scaling exponent (β₁) from the first log term
        beta_1 = model.params.get("Log_Total_Employment", np.nan)

        # Store results
        if not np.isnan(beta_1):
            results.append((naics_code, industry_name, beta_1, model, df_industry))

    # Sort industries by β₁ closest to 1 (most "interesting" scaling)
    results_sorted = sorted(results, key=lambda x: abs(x[2] - 1))[:top_n]

    # Plot the selected top N industries
    fig, axes = plt.subplots(len(results_sorted), 1, figsize=(8, 5 * len(results_sorted)))

    if len(results_sorted) == 1:
        axes = [axes]  # Ensure axes is iterable for a single plot

    for i, (naics_code, industry_name, beta_1, model, df_industry) in enumerate(results_sorted):
        ax = axes[i]

        # Generate predicted values
        df_industry['Predicted_Share'] = model.predict(df_industry)

        # Sort values for smooth line
        sorted_df = df_industry.sort_values('Log_Total_Employment')

        # Scatter plot
        ax.scatter(df_industry['Log_Total_Employment'], df_industry['Employment_Share'], alpha=0.5, label='Data')

        # Regression line
        ax.plot(sorted_df['Log_Total_Employment'], sorted_df['Predicted_Share'], color='red', label='PPML Fit')

        # Labels and title
        ax.set_xlabel("Log Total Employment")
        ax.set_ylabel("Employment Share")
        ax.set_title(f"{industry_name} (NAICS {naics_code}) - β₁ = {beta_1:.3f}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage:
find_top_ppml_fits(df_merged, top_n=5)




'''
#################################################################################
def run_ppml(df, industry):
    """
    Runs a Pseudo-Poisson Maximum Likelihood (PPML) regression for a specific industry
    across all commuting zones (CZ20) and generates a single plot.
    
    Parameters:
    df (DataFrame): The dataset containing employment data.
    industry (str): The industry name to filter by.
    """

    # Filter the dataset for the given industry
    df_industry = df_merged[df_merged['Industry Name'] == industry].copy()

    # Check if there is data
    if df_industry.empty:
        print(f"No data available for {industry}.")
        return

    # Drop missing or infinite values
    df_industry = df_industry.replace([np.inf, -np.inf], np.nan).dropna()

    # Define the regression formula
    formula = "Employment_Share ~ Log_Total_Employment + Log_Total_Employment_Squared"

    # Run PPML regression across all zones
    model = smf.glm(formula=formula, data=df_industry, family=sm.families.Poisson()).fit()

    # Print regression summary
    print(model.summary())

    # Plot Employment Share vs Log Total Employment for all zones
    plt.figure(figsize=(8, 5))
    plt.scatter(df_industry['Log_Total_Employment'], df_industry['Employment_Share'], alpha=0.5, label='Data')

    # Generate predicted values
    df_industry['Predicted_Share'] = model.predict(df_industry)

    # Sort values for smooth line
    sorted_df = df_industry.sort_values('Log_Total_Employment')

    # Plot regression line
    plt.plot(sorted_df['Log_Total_Employment'], sorted_df['Predicted_Share'], color='red', label='PPML Fit')

    # Labels and title
    plt.xlabel("Log Total Employment")
    plt.ylabel("Employment Share")
    plt.title(f"PPML Regression for {industry} (All CZs)")
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.show()


run_ppml(df_merged, industry="Research and Development in Nanotechnology")


'''







###################################################################################

'''
def run_ppml_ns(df, industry):
    """
    Runs a Pseudo-Poisson Maximum Likelihood (PPML) regression for a specific industry
    across all commuting zones (CZ20) and generates a single plot.
    
    Parameters:
    df (DataFrame): The dataset containing employment data.
    industry (str): The industry name to filter by.
    """

    # Filter the dataset for the given industry
    df_industry = df_merged[df_merged['Industry Name'] == industry].copy()

    # Check if there is data
    if df_industry.empty:
        print(f"No data available for {industry}.")
        return

    # Drop missing or infinite values
    df_industry = df_industry.replace([np.inf, -np.inf], np.nan).dropna()

    # Define the regression formula
    formula = "Employment_Share ~ Log_Total_Employment"

    # Run PPML regression across all zones
    model = smf.glm(formula=formula, data=df_industry, family=sm.families.Poisson()).fit()

    # Print regression summary
    print(model.summary())

    # Plot Employment Share vs Log Total Employment for all zones
    plt.figure(figsize=(8, 5))
    plt.scatter(df_industry['Log_Total_Employment'], df_industry['Employment_Share'], alpha=0.5, label='Data')

    # Generate predicted values
    df_industry['Predicted_Share'] = model.predict(df_industry)

    # Sort values for smooth line
    sorted_df = df_industry.sort_values('Log_Total_Employment')

    # Plot regression line
    plt.plot(sorted_df['Log_Total_Employment'], sorted_df['Predicted_Share'], color='red', label='PPML Fit')

    # Labels and title
    plt.xlabel("Log Total Employment")
    plt.ylabel("Employment Share")
    plt.title(f"PPML Regression for {industry} (All CZs)")
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.show()


run_ppml_ns(df_merged, industry="Testing Laboratories")


'''






###############################################################################
'''

def estimate_scaling_exponent(df, industry):
    """
    Estimates the power-law scaling relationship for a given industry using OLS on log-transformed data.
    
    Model: log(Industry Employment) = β0 + β1 * log(Total Employment) + ε
    
    Parameters:
    df (DataFrame): The dataset containing employment data.
    industry (str): The industry name to filter by.
    
    Returns:
    Prints the regression summary and plots the scaling relationship.
    """

    # Filter the dataset for the specified industry
    df_industry = df[df['Industry Name'] == industry].copy()

    # Check if data is available
    if df_industry.empty:
        print(f"No data available for {industry}.")
        return

    # Drop zero employment values to avoid log(0) issues
    df_industry['Jobs'] = df_industry['Jobs'].replace(0, 1)

    # Apply log transformation
    df_industry['Log_Industry_Employment'] = np.log(df_industry['Jobs'])
    df_industry['Log_Total_Employment'] = np.log(df_industry['Total_Employment'])

    # Define independent (X) and dependent (y) variables
    X = sm.add_constant(df_industry['Log_Total_Employment'])  # Add intercept term
    y = df_industry['Log_Industry_Employment']

    # Run OLS regression
    model = sm.OLS(y, X).fit()

    # Print the regression results
    print(f"\nScaling Regression for Industry: {industry}")
    print(model.summary())

    # Extract β₁ to interpret scaling behavior
    beta_1 = model.params['Log_Total_Employment']
    
    if beta_1 > 1:
        scaling_type = "Superlinear Scaling (β₁ > 1)"
    elif beta_1 < 1:
        scaling_type = "Sublinear Scaling (β₁ < 1)"
    else:
        scaling_type = "Linear Scaling (β₁ ≈ 1)"
    
    print(f"\nEstimated Scaling Exponent (β₁) = {beta_1:.3f} → {scaling_type}")

    # Plot the scaling relationship
    plt.figure(figsize=(8, 5))
    plt.scatter(df_industry['Log_Total_Employment'], df_industry['Log_Industry_Employment'], alpha=0.5, label='Data')
    
    # Generate predicted values for visualization
    df_industry['Predicted_Employment'] = model.predict(X)

    # Plot the regression line
    plt.plot(df_industry['Log_Total_Employment'], df_industry['Predicted_Employment'], color='red', label='OLS Fit')

    # Labels and title
    plt.xlabel("Log Total Employment")
    plt.ylabel("Log Industry Employment")
    plt.title(f"Scaling Relationship for {industry}")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

# Example usage:
estimate_scaling_exponent(df_merged, industry="Testing Laboratories")

'''








'''

def run_ppml_and_plot(df, industry, cz20):
    """
    Runs a Pseudo-Poisson Maximum Likelihood (PPML) regression for a specific industry 
    in a given commuting zone (CZ20) and generates a plot.
    
    Parameters:
    df (DataFrame): The dataset containing employment data.
    industry (str): The industry name to filter by.
    cz20 (float/int): The commuting zone ID to filter by.
    """
    
    # Filter for the specific industry and commuting zone
    df_subset = df_merged[(df['Industry Name'] == industry) & (df_merged['CZ20'] == cz20)].copy()
    
    if df_subset.empty:
        print(f"No data available for {industry} in commuting zone {cz20}")
        return

    # Drop missing or infinite values
    df_subset = df_subset.replace([np.inf, -np.inf], np.nan).dropna()

    # Define the regression formula
    formula = "Employment_Share ~ Log_Total_Employment + Log_Total_Employment_Squared"

    # Run PPML regression
    model = smf.glm(formula=formula, data=df_subset, family=sm.families.Poisson()).fit()
    
    # Print regression summary
    print(model.summary())

    # Plot Employment Share vs Log Total Employment
    plt.figure(figsize=(8, 5))
    plt.scatter(df_subset['Log_Total_Employment'], df_subset['Employment_Share'], alpha=0.5, label='Data')
    
    # Generate predicted values
    df_subset['Predicted_Share'] = model.predict(df_subset)
    
    # Plot the regression line
    sorted_df = df_subset.sort_values('Log_Total_Employment')
    plt.plot(sorted_df['Log_Total_Employment'], sorted_df['Predicted_Share'], color='red', label='PPML Fit')

    # Labels and title
    plt.xlabel("Log Total Employment")
    plt.ylabel("Employment Share")
    plt.title(f"PPML Regression: {industry} in CZ {cz20}")
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.show()

run_ppml_and_plot(df_merged, industry="Commercial and Institutional Building Construction", cz20=152)


'''
