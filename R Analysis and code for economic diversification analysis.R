# Import and Prepare the Employment Dataset
# Objective: Load the raw industry employment dataset and prepare it for analysis.

# Remove unnecessary columns that are not needed for analysis (Area Bucket and Industry Bucket).

> install.packages("dplyr")
> library(dplyr)

> data_cleaned <- data %>%
  +     select(-`Area Bucket`, -`Industry Bucket`)
> print(data_cleaned)

# Filter out non-county data to ensure only geographically relevant areas are included.
#Remove rows where the county is not reported or labeled ambiguously.
#The code should filter rows that include the expression “county not reported”

> data_cleaned <- data_cleaned %>%
  +     filter(!grepl("county not reported", `Area Name`, ignore.case = TRUE))

# Integrate Commuting Zone Data
# Objective: Match counties to their corresponding commuting zones.

# Obtain a crosswalk dataset that maps counties to commuting zones.

commuting_zones <- read.csv("C:\Users\ViniciusBueno\OneDrive - Wyoming Business Council\Desktop\Economic Diversification\county20.csv")

# Change the nomenclature in the commuting_zones database to match the data_cleaned database
commuting_zones <- commuting_zones %>%
  rename(Area = GEOID)

# Merge the employment dataset with the commuting zone crosswalk using a common geographic identifier (e.g., county FIPS code).

merged_data <- data_cleaned %>%
  left_join(commuting_zones, by = "Area")


# Aggregate Employment Data by Industry and Commuting Zone
# Objective: Compute total employment per industry within each commuting zone.

# Convert job counts to numeric format to prevent errors in calculations.

# Extract the values from the list and convert to numeric
merged_data$Jobs <- as.numeric(merged_data$Jobs)

# Check the structure again to confirm the change
str(merged_data)
summary(merged_data$Jobs)

# Calculate total industry employment within each commuting zone.
# Group the dataset by commuting zone and industry.
# Compute the sum of jobs for each industry in a given commuting zone

# Calculate total employment for all industries within each commuting zone.
# Sum job counts across all industries in a commuting zone to obtain total workforce size.
total_employment_by_cz <- merged_data %>%
  group_by(CZ20) %>%
  summarize(TotalJobs = sum(Jobs, na.rm = TRUE))

# Calculate total industry employment within each commuting zone.
# Group the dataset by commuting zone and industry.
# Compute the sum of jobs for each industry in a given commuting zone.

total_employment_by_cz_industry <- merged_data %>%
  group_by(CZ20, `Industry Name`, Industry) %>%
  summarize(TotalJobs = sum(Jobs, na.rm = TRUE))

# Compute the industry's employment share within each commuting zone.
# Divide the industry-specific employment total by the overall commuting zone employment to calculate Employment Share.
# Rename the Total Jobs column in total_employment_by_cz to avoid conflict
total_employment_by_cz <- total_employment_by_cz %>%
  rename(`Total Jobs CZ` = `TotalJobs`)

# Combine the datasets
combined_data <- total_employment_by_cz_industry %>%
  left_join(total_employment_by_cz, by = "CZ20")

# Compute the employment share
combined_data <- combined_data %>%
  mutate(EmploymentShare = TotalJobs / `Total Jobs CZ`)

# Apply log transformations to adjust for scaling effects

# Compute the log-transformed total employment for each commuting zone
combined_data <- combined_data %>%
  mutate(LogTotalJobsCZ = log(`Total Jobs CZ`))

# Compute squared log-transformed total employment to capture potential nonlinear scaling patterns.
combined_data <- combined_data %>%
  mutate(SquaredLogTotalJobsCZ = LogTotalJobsCZ^2)

# Compute log-transformed total employment for each industry in each commuting zone.
combined_data <- combined_data %>%
  mutate(LogJobsbyindustryCZ = log(`TotalJobs`))

# Replace -Inf with NA
combined_data$LogJobsbyindustryCZ[is.infinite(combined_data$LogJobsbyindustryCZ)] <- NA

# Perform Pseudo-Poisson Maximum Likelihood (PPML) Regression
# Objective: Model the relationship between industry employment share and commuting zone employment.

# Select a target industry using its NAICS code.
# Filter the dataset to include only data for the selected industry.

filtered_data <- combined_data %>% filter(Industry == 513210)

# Define the regression model:
# Dependent Variable: Industry’s share of total employment in the commuting zone.
# Independent Variables:
# Log-transformed total employment.
# Squared log-transformed total employment.
# Run the Poisson regression model to estimate the scaling relationship.

regression_model <- lm(TotalJobs ~ LogTotalJobsCZ + SquaredLogTotalJobsCZ, data = filtered_data)

# View the summary of the regression model
summary(regression_model)

# Access ggplot2
library(ggplot2)

# Load the ggplot2 package
library(ggplot2)

#Visualize the Scaling Relationship
# Objective: Generate a plot to visualize employment scaling patterns.
# Create a scatterplot with a regression fit line and annotations

# Load the ggplot2 package
library(ggplot2)

# Create a scatterplot with an exponential regression fit line and annotations
ggplot(filtered_data, aes(x = LogTotalJobsCZ, y = `EmploymentShare`, label = Industry)) +
  geom_point() + 
  stat_smooth(method = "nls", formula = y ~ a * exp(b * x), 
              method.args = list(start = list(a = 1, b = 0.1)), col = "blue") + 
  labs(x = "Log-transformed total employment", y = "Industry employment share") +
  theme_minimal()

#  Obtain Scaling Coefficients for All Industries
# Objective: Automate the process to estimate scaling coefficients (β₁) for all industries in the dataset.

# Retrieve all unique NAICS codes from the dataset.
unique_naics_codes <- unique(combined_data$Industry)

# View the unique NAICS codes
print(unique_naics_codes)

