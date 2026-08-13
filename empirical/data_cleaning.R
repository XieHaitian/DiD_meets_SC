library(dplyr)
library(data.table)
library(haven)

control.states <- c(51, 33, 24, 26, 39, 49)
states <- c(2, control.states)  # Plus Alaska
N_G <- length(control.states)

# Load the data from the Stata file
df <- read_dta(paste0("march_regready_1996.dta")) %>% setDT()

# Collapsing the data
data <- df[(state_fips %in% states) & 
             (demgroup1 == 1) & 
             (year %in% 1998:2003) &
             (nchild <= 3),
           .(age = mean(age), 
             educ = pmin(sum(as.numeric(educ) > 2),2), 
             contpov = mean(contpov), 
             nchild = pmin(max(nchild),2)), 
           by = c('state_fips', 'year', 'hhseq')]

# Normalize age to the year 1998
data$age <- data$age - (data$year - 1998)

# Save the cleaned household-level repeated cross sections
write.csv(data, "Alaska_MW.csv")
