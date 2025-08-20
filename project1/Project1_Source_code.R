
#Data Cleaning and Feature Creation (R)
#Load necessary library (assuming they are all installed already)
library(dplyr)        # data wrangling
library(lubridate)    # dates & time features
library(skimr)        # quick data summaries
library(rsample)      # train/test splits (incl. time-based)
library(MLmetrics)    # metrics
library(rpart)        # decision tree
library(rpart.plot)   # tree visualization
library(ggplot2)	   # data visualization
library(DBI)
library(RSQLite)

# Connect to the SQLite database
con <- dbConnect(SQLite(), "C:/Users/OLUCHI/OneDrive/Desktop/Codecademy/ITCS-DS-PROJECT1-DIST-V5/project1_raw_data.db") #replace with your file address

# Recreate the join query
query <- "
SELECT 
    t.date,
    t.store_nbr,
    t.item_nbr,
    t.units,
    w.station_nbr,
    w.tmax,
    w.tmin,
    w.tavg,
    w.depart,
    w.dewpoint,
    w.wetbulb,
    w.heat,
    w.cool,
    w.sunrise,
    w.sunset,
    w.codesum,
    w.snowfall,
    w.preciptotal,
    w.stnpressure,
    w.sealevel,
    w.resultspeed,
    w.resultdir,
    w.avgspeed
FROM train t
JOIN key k ON t.store_nbr = k.store_nbr
JOIN weather w ON k.station_nbr = w.station_nbr AND t.date = w.date;
"

# Execute the query and load into R as a dataframe
joined_data <- dbGetQuery(con, query)

# Preview
head(joined_data)

# Disconnect
dbDisconnect(con)

#data summary
skim(joined_data)

# Dates -> Date format
joined_data <- joined_data %>%
  mutate(
    date = as.Date(date)  
  )


# Handle special characters & missing weather values 
# preciptotal sometimes arrives as character; "T" means trace precipitation (tiny > 0)
# Strategy:
# - Replace "" with NA
# - Replace "T" with a small positive value (e.g., 0.002 inches) to reflect trace, not 0
# - Cast to numeric
# - Impute remaining NAs with station- & month-wise median (seasonal + local), fallback to global median

## Handle missing weather values

joined_data <- joined_data %>%
  mutate(
    # normalize preciptotal
    preciptotal = as.character(preciptotal), # Ensure character for cleaning
    preciptotal = na_if(preciptotal, ""),             # "" -> NA
    preciptotal = ifelse(preciptotal == "T", "0.002", preciptotal),
    preciptotal = as.numeric(preciptotal),                # Convert to numeric
    year  = year(date),
    month = month(date)
  )

# Impute preciptotal by station-month median, fallback to global median
precip_global_med <- median(joined_data$preciptotal, na.rm = TRUE)
joined_data <- joined_data %>%
  group_by(station_nbr, month) %>%
  mutate(
    precip_med_sm = median(preciptotal, na.rm = TRUE),
    preciptotal   = ifelse(is.na(preciptotal),
                           ifelse(is.finite(precip_med_sm), precip_med_sm, precip_global_med),
                           preciptotal)
  ) %>%
  ungroup() %>%
  select(-precip_med_sm)

# tavg: ensure numeric, impute station-month median (fallback to global median)
joined_data <- joined_data %>%
  mutate(tavg = suppressWarnings(as.numeric(tavg)))
tavg_global_med <- median(joined_data$tavg, na.rm = TRUE)
joined_data <- joined_data %>%
  group_by(station_nbr, month) %>%
  mutate(
    tavg_med_sm = median(tavg, na.rm = TRUE),
    tavg        = ifelse(is.na(tavg),
                         ifelse(is.finite(tavg_med_sm), tavg_med_sm, tavg_global_med),
                         tavg)
  ) %>%
  ungroup() %>%
  select(-tavg_med_sm)



# ----  Outliers in ‘units’ (zero-inflated) ----
# Rather than IQR-only, I use percentile capping  at the 99th percentile per product.
# This curbs extreme spikes (maybe due to promos/errors) without dropping rows.
units_caps <- joined_data %>%
  group_by(item_nbr) %>%
  summarize(upper_99 = quantile(units, 0.99, na.rm = TRUE), .groups = "drop")

joined_data <- joined_data %>%
  left_join(units_caps, by = "item_nbr") %>%
  mutate(
    units_capped = pmin(units, upper_99),
    units_was_capped = as.integer(units > upper_99)
  ) %>%
  select(-upper_99)

#  Feature Engineering
joined_data <- joined_data %>%
  mutate(
    # Weekend flag
    is_weekend = wday(date) %in% c(1, 7), # Sunday=1, Saturday=7
    
    # Rainy day flag
    is_rainy_day = grepl("ra|dz|sh", codesum, ignore.case = TRUE),
    
    # Snow/Ice day flag
    is_snowy_day = grepl("sn|sg|gs|pl|ic", codesum, ignore.case = TRUE),
    
    # Stormy day flag (high impact weather)
    is_stormy_day = grepl("\\+fc|fc|ts|gr|sq", codesum, ignore.case = TRUE)
  )

# Why did I create these features?
# is_weekend - consumer shopping patterns differ on weekends
# is_rainy_day -  bad weather may deter in-store sales but boost online orders
# is_snowy_day - extreme cold might reduce mobility, affecting sales
# is_stormy_day - severe weather can disrupt both supply & demand



# Predictive Model(s) (R):

# Split Data
set.seed(123) # Set seed for reproducibility so results are consistent every time the code runs
data_split <- initial_split(joined_data, prop = 0.8) # Split the dataset into training (80%) and testing (20%) sets
train_data <- training(data_split) # Extract the training portion of the data (used to build the model)
test_data  <- testing(data_split) # Extract the testing portion of the data (used to evaluate model performance)



# Get the top 3 products by total units sold
top_products <- train_data %>%
  group_by(item_nbr) %>%
  summarise(total_units = sum(units, na.rm = TRUE)) %>%
  arrange(desc(total_units)) %>%
  slice_head(n = 3) %>%
  pull(item_nbr)


#Let's filter training data for only the top 3 products
train_top <- train_data %>%
  filter(item_nbr %in% top_products)


#MODEL COMPARISON BY PRODUCT
#Let's compare the models, to capture which model is most appropriate and why, for each product
#split 
results <- data.frame(
  item_nbr = character(),
  model = character(),
  R2 = numeric(),
  RMSE = numeric(),
  MAE = numeric(),
  stringsAsFactors = FALSE
)

for (pid in top_products) {
  # filter product data
  train_p <- train_set %>% filter(item_nbr == pid)
  test_p  <- test_set %>% filter(item_nbr == pid)
  
  set.seed(123)
  train_idx <- sample(seq_len(nrow(train_top)), size = 0.8 * nrow(train_top))
  train_set <- train_top[train_idx, ]
  test_set  <- train_top[-train_idx, ]
  
  #train models for each product
  
  
  # Linear Regression
  lm_model <- lm(units ~ tavg + preciptotal + is_weekend + is_rainy_day + is_snowy_day + is_stormy_day,
                 data = train_p)
  preds_lm <- predict(lm_model, newdata = test_p)
  
  # Decision Tree
  tree_model <- rpart(units ~ tavg + preciptotal + is_weekend + is_rainy_day + is_snowy_day + is_stormy_day,
                      data = train_p,
                      method = "anova",
                      control = rpart.control(cp = 0.0001, minsplit = 20))
  preds_tree <- predict(tree_model, newdata = test_p)
  
  # Metrics for Linear Regression
  R2_lm <- 1 - sum((test_p$units - preds_lm)^2) / sum((test_p$units - mean(test_p$units))^2)
  RMSE_lm <- RMSE(preds_lm, test_p$units)
  MAE_lm  <- MAE(preds_lm, test_p$units)
  
  results <- rbind(results, data.frame(
    item_nbr = pid, model = "Linear Regression",
    R2 = R2_lm, RMSE = RMSE_lm, MAE = MAE_lm
  ))
  
  # Metrics for Tree
  R2_tree <- 1 - sum((test_p$units - preds_tree)^2) / sum((test_p$units - mean(test_p$units))^2)
  RMSE_tree <- RMSE(preds_tree, test_p$units)
  MAE_tree  <- MAE(preds_tree, test_p$units)
  
  results <- rbind(results, data.frame(
    item_nbr = pid, model = "Decision Tree",
    R2 = R2_tree, RMSE = RMSE_tree, MAE = MAE_tree
  ))
}





#VARIABLES IMPACT BY PRODUCT
#Lets now focus on summarizing how each variable impact sales for each product
# Starting with results dataframe for storing variable impacts
impact_results <- data.frame(
  item_nbr = character(),
  model = character(),
  variable = character(),
  importance = numeric(),
  stringsAsFactors = FALSE
)

for (pid in top_products) {
  # filter product data
  train_p <- train_set %>% filter(item_nbr == pid)
  
  # ---------------- Linear Regression ----------------
  lm_model <- lm(units ~ tavg + preciptotal + is_weekend + is_rainy_day + is_snowy_day + is_stormy_day,
                 data = train_p)
  
  coefs <- summary(lm_model)$coefficients
  for (var in rownames(coefs)) {
    impact_results <- rbind(impact_results, data.frame(
      item_nbr = pid,
      model = "Linear Regression",
      variable = var,
      importance = coefs[var, "Estimate"]
    ))
  }
  
  # ---------------- Decision Tree ----------------
  tree_model <- rpart(units ~ tavg + preciptotal + is_weekend + is_rainy_day + is_snowy_day + is_stormy_day,
                      data = train_p,
                      method = "anova",
                      control = rpart.control(cp = 0.0001, minsplit = 20))
  
  var_imp <- tree_model$variable.importance
  for (var in names(var_imp)) {
    impact_results <- rbind(impact_results, data.frame(
      item_nbr = pid,
      model = "Decision Tree",
      variable = var,
      importance = var_imp[var]
    ))
  }
}

# Clean final table: sort by product, model, and importance (descending for tree)
impact_summary <- impact_results %>%
  group_by(item_nbr, model) %>%
  arrange(item_nbr, model, desc(abs(importance)), .by_group = TRUE)




# Model Outputs (R)
#show results of comparison of models
print(results)

#show results of variables’ impact on sales for each product
print(impact_summary, n = Inf)




# Visualizations (R)
# 1. Scatter plot: Temperature vs Sales for each product 
ggplot(train_set %>% filter(item_nbr %in% c(5, 9, 45)),
       aes(x = tavg, y = units, color = factor(item_nbr))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Temperature vs Sales by Product",
       x = "Average Temperature (°F)",
       y = "Units Sold",
       color = "Product") +
  coord_cartesian(ylim = c(0, 500)) +  
  theme_minimal()




# 2. Bar chart: Average sales on weekends vs weekdays
ggplot(train_set %>% filter(item_nbr %in% c(5,9,45)) %>%
         group_by(item_nbr, is_weekend) %>%
         summarise(avg_units = mean(units), .groups = "drop"),
       aes(x = factor(is_weekend), y = avg_units, fill = factor(item_nbr))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Average Sales: Weekends vs Weekdays",
       x = "Weekend (TRUE/FALSE)",
       y = "Average Units Sold",
       fill = "Product") +
  coord_cartesian(ylim = c(0, 50)) +  
  theme_minimal()



# 3. Scatter plot: Precipitation vs Sales
ggplot(train_data %>% filter(item_nbr %in% c(5,9,45)),
       aes(x = preciptotal, y = units, color = factor(item_nbr))) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Precipitation vs Sales by Product",
       x = "Total Precipitation (inches)",
       y = "Units Sold",
       color = "Product") +
  coord_cartesian(ylim = c(0, 500)) +
  theme_minimal()
