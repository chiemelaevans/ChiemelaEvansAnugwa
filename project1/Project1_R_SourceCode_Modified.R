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
con <- dbConnect(SQLite(), "C:/Users/OLUCHI/OneDrive/Desktop/Codecademy/ITCS-DS-PROJECT1-DIST-V5/project1_raw_data.db") # connect to the DB 
dbListTables(con) # list database tables 


# 4.1.1. The top 3 products with the highest total sales

top3_products <- dbGetQuery(con, "SELECT 
    item_nbr,
    SUM(units) AS total_units_sold
FROM train
GROUP BY item_nbr
ORDER BY total_units_sold DESC
LIMIT 3
")


# 4.1.2. Query to join the  Tables

joined_table <- dbGetQuery(con, "SELECT 
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
JOIN weather w ON k.station_nbr = w.station_nbr AND t.date = w.date")



# 4.1.3. Daily sales and average temperature (tavg) for one of the top 3 products

daily_sales_and_tavg <- dbGetQuery(con, "SELECT 
    t.date,
    t.item_nbr,
    SUM(t.units) AS daily_units_sold,
    w.tavg
FROM train AS t
JOIN key AS k
    ON t.store_nbr = k.store_nbr
JOIN weather AS w
    ON k.station_nbr = w.station_nbr
   AND t.date = w.date
WHERE t.item_nbr = (
    SELECT item_nbr
    FROM train
    GROUP BY item_nbr
    ORDER BY SUM(units) DESC
    LIMIT 1   -- OFFSET 1 or 2 for 2nd/3rd top product
)
GROUP BY t.date, t.item_nbr, w.tavg
ORDER BY t.date")


# preview the results of sql codes above 
head(joined_table)
top3_products
daily_sales_and_tavg

# joined data summary
skim(joined_table) #this clearly shows that the date column is in character format


#4.2.1.  Load the joined dataset into R and d convert date column to Date format 
# Note that the joined dataset is called joined_table above as a dataframe in R, so we go ahead and convert the date column to Date format

joined_table <- joined_table %>%
  mutate(
    date = as.Date(date)  
  )


#4.2.2. Data Cleaning

# Handle special characters & missing weather values 
# preciptotal sometimes arrives as character; "T" means trace precipitation (tiny > 0)
# Strategy:
# - Replace "" with NA
# - Replace "T" with a small positive value (e.g., 0.002 inches) to reflect trace, not 0
# - Cast to numeric
# - Impute remaining NAs with station- & month-wise median (seasonal + local), fallback to global median

## Handle missing weather values

joined_table <- joined_table %>%
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
joined_table <- joined_table %>%
  mutate(tavg = suppressWarnings(as.numeric(tavg)))
tavg_global_med <- median(joined_data$tavg, na.rm = TRUE)
joined_table <- joined_table %>%
  group_by(station_nbr, month) %>%
  mutate(
    tavg_med_sm = median(tavg, na.rm = TRUE),
    tavg        = ifelse(is.na(tavg),
                         ifelse(is.finite(tavg_med_sm), tavg_med_sm, tavg_global_med),
                         tavg)
  ) %>%
  ungroup() %>%
  select(-tavg_med_sm)



# ----  Handling Outliers in ‘units’ (zero-inflated) ----
# Rather than IQR-only (because there are many zeros in units), I use percentile capping  at the 99th percentile per product.
# This curbs extreme spikes (maybe due to promos/errors) without dropping rows.
# This method is used since it is robust against heavy skewness, and the problem with zero-inflated data is usually extreme positive values, not the zeros themselves

units_caps <- joined_table %>%
  group_by(item_nbr) %>%
  summarize(upper_99 = quantile(units, 0.99, na.rm = TRUE), .groups = "drop")

joined_table <- joined_table %>%
  left_join(units_caps, by = "item_nbr") %>%
  mutate(
    units_capped = pmin(units, upper_99),
    units_was_capped = as.integer(units > upper_99)
  ) %>%
  select(-upper_99)




# 4.2.3. Creating new features needed for modelling
# Let's create four new features: is_weekend, is_rainy_day, is_snowy_day and is_stormy_day

joined_table <- joined_table %>%
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


#preview the new table to see if it includes the four new features created
head(joined_table)



# 4.2.4 Predictive Model(s) (R):

# Predictive Model(s) (R):

# Split Data
set.seed(123) # Set seed for reproducibility so results are consistent every time the code runs
data_split <- initial_split(joined_table, prop = 0.8) # Split the dataset into training (80%) and testing (20%) sets
train_data <- training(data_split) # Extract the training portion of the data (used to build the model)
test_data  <- testing(data_split) # Extract the testing portion of the data (used to evaluate model performance)



# Use the top 3 products by total units sold, found using sql code
top3_products <- top3_products %>%
  group_by(item_nbr) %>%
  arrange(desc(total_units_sold)) %>%
  pull(item_nbr)

top3_products


# Filter training and testing data for top 3 products
train_top <- train_data %>% filter(item_nbr %in% top3_products)
test_top  <- test_data %>% filter(item_nbr %in% top3_products)

# Prepare results dataframe
results <- data.frame(
  item_nbr = character(),
  model = character(),
  R2 = numeric(),
  RMSE = numeric(),
  MAE = numeric(),
  stringsAsFactors = FALSE
)

# Loop over each product
for (pid in top3_products) {
  # Filter data for this product
  train_p <- train_top %>% filter(item_nbr == pid)
  test_p  <- test_top %>% filter(item_nbr == pid)
  
  # Skip if not enough training/test data
  if (nrow(train_p) < 10 || nrow(test_p) < 5) next
  
  # Remove rows with NA in key variables
  train_p <- train_p[complete.cases(train_p[c("units", "tavg", "preciptotal", 
                                              "is_weekend", "is_rainy_day", 
                                              "is_snowy_day", "is_stormy_day")]), ]
  test_p  <- test_p[complete.cases(test_p[c("units", "tavg", "preciptotal", 
                                            "is_weekend", "is_rainy_day", 
                                            "is_snowy_day", "is_stormy_day")]), ]
  
  # Skip if not enough clean data
  if (nrow(train_p) < 10 || nrow(test_p) < 5) next
  
  # Only compute R2 safely
  safe_R2 <- function(pred, obs) {
    ss_res <- sum((obs - pred)^2, na.rm = TRUE)
    ss_tot <- sum((obs - mean(obs, na.rm = TRUE))^2, na.rm = TRUE)
    if (ss_tot == 0) return(1)  # Perfect prediction if obs is constant and pred matches
    if (ss_tot == 0 || is.na(ss_res) || is.na(ss_tot)) return(NA_real_)
    1 - ss_res / ss_tot
  }
  
  # --- Linear Regression ---
  lm_model <- try(lm(units ~ tavg + preciptotal + is_weekend + 
                       is_rainy_day + is_snowy_day + is_stormy_day,
                     data = train_p), silent = TRUE)
  
  if (!inherits(lm_model, "try-error") && !any(is.na(coef(lm_model)))) {
    preds_lm <- predict(lm_model, newdata = test_p)
    
    R2_lm <- safe_R2(preds_lm, test_p$units)
    RMSE_lm <- RMSE(preds_lm, test_p$units)
    MAE_lm <- MAE(preds_lm, test_p$units)
    
    results <- rbind(results, data.frame(
      item_nbr = pid,
      model = "Linear Regression",
      R2 = R2_lm,
      RMSE = RMSE_lm,
      MAE = MAE_lm
    ))
  }
  
  # --- Decision Tree ---
  tree_model <- try(rpart(units ~ tavg + preciptotal + is_weekend + 
                            is_rainy_day + is_snowy_day + is_stormy_day,
                          data = train_p, method = "anova",
                          control = rpart.control(cp = 0.0001, minsplit = 20)), silent = TRUE)
  
  if (!inherits(tree_model, "try-error")) {
    preds_tree <- predict(tree_model, newdata = test_p)
    
    R2_tree <- safe_R2(preds_tree, test_p$units)
    RMSE_tree <- RMSE(preds_tree, test_p$units)
    MAE_tree <- MAE(preds_tree, test_p$units)
    
    results <- rbind(results, data.frame(
      item_nbr = pid,
      model = "Decision Tree",
      R2 = R2_tree,
      RMSE = RMSE_tree,
      MAE = MAE_tree
    ))
  }
}

#show results of comparison of models
print(results)




# VARIABLES IMPACT BY PRODUCT
# Now analyze how each variable impacts sales for each top product

# Results dataframe for storing variable impacts
impact_results <- data.frame(
  item_nbr = character(),
  model = character(),
  variable = character(),
  importance = numeric(),
  stringsAsFactors = FALSE
)

# Loop over each top product
for (pid in top3_products) {
  # Filter training data for this product
  train_p <- train_top %>% filter(item_nbr == pid)
  
  # Skip if not enough data
  if (nrow(train_p) < 10) {
    cat("Skipping variable impact for product", pid, ": insufficient training data\n")
    next
  }
  
  # Remove rows with NA in key variables
  relevant_cols <- c("units", "tavg", "preciptotal", "is_weekend", 
                     "is_rainy_day", "is_snowy_day", "is_stormy_day")
  train_p <- train_p[complete.cases(train_p[relevant_cols]), ]
  
  # Skip if not enough clean data after NA removal
  if (nrow(train_p) < 10) next
  
  # ---------------- Linear Regression ----------------
  lm_formula <- units ~ tavg + preciptotal + is_weekend + 
    is_rainy_day + is_snowy_day + is_stormy_day
  
  lm_model <- try(lm(lm_formula, data = train_p), silent = TRUE)
  
  if (!inherits(lm_model, "try-error")) {
    coefs <- summary(lm_model)$coefficients
    # Extract estimates for all variables (excluding intercept)
    for (var in rownames(coefs)) {
      if (var == "(Intercept)") next  # Optional: include or skip intercept
      impact_results <- rbind(impact_results, data.frame(
        item_nbr = pid,
        model = "Linear Regression",
        variable = var,
        importance = as.numeric(coefs[var, "Estimate"])
      ))
    }
  }
  
  # ---------------- Decision Tree ----------------
  tree_model <- try(rpart(lm_formula, data = train_p,
                          method = "anova",
                          control = rpart.control(cp = 0.0001, minsplit = 20)),
                    silent = TRUE)
  
  if (!inherits(tree_model, "try-error") && !is.null(tree_model$variable.importance)) {
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
}

# Finalize: convert to tibble and sort
impact_summary <- impact_results %>%
  dplyr::group_by(item_nbr, model) %>%
  dplyr::arrange(item_nbr, model, desc(abs(importance))) %>%
  dplyr::ungroup()

#show results of variables’ impact on sales for each product
print(impact_summary, n = Inf)




# Visualizations (R)

# 1. Scatter plot: Temperature vs Sales for each product 
ggplot(train_top %>% filter(item_nbr %in% c(5, 9, 45)),
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
ggplot(train_top %>% filter(item_nbr %in% c(5,9,45)) %>%
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
ggplot(train_top %>% filter(item_nbr %in% c(5,9,45)),
       aes(x = preciptotal, y = units, color = factor(item_nbr))) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Precipitation vs Sales by Product",
       x = "Total Precipitation (inches)",
       y = "Units Sold",
       color = "Product") +
  coord_cartesian(ylim = c(0, 500)) +
  theme_minimal()
