# ---- 1. Load required package ----
if (!require("frontier")) install.packages("frontier")
library(frontier)

# ---- 2. Function to generate Translog formula (auto-clean version) ----
make_translog_formula <- function(data, dep_var = "value",
                                  exclude_vars = c("iso3c", "iso2c", "iso2c.x", "indicator", "year"),
                                  na_threshold = 0.5) {
  # Ensure dependent variable exists
  if (!(dep_var %in% names(data))) {
    stop(paste("Dependent variable", dep_var, "not found in dataset."))
  }
  
  # Candidate independent variables
  inputs <- setdiff(names(data), c(dep_var, exclude_vars))
  
  # Keep only numeric inputs
  inputs <- inputs[sapply(data[inputs], is.numeric)]
  
  
  
  # If nothing left, stop
  if (length(inputs) == 0) stop("No valid input variables found after filtering.")
  
  # LHS (dependent variable in logs)
  lhs <- paste0("log(", dep_var, ")")
  
  # First-order terms
  first_order <- paste0("log(", inputs, ")", collapse = " + ")
  
  # Quadratic terms
  squares <- paste0("I(log(", inputs, ")^2)", collapse = " + ")
  
  # Cross terms
  cross_terms <- c()
  for (i in 1:(length(inputs)-1)) {
    for (j in (i+1):length(inputs)) {
      cross_terms <- c(cross_terms,
                       paste0("I(log(", inputs[i], ")*log(", inputs[j], "))"))
    }
  }
  cross_terms <- paste(cross_terms, collapse = " + ")
  
  # RHS
  rhs <- paste(c(first_order, squares, cross_terms), collapse = " + ")
  
  # Final formula
  formula_str <- paste(lhs, "~", rhs)
  as.formula(formula_str)
}

# ---- 3. Load your CSV data ----
data <- read.csv("C:/Users/jrady/Desktop/SDG Costing/Data/merged_sdg_dataset_testing.csv")

# ---- 4. Create Translog formula ----
translog_formula <- make_translog_formula(data)
print(translog_formula)

# ---- 5. Estimate SFA model with time-varying inefficiency ----
sfa_model <- sfa(
  formula = translog_formula,
  data = data,
  timeEffect = TRUE  
)


summary(sfa_model)


TE <- efficiencies(sfa_model)

