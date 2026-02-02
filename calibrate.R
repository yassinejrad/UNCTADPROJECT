suppressMessages({
  if (!require("frontier")) install.packages("frontier", repos="http://cran.us.r-project.org")
  library(frontier)
  if (!require("optparse")) install.packages("optparse", repos="http://cran.us.r-project.org")
  library(optparse)
  if (!require("openxlsx")) install.packages("openxlsx", repos="http://cran.us.r-project.org")
  library(openxlsx)
  if (!require("plm")) install.packages("plm", repos="http://cran.us.r-project.org")
  library(plm)
  if (!require("dplyr")) install.packages("dplyr", repos="http://cran.us.r-project.org")
  library(dplyr)
})

# ------------------------
# Command-line options
# ------------------------
option_list <- list(
  make_option("--input", type="character"),
  make_option("--indicator", type="character"),
  make_option("--expenditures", type="character", default = ""),
  make_option("--controls", type="character", default = ""),
  make_option("--eff_controls", type="character", default = ""),
  make_option("--eff_fe", type="character", default = "1"),
  make_option("--main_fe", type="character", default = "1"),
  make_option("--main_intercept", type="character", default = "1"),
  make_option("--eff_intercept", type="character", default = "1"),
  make_option("--ref_country", type="character", default = "China"),
  make_option("--model", type="character", default = "translog"),
  make_option("--target_direction", type="character"),
  
  make_option("--output", type="character")
)

opt <- parse_args(OptionParser(option_list = option_list))

# ------------------------
# Load data
# ------------------------
data <- read.csv(opt$input, stringsAsFactors = FALSE)
names(data) <- make.names(names(data))
pdata <- pdata.frame(data, index = c("Country_name", "years"))

# ------------------------
# Reference country handling
# ------------------------
if (opt$ref_country != "") {
  if (!"Country_name" %in% names(data)) stop("❌ Country_name column not found in data.")
  if (!opt$ref_country %in% data$Country_name)
    stop(paste0("❌ Reference country not found: ", opt$ref_country))
  data$Country_name <- relevel(factor(data$Country_name), ref = opt$ref_country)
}

# ------------------------
# Helpers
# ------------------------
parse_vars <- function(x) {
  if (is.null(x) || x == "") return(character(0))
  make.names(trimws(unlist(strsplit(x, ","))))
}

exp_vars  <- parse_vars(opt$expenditures)
ctrl_vars <- parse_vars(opt$controls)
eff_ctrls <- parse_vars(opt$eff_controls)

eff_fe         <- as.logical(as.integer(opt$eff_fe))
main_fe        <- as.logical(as.integer(opt$main_fe))
main_intercept <- as.logical(as.integer(opt$main_intercept))
eff_intercept  <- as.logical(as.integer(opt$eff_intercept))

model_type <- tolower(opt$model)
if (model_type != "translog") stop("❌ Only TRANSLOG model is supported.")

# ------------------------
# TargetDirection → ineffDecrease
# ------------------------
target_direction <- tolower(trimws(opt$target_direction))
ineff_decrease <- FALSE  # default (safe)

if (target_direction == "upperorequal") {
  ineff_decrease <- TRUE
} else if (target_direction == "lowerorequal") {
  ineff_decrease <- FALSE
} else if (target_direction == "equal") {
  ineff_decrease <- TRUE
}

cat("ℹ Indicator        :", opt$indicator, "\n")
cat("ℹ TargetDirection  :", target_direction, "\n")
cat("ℹ ineffDecrease    :", ineff_decrease, "\n")

# ------------------------
# Formula builder
# ------------------------
make_model_formula <- function(
    data, dep_var, exp_vars, ctrl_vars,
    eff_ctrls, eff_fe, main_fe,
    main_intercept, eff_intercept
) {
  
  inputs <- exp_vars[exp_vars %in% names(data)]
  inputs <- inputs[sapply(inputs, function(x) is.numeric(data[[x]]))]
  if (length(inputs) == 0) stop("❌ No valid numeric expenditure variables.")
  
  ctrl_vars <- ctrl_vars[ctrl_vars %in% names(data)]
  ctrl_vars <- ctrl_vars[sapply(ctrl_vars, function(x) is.numeric(data[[x]]))]
  
  lhs <- paste0("log(", dep_var, ")")
  
  first_order <- paste0("log(", inputs, ")", collapse = " + ")
  squares <- paste0("I(log(", inputs, ")^2)", collapse = " + ")
  
  cross_terms <- c()
  if (length(inputs) > 1) {
    for (i in 1:(length(inputs)-1)) {
      for (j in (i+1):length(inputs)) {
        cross_terms <- c(
          cross_terms,
          paste0("I(log(", inputs[i], ") * log(", inputs[j], "))")
        )
      }
    }
  }
  
  rhs_terms <- c(first_order, squares, cross_terms)
  
  if (length(ctrl_vars) > 0)
    rhs_terms <- c(rhs_terms, paste0("log(", ctrl_vars, ")"))
  
  if (main_fe)
    rhs_terms <- c(rhs_terms, "factor(Country_name)")
  
  rhs <- paste(rhs_terms, collapse = " + ")
  if (!main_intercept)
    rhs <- paste0(rhs, " -1")
  
  if (length(eff_ctrls) > 0) {
    eff_ctrls <- eff_ctrls[eff_ctrls %in% names(data)]
    eff_ctrls <- eff_ctrls[sapply(eff_ctrls, function(x) is.numeric(data[[x]]))]
    if (length(eff_ctrls) == 0) stop("❌ Invalid efficiency controls.")
    
    ineff_terms <- eff_ctrls
    if (eff_fe)
      ineff_terms <- c(ineff_terms, "factor(Country_name)")
    
    ineff_suffix <- ifelse(!eff_intercept, "-1", "")
    formula_str <- paste(
      lhs, "~", rhs, "|",
      paste(ineff_terms, collapse = " + "),
      ineff_suffix
    )
  } else {
    formula_str <- paste(lhs, "~", rhs)
  }
  
  as.formula(formula_str)
}

# ------------------------
# Build & fit SFA model
# ------------------------
dep_var <- make.names(opt$indicator)

model_formula <- make_model_formula(
  data, dep_var, exp_vars, ctrl_vars,
  eff_ctrls, eff_fe, main_fe,
  main_intercept, eff_intercept
)

sfa_model <- sfa(
  formula = model_formula,
  data = pdata,
  truncNorm = TRUE,
  ineffDecrease = ineff_decrease
)

# ------------------------
# Export results
# ------------------------
base_output_dir <- dirname(opt$output)
indicator_name <- dep_var

coef_dir        <- file.path(base_output_dir, "coef")
summary_dir     <- file.path(base_output_dir, "summary")
descriptive_dir <- file.path(base_output_dir, "descriptive")

dir.create(coef_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(summary_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(descriptive_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Coefficients
coef_file <- file.path(coef_dir, paste0("sfa_coefficients_", indicator_name, ".csv"))

coef_df <- data.frame(
  Name  = names(coef(sfa_model)),
  Value = as.numeric(coef(sfa_model))
)


write.csv(coef_df, coef_file, row.names = FALSE)
cat("✅ Coefficients saved to:", coef_file, "\n")

# ---- Summary
summary_file <- file.path(summary_dir, paste0("sfa_summary_", indicator_name, ".txt"))
summary_txt <- capture.output(summary(sfa_model))
writeLines(summary_txt, summary_file)
cat("✅ Summary saved to:", summary_file, "\n")
