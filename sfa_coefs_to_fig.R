############################################################
# SP Costing – SFA Coefficients to Fig Format 
############################################################

suppressMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(openxlsx)
  library(RColorBrewer)
})

# ----------------------------
# Base paths
# ----------------------------
base_input_dir  <- "C:/Users/jrady/Desktop/SDG Costing/Data/coef"
base_output_dir <- "C:/Users/jrady/Desktop/SDG Costing/Data/Fig"
dir.create(base_output_dir, showWarnings = FALSE, recursive = TRUE)

# ----------------------------
# List files
# ----------------------------
files <- list.files(base_input_dir, pattern = "^sfa_coefficients_.*\\.csv$", full.names = TRUE)
if(length(files) == 0) stop("No sfa_coefficients_*.csv files found!")

# ----------------------------
# Parsing function
# ----------------------------
parse_term <- function(term, value){
  term <- str_replace_all(term, " ", "")
  gvt_pattern <- "GVT_[A-Z_]+(_PC)?"
  
  if(str_detect(term, "\\*")){
    vars <- str_extract_all(term, gvt_pattern)[[1]]
    from <- vars[1]; to <- vars[2]
  } else if(str_detect(term, "\\^2")){
    var <- str_extract(term, gvt_pattern)
    from <- paste0(var,"^2"); to <- "-"
  } else {
    var <- str_extract(term,gvt_pattern)
    from <- var; to <- "-"
  }
  
  clean <- function(x){
    x %>% str_remove("^GVT_") %>% str_replace_all("_"," ") %>% str_to_title()
  }
  
  from <- clean(from)
  to <- ifelse(to=="-","-",clean(to))
  
  tibble(From=from, To=to, Weight=value)
}

# ----------------------------
# Dynamic ColorIndex & colors
# ----------------------------
all_indicators <- str_extract(basename(files), "X[0-9\\.]+") %>% unique()
color_index_map <- setNames(seq_along(all_indicators), all_indicators)
indices <- sort(unique(color_index_map))
n <- length(indices)
colors <- brewer.pal(min(n,12),"Set3")
if(n>12) colors <- colorRampPalette(colors)(n)
color_map <- setNames(colors, indices)

# ----------------------------
# Process files
# ----------------------------
all_fig_data <- list()
for(file_path in files){
  coeffs <- read_csv(file_path, show_col_types = FALSE)
  indicator <- str_trim(str_extract(basename(file_path), "X[0-9\\.]+"))
  color_index <- color_index_map[[indicator]]
  
  coeffs <- coeffs %>% filter(str_detect(Name,"GVT_"))
  
  fig_data <- coeffs %>%
    rowwise() %>%
    do(parse_term(.$Name,.$Value)) %>%
    ungroup() %>%
    mutate(
      ColorIndex = color_index,
      Indicator = indicator
    )
  
  all_fig_data[[indicator]] <- fig_data
}

final_fig_data <- bind_rows(all_fig_data)

# ----------------------------
# Create workbook
# ----------------------------
wb <- createWorkbook()
addWorksheet(wb, "FigData")

# Write main data 
writeData(wb, "FigData", final_fig_data)

# Apply color to ColorIndex column
for(i in seq_len(nrow(final_fig_data))){
  addStyle(wb, "FigData", style = createStyle(fgFill = color_map[as.character(final_fig_data$ColorIndex[i])]), 
           rows = i+1, cols = which(names(final_fig_data)=="ColorIndex"), gridExpand = TRUE)
}

# ----------------------------
# Add legend next to data
# ----------------------------
start_col <- ncol(final_fig_data) + 3  # space of 2 columns
legend_df <- tibble(ColorIndex = indices)
writeData(wb, "FigData", legend_df, startCol = start_col, startRow = 1)

# Color the adjacent cell in legend
for(i in seq_along(indices)){
  addStyle(wb, "FigData", style = createStyle(fgFill = colors[i]), 
           rows = i+1, cols = start_col+1, gridExpand = TRUE)
}

# ----------------------------
# Save workbook
# ----------------------------
output_path <- file.path(base_output_dir, "Fig_All_Indicators.xlsx")
saveWorkbook(wb, output_path, overwrite = TRUE)
cat("\n✅ File with colored ColorIndex and legend (side by side) saved to:\n", output_path, "\n")
