#'*=======================================*'#
#'* Data analysis TON study Nicolas, EXAN *'#
#'*=======================================*'#

# 02_missing_data_imputation.R

rm(list = ls())

box::use(
  tidytable[...],
  mice[...]
)

# load in the data
source("00_import_preprocess.R")



imp <- mice(data.ton, maxit = 0)

pred <- imp$predictorMatrix
pred[,c("specimen_id","fresh_specimen")] <- 0L
pred[,c("ton_notch2","deg_lipping2")] <- 0L

meth <- imp$method
meth["ton_notch2"] <- '~ case_when(
  ton_notch == "Not present" ~ "Not present",
  TRUE                       ~ "Present"
)'
meth["deg_lipping2"] <- '~ case_when(
  deg_lipping == "Not present" ~ "Not present",
  TRUE                         ~ "Present"
)'

imp <- mice(data.ton, method = meth, predictorMatrix = pred, 
            m = 20, maxit = 50)

save(
  imp,
  file = "data/data_ton_MI.rda"
)

library(ggmice)

plot_trace(imp, 'ton_notch')
plot_trace(imp, 'deg_lipping')
plot_trace(imp, 'distance_facet2notch')

