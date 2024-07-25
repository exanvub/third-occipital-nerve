#'*=======================================*'#
#'* Data analysis TON study Nicolas, EXAN *'#
#'*=======================================*'#

# 00_import_preprocess.R

rm(list = ls())


# read dataset
data.ton <- readxl::read_excel(
  path = "data/TON_DATA_for_Ben_Copy.xlsx",
  sheet = "clean"
)


# formatting variables
data.ton <- data.ton %>%
  mutate(
    specimen_id = factor(specimen_id),
    side = factor(side,c("L","R")),
    fresh_specimen = factor(fresh_specimen,c("1","0")),
    ton_crossing = factor(ton_crossing,
                          c("lateral",
                            "laterodorsal",
                            "dorsal",
                            "underneed")),
    ton_cross_angle = factor(ton_cross_angle, 
                             c("very small",
                               "small",
                               "medium",
                               "large")),
    distance_joint = factor(distance_joint, 
                            c("close",
                              "far")),
    deg_lipping = factor(deg_lipping, 
                         c("not present",
                           "slightly present",
                           "clearly present"),
                         c("no degeneration",
                           "slight degeneration",
                           "clear degeneration")),
    ton_notch = factor(ton_notch, 
                       c("not present",
                         "slightly present",
                         "clearly present"),
                       c("no notch",
                         "slight notch",
                         "clear notch"))
  ) %>%
  # make binary versions of categorical variables
  mutate(
    ton_notch2 = factor(ton_notch, 
                        c("no notch",
                          "slight notch",
                          "clear notch"),
                        c("no notch","notch","notch")),
    ton_notch2bis = factor(ton_notch, 
                           c("no notch",
                             "slight notch",
                             "clear notch"),
                           c("no notch","no notch","notch")),
    deg_lipping2 = factor(deg_lipping,
                          c("not present","slightly present","clearly present"),
                          c("not present","present","present"))
  )

# specimen 70L & 74L have missing data on every variable => remove observations
data.ton <- data.ton %>%
  filter(
    !(specimen_id == "70" & side == "L"),
    !(specimen_id == "74" & side == "L")
  )




