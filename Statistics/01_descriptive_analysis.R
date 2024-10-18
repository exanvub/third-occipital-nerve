#'*=======================================*'#
#'* Data analysis TON study Nicolas, EXAN *'#
#'*=======================================*'#

# 01_descriptive_analysis.R

rm(list = ls())

box::use(
  tidytable[...],
  ggplot2[...],
  gtsummary[...]
)

theme_gtsummary_compact()


# load in the data
source("00_import_preprocess.R")


# number of observations and specimens
N_obs  <- nrow(data.ton)
N_spec <- length(unique(data.ton$specimen_id))



# -------------------------------------------------------------------------
# missing data ------------------------------------------------------------

# inspect rows with missing data: 
# 10/78 rows have missing data on at least 1 variable
cc <- complete.cases(data.ton)
sum(cc); sum(!cc)
data.ton.missing <- data.ton[!cc,]
View(data.ton.missing)




# -------------------------------------------------------------------------
# specimen characteristics ------------------------------------------------

data.ton %>%
  distinct(specimen_id, .keep_all = T) %>%
  select(fresh_specimen) %>%
  tbl_summary() %>%
  bold_labels()

# other specimen characteristics? 
# age at death & sex


# -------------------------------------------------------------------------
# all characteristics of joints (total & L/R) -----------------------------

data.ton %>%
  select(-c(specimen_id, fresh_specimen, ton_notch2, deg_lipping2)) %>%
  tbl_summary(by = side) %>%
  add_overall() %>%
  bold_labels() %>%
  add_p()


# -------------------------------------------------------------------------
# Hypothesis: more degeneration = more chance of notch? -------------------

data.ton %>%
  select(deg_lipping, ton_notch, ton_notch2) %>%
  tbl_summary(by = deg_lipping) %>%
  add_overall() %>%
  bold_labels() %>%
  add_p() 

# => Fisher's exact test confirms the association.


# -------------------------------------------------------------------------
# Hypothesis: more lateral = more chance of a notch? ----------------------

data.ton %>%
  select(ton_crossing, ton_notch, ton_notch2) %>%
  tbl_summary(by = ton_crossing) %>% 
  add_overall() %>%
  bold_labels() %>%
  add_p()

# => Fisher's exact test confirms the association.

data.ton %>%
  select(distance_before, ton_notch) %>%
  tbl_summary(by = ton_notch,
              type = all_continuous() ~ "continuous2",
              statistic = all_continuous() ~ c(
                "{median} ({p25}, {p75})",
                "{min}, {max}"
              )) %>% 
  add_overall() %>%
  bold_labels() %>%
  add_p()

# Hypothesis: more angle = more chance of a notch? ----------------------

data.ton %>%
  select(ton_cross_angle, ton_notch, ton_notch2) %>%
  tbl_summary(by = ton_cross_angle) %>% 
  add_overall() %>%
  bold_labels() %>%
  add_p()

# -------------------------------------------------------------------------
# Hypothesis: more tissue = smaller chance of a notch ---------------------

data.ton %>%
  select(distance_joint, ton_notch, ton_notch2) %>%
  tbl_summary(by = distance_joint) %>% 
  add_overall() %>%
  bold_labels() %>%
  add_p() 
# data.ton %>%
#   select(distance_joint, ton_notch, ton_notch2) %>%
#   tbl_summary(by = distance_joint) %>% 
#   add_overall() %>%
#   bold_labels() %>%
#   add_p(test = list(ton_notch = "chisq.test",
#                     ton_notch2 = "chisq.test")) 

# => Fisher's exact test confirms the association.


# -------------------------------------------------------------------------
# Hypothesis: more lateral = greater angle --------------------------------

data.ton %>%
  select(ton_crossing, ton_cross_angle, angle_avg) %>%
  tbl_summary(by = ton_crossing) %>% 
  add_overall() %>%
  bold_labels() %>%
  add_p(test = list(ton_cross_angle = "chisq.test")) 

# => convergence issue for Fisher's exact test (but association confirmed 
# by Pearson's chi² test and kruskal-wallis rank sum test)


# ----------------------------------------------------------------------------
# Hypothesis: more dorsal = bigger chance for connection between NOT & GON ---

data.ton %>%
  select(ton_crossing, comm_branch, comm_branch_after, comm_branch_lateral) %>%
  tbl_summary(by = ton_crossing) %>% 
  add_overall() %>%
  bold_labels() %>%
  add_p()

data.ton %>%
  select(comm_branch, distance_before) %>%
  tbl_summary(by = comm_branch) %>%
  add_overall() %>%
  bold_labels() %>%
  add_p()


# -------------------------------------------------------------------------
# ton notch & comm_branch notch ~ deg_lipping

data.ton %>%
  select(deg_lipping, ton_notch, comm_branch_notch) %>%
  mutate(any_notch = case_when(
    ton_notch %in% c("slight notch","clear notch") | comm_branch_notch == 1 ~ "yes",
    TRUE ~ "no"
  )) %>%
  tbl_summary(by = deg_lipping) %>%
  add_overall() %>%
  bold_labels() %>%
  add_p()


# -------------------------------------------------------------------------
# left-right (a)symmetry --------------------------------------------------

# categorical variables
prop.symmetric <- function(x) {
  data.ton %>%
    filter(.by = specimen_id, sum(is.na(!!!rlang::syms(x))) == 0) %>%
    summarise(.by = specimen_id, symm = n_distinct(!!!rlang::syms(x))) %>%
    summarise(symm = sum(symm == 1)) %>% pull(symm)
}

prop.symmetric("ton_crossing")
prop.symmetric("ton_cross_angle")
prop.symmetric("distance_joint")
prop.symmetric("deg_lipping")
prop.symmetric("ton_notch")


# continuous variables
abs.symmetric <- function(x) {
  data.ton %>%
    filter(.by = specimen_id, sum(is.na(!!!rlang::syms(x))) == 0) %>%
    summarise(.by = specimen_id, symm = max(!!!rlang::syms(x)) - min(!!!rlang::syms(x))) %>%
    summarise(mean_symm = mean(symm)) %>% pull(mean_symm)
}

abs.symmetric("angle_avg")
abs.symmetric("distance_before")


# -------------------------------------------------------------------------
# numerical vs. categorical -----------------------------------------------

ggplot(data.ton, aes(ton_cross_angle, angle_avg)) +
  geom_boxplot() + geom_jitter(height=0)

ggplot(data.ton, aes(ton_cross_angle, angle_crossing)) +
  geom_boxplot() + geom_jitter(height=0)

ggplot(data.ton, aes(ton_cross_angle, angle_closest)) +
  geom_boxplot() + geom_jitter(height=0)

ggplot(data.ton, aes(ton_cross_angle, angle_plane)) +
  geom_boxplot() + geom_jitter(height=0)

ggplot(data.ton, aes(ton_crossing, distance_before)) +
  geom_boxplot() + geom_jitter(height=0)


# -------------------------------------------------------------------------
# graphical summaries -----------------------------------------------------

# notch ~ lipping
ggplot(data.ton, aes(deg_lipping, fill = ton_notch)) +
  geom_bar()
ggplot(data.ton, aes(deg_lipping, fill = ton_notch)) +
  geom_bar(position = position_fill())

# notch ~ crossing
ggplot(data.ton, aes(ton_crossing, fill = ton_notch)) +
  geom_bar()
ggplot(data.ton, aes(ton_crossing, fill = ton_notch)) +
  geom_bar(position = position_fill())
ggplot(data.ton, aes(distance_before, ton_notch)) +
  geom_point()
ggplot(data.ton, aes(distance_before, ton_notch2, group = 1)) +
  geom_point() + geom_smooth() + geom_smooth(method = "lm", se = F, color = "red")
ggplot(data.ton, aes(distance_before, ton_notch, group = 1)) +
  geom_point() + geom_smooth() + geom_smooth(method = "lm", se = F, color = "red")
#"weighted" version
data.ton.clearnotch <- data.ton %>% filter(ton_notch == "clear notch")
data.ton.clearnotch <- data.table::rbindlist(rep(data.ton.clearnotch, 20))
ggplot(rbind(data.ton,data.ton.clearnotch), aes(distance_before, ton_notch, group = 1)) +
  geom_point() + geom_smooth() + geom_smooth(method = "lm", se = F, color = "red")


# notch ~ tissue
ggplot(data.ton, aes(distance_joint, fill = ton_notch)) +
  geom_bar()
ggplot(data.ton, aes(distance_joint, fill = ton_notch)) +
  geom_bar(position = position_fill())

# notch ~ lipping + crossing + tissue
data.ton %>% 
  filter(!is.na(distance_joint), !is.na(ton_notch),
         !is.na(ton_crossing), !is.na(deg_lipping)) %>%
  ggplot(aes(deg_lipping, fill = ton_notch)) +
  facet_grid(distance_joint ~ ton_crossing) +
  geom_bar() 
data.ton %>% 
  filter(!is.na(distance_joint), !is.na(ton_notch),
         !is.na(ton_crossing), !is.na(deg_lipping)) %>%
  ggplot(aes(deg_lipping, fill = ton_notch)) +
  facet_grid(distance_joint ~ ton_crossing) +

    geom_bar(position = position_fill()) 

