#'*=======================================*'#
#'* Data analysis TON study Nicolas, EXAN *'#
#'*=======================================*'#

# 03_statistical_modeling.R

rm(list = ls())

box::use(
  tidytable[...],
  ggplot2[...],
  gtsummary[...],
  nnet[...],
  MASS[...],
  gofcat[brant.test],
  lme4[...]
)

# load in the data
source("00_import_preprocess.R")





# -------------------------------------------------------------------------
# ignoring the clustering of observations within subjects -----------------


### multinomial models ###
m1 <- multinom(
  ton_notch ~ deg_lipping,
  data = data.ton
); summary(m1)
m2 <- multinom(
  ton_notch ~ ton_crossing,
  data = data.ton
); summary(m2)
m3 <- multinom(
  ton_notch ~ distance_joint,
  data = data.ton
); summary(m3)
m4 <- multinom(
  ton_notch ~ deg_lipping + ton_crossing + distance_joint,
  data = data.ton
); summary(m4)
m5 <- multinom(
  ton_notch ~ deg_lipping + ton_crossing,
  data = data.ton
); summary(m5) 
m6 <- multinom(
  ton_notch ~ deg_lipping + distance_joint,
  data = data.ton
); summary(m6)
m7 <- multinom(
  ton_notch ~ deg_lipping * ton_crossing,
  data = data.ton
); summary(m7)

# best fit (AIC) for model m5
summary(m5) 
#' interpretation of coefficients:
#' given the location of crossing of the TON, the odds that there is a slightly 
#' present notch compared to no notch is exp(3.257435) = 26 times higher when 
#' there is slightly degenerate lipping compared to no degenerate lipping


### proportional odds models ###
m8 <- polr(
  ton_notch ~ deg_lipping,
  data = data.ton, Hess = T
); summary(m8)
m9 <- polr(
  ton_notch ~ ton_crossing,
  data = data.ton, Hess = T
); summary(m9)
m10 <- polr(
  ton_notch ~ distance_joint,
  data = data.ton, Hess = T
); summary(m10)
m11 <- polr(
  ton_notch ~ deg_lipping + ton_crossing + distance_joint,
  data = data.ton, Hess = T
); summary(m11)
m12 <- polr(
  ton_notch ~ deg_lipping + ton_crossing,
  data = data.ton, Hess = T
); summary(m12) 
m13 <- polr(
  ton_notch ~ deg_lipping + distance_joint,
  data = data.ton, Hess = T
); summary(m13)
m14 <- polr(
  ton_notch ~ deg_lipping * ton_crossing,
  data = data.ton, Hess = T
); summary(m14)

# best fit (AIC) for model m12
summary(m12) 
brant.test(m12) # test of proportional odds assumption
#' interpretation of coefficients:
#' given the location of crossing of the TON, the odds that there is no notch 
#' or a slightly present notch increases by a factor of exp(2.6724) = 14 when
#' there is slightly degenerate lipping


### binomial models ###
m13 <- glm(
  ton_notch2 ~ deg_lipping, family = binomial,
  data = data.ton
); summary(m13)
m14 <- glm(
  ton_notch2 ~ ton_crossing, family = binomial,
  data = data.ton
); summary(m14)
m15 <- glm(
  ton_notch2 ~ distance_joint, family = binomial,
  data = data.ton
); summary(m15)
m16 <- glm(
  ton_notch2 ~ deg_lipping + ton_crossing + distance_joint, family = binomial,
  data = data.ton
); summary(m16)
m17 <- glm(
  ton_notch2 ~ deg_lipping + ton_crossing, family = binomial,
  data = data.ton
); summary(m17) 
m18 <- glm(
  ton_notch2 ~ deg_lipping + distance_joint, family = binomial,
  data = data.ton
); summary(m18)
m19 <- glm(
  ton_notch2 ~ deg_lipping * ton_crossing, family = binomial,
  data = data.ton
); summary(m19)
m20 <- glm(
  ton_notch2 ~ deg_lipping * distance_joint, family = binomial,
  data = data.ton
); summary(m20)

# best fit (AIC) for model m17
summary(m17) 
#' interpretation of coefficients:
#' given the location of crossing of the TON, the odds that there is a notch 
#' increases by a factor of exp(3.2572) = 26 when there is slightly degenerate lipping




# -------------------------------------------------------------------------
# accounting for the clustering of observations within subjects -----------


### binomial mixed models ###
m21 <- glmer(
  ton_notch2 ~ deg_lipping + (1|specimen_id), family = binomial,
  data = data.ton
); summary(m21)
m22 <- glmer(
  ton_notch2 ~ ton_crossing + (1|specimen_id), family = binomial,
  data = data.ton
); summary(m22) # convergence issues
m23 <- glmer(
  ton_notch2 ~ distance_joint + (1|specimen_id), family = binomial,
  data = data.ton
); summary(m23)
m24 <- glmer(
  ton_notch2 ~ deg_lipping + ton_crossing + distance_joint + (1|specimen_id), 
  family = binomial, data = data.ton
); summary(m24) # convergence issues
m25 <- glmer(
  ton_notch2 ~ deg_lipping + distance_joint + (1|specimen_id), 
  family = binomial, data = data.ton
); summary(m25) 
m26 <- glmer(
  ton_notch2 ~ deg_lipping + ton_crossing + (1|specimen_id), 
  family = binomial, data = data.ton
); summary(m26) # convergence issues

# best fit (AIC) for model m25
summary(m25) 
#' interpretation of coefficients:
#' given the distance between the TON and joint, the odds that there is a notch 
#' increases by a factor of exp(38.224) = 4e+16 when there is slightly degenerate lipping
#' = nonsensical model!

# univariable binomial mixed model for deg_lipping fits better (m21)
summary(m21)
#' interpretation of coefficients:
#' the odds that there is a notch increases by a factor of exp(2.989) = 20 when 
#' there is slightly degenerate lipping
#' SD of the random intercept (baseline log odds) = 1.66




# -------------------------------------------------------------------------


m <- glm(
  ton_notch2 ~ deg_lipping + distance_after + distance_joint,
  family = binomial,
  data = data.ton
)
summary(m)

newdats <- expand_grid(
  deg_lipping = unique(data.ton$deg_lipping),
  distance_joint = unique(data.ton$distance_joint),
#  distance_before = 0:101
  distance_after = 100:-1
)

newdats <- cbind(
  newdats, prob_notch = predict(m, newdata = newdats, type = "response"))

ggplot(newdats, aes(distance_after, prob_notch, color = deg_lipping,
                    lty = distance_joint)) +
  geom_line() 

# -------------------------------------------------------------------------


m <- glm(
  ton_notch2 ~ deg_lipping + angle_crossing + distance_joint,
  family = binomial,
  data = data.ton
)
summary(m)

newdats <- expand_grid(
  deg_lipping = unique(data.ton$deg_lipping),
  distance_joint = unique(data.ton$distance_joint),
  #  distance_before = 0:101
  angle_crossing = 0:100
)

newdats <- cbind(
  newdats, prob_notch = predict(m, newdata = newdats, type = "response"))

ggplot(newdats, aes(angle_crossing, prob_notch, color = deg_lipping,
                    lty = distance_joint)) +
  geom_line() 


# too little observations with clear notch (6) to model
# m <- glm(
#   ton_notch2bis ~ deg_lipping + distance_before + distance_joint,
#   family = binomial,
#   data = data.ton
# )
# summary(m)

# m <- glmer(
#   ton_notch2 ~ deg_lipping + (1|specimen_id),
#   family = binomial,
#   data = data.ton
# )
# summary(m)




# -------------------------------------------------------------------------
# weighted version

data.ton <- data.ton %>%
  mutate(
    weight = case_when(ton_notch == "clear notch" ~ 2,
                       TRUE ~ 1)
  )

m <- glm(
  ton_notch2 ~ deg_lipping + distance_before + distance_joint,
  family = binomial, weights = weight,
  data = data.ton
)
summary(m)

newdats <- expand_grid(
  deg_lipping = unique(data.ton$deg_lipping),
  distance_joint = unique(data.ton$distance_joint),
  distance_before = 0:101
)

newdats <- cbind(
  newdats, prob_notch = predict(m, newdata = newdats, type = "response"))

ggplot(newdats, aes(distance_before, prob_notch, color = deg_lipping,
                    lty = distance_joint)) +
  geom_line() 


# -------------------------------------------------------------------------

# Fit the GLM including the new variable 'angle_crossing'
m <- glm(
  ton_notch2 ~ deg_lipping + distance_after + distance_joint + angle_crossing,
  family = binomial,
  data = data.ton
)

# Print the summary of the model
summary(m)

# Generate new data for predictions
newdats <- expand.grid(
  deg_lipping = unique(data.ton$deg_lipping),
  distance_joint = unique(data.ton$distance_joint),
  distance_after = 100:0,
  angle_crossing = seq(0, 90, length.out = 10)  # Example range for angle_crossing
)

# Add the predicted probabilities to the new data
newdats <- cbind(
  newdats, prob_notch = predict(m, newdata = newdats, type = "response")
)

# Plot the results
library(ggplot2)
ggplot(newdats, aes(x = distance_after, y = prob_notch, color = factor(deg_lipping), linetype = factor(distance_joint))) +
  geom_line() +
  facet_wrap(~ angle_crossing, labeller = label_both) +  # Add faceting by angle_crossing
  labs(x = "Distance After", y = "Predicted Probability of Notch", color = "Deg Lipping", linetype = "Distance Joint") +
  theme_minimal()


# -------------------------------------------------------------------------

# Run your glm model
m <- glm(
  ton_notch2 ~ deg_lipping + distance_after + distance_joint,
  family = binomial,
  data = data.ton
)
summary(m)

# Create new data for prediction
newdats <- expand_grid(
  deg_lipping = unique(data.ton$deg_lipping),
  distance_joint = unique(data.ton$distance_joint),
  distance_after = 100:-1
)

# Predict probabilities
newdats <- cbind(
  newdats, prob_notch = predict(m, newdata = newdats, type = "response")
)

# Plot with updated axis labels, legend titles, thicker lines, and custom colors
ggplot(newdats, aes(distance_after, prob_notch, color = deg_lipping, lty = distance_joint)) +
  geom_line(size = 1) +  # Adjust size to make lines thicker
  scale_color_manual(
    values = c("no degeneration" = "green", "slight degeneration" = "blue", "clear degeneration" = "red")
  ) +
  labs(
    x = "Crossing Location (Dorsal to Lateral)",
    y = "Probability of a TON notch",
    color = "Joint Degeneration",
    lty = "Proximity to the joint"
  )


