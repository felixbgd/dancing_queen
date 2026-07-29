library(readr)
library(lme4)
library(emmeans)
library(car)
library(ggplot2)
library(dplyr)
library(ordinal)
library(this.path)

# --- 1. Set the dataframe ---
currentdir <- this.path::here()
setwd(paste(currentdir, "DATA/study5_rating_csv", sep = "/"))

# --- 2. Test recognition against chance ---
data <- read_csv("df_score.csv", col_types = cols(...1 = col_skip()))
data$songfunction <- as.factor(data$songfunction)
data$sex <- as.factor(data$sex)

# Make sure your response is an ordered factor
# IMPORTANT: define levels in the correct order (from lullaby to playsong)
data$score <- ordered(
  data$score,
  levels = c(
    "-1",
    "-0.66",
    "-0.33",
    "0.33",
    "0.66",
    "1"
  )
)

data %>%
  mutate(score_num = as.numeric(as.character(score))) %>%
  group_by(PMmanip,songfunction) %>%
  summarise(mean_score = mean(score_num, na.rm = TRUE))

# Fit the CLMM
model <- clmm("score ~ songfunction * PMmanip * sex + (1 | participant)",data = data)

# Post-hoc contrasts
emt = emmeans(model, pairwise ~ songfunction * PMmanip)
summary(emt, infer = c(TRUE, TRUE)) #test if different from zero


# --- 3. Test causal effect of sway/bounce removal on recognition drop ---
data <- read_csv("df_delta.csv", col_types = cols(...1 = col_skip()))
data$songfunction <- as.factor(data$songfunction)
data$PMmanip <- as.factor(data$PMmanip)
data$sex <- as.factor(data$sex)

# The "ideal" delta values
allowed_deltas <- c(-2, -1.66, -1.33, -1, -0.66, -0.33, 0, 0.33, 0.66, 1, 1.33, 1.66, 2)

data$delta_fixed <- sapply(data$delta, function(x) {
  allowed_deltas[which.min(abs(x - allowed_deltas))]
})

# Make sure your response is an ordered factor
# IMPORTANT: define levels in the correct order (from lullaby to playsong)
data$delta <- ordered(
  data$delta_fixed,
  levels = c(-2, -1.66, -1.33, -1, -0.66, -0.33, 0, 0.33, 0.66, 1, 1.33, 1.66, 2
  )
)

# Fit the CLMM
model <- clmm("delta ~ songfunction * PMmanip * sex  + (1|participant)" , data = data)
Anova(model, type = 3)

# Post-hoc contrasts
emt = emmeans(model, pairwise ~ songfunction * PMmanip)
summary(emt, infer = c(TRUE, TRUE)) #test if different from zero


# --- 4. Better understand sex differences ---
# --- Separate models per male/female ---
# Female
data_fem <- filter(data, sex == "female")
model_fem <- clmm("delta ~ songfunction * PMmanip + (1 | participant)", data = data_fem)
anova_fem <- Anova(model_fem, type = 3)
print("=== Female model results ===")
print(anova_fem)

# Post-hoc contrasts
emt = emmeans(model_fem, pairwise ~ songfunction * PMmanip)
summary(emt, infer = c(TRUE, TRUE)) #test if different from zero


# Male
data_mal <- filter(data, sex == "male")
model_mal <- clmm("delta ~ songfunction * PMmanip  + (1 | participant)", data = data_mal)
anova_mal <- Anova(model_mal, type = 3)
print("=== Male model results ===")
print(anova_mal)

# Post-hoc contrasts
emt = emmeans(model_mal, pairwise ~ songfunction * PMmanip)
summary(emt, infer = c(TRUE, TRUE)) #test if different from zero

