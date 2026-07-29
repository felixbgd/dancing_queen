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
setwd(paste(currentdir, "DATA/study3_rating_csv", sep = "/"))

data <- read_csv("df_score.csv", col_types = cols(...1 = col_skip()))
data$songfunction <- as.factor(data$songfunction)
data$tempocontrol <- as.factor(data$tempocontrol)
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

# --- 2. Fit the mixed model ---
# Fit the CLMM
model <- clmm(
  score ~ songfunction * tempocontrol * sex + (1 | participant),
  data = data,
  link = "logit"  # logit link is standard
)

# Likelihood-ratio tests
Anova(model, type = 3)   # from the car package

# Post-hoc contrasts
emt = emmeans(model, pairwise ~  songfunction * tempocontrol)
summary(emt, infer = c(TRUE, TRUE)) 

emt = emmeans(model, pairwise ~  sex)
summary(emt, infer = c(TRUE, TRUE)) 

emt = emmeans(model, pairwise ~  sex * songfunction)
summary(emt, infer = c(TRUE, TRUE)) 

# --- 3. Test for covariation with Sway and Bounce activation ---
# Read data
data <- read_csv("df_score_signed_PMmodel.csv", col_types = cols(...1 = col_skip()))

# Ensure correct variable types
data$bounce_var <- as.numeric(data$bounce_var)
data$sway_var   <- as.numeric(data$sway_var)
data$tempocontrol  <- as.factor(data$tempocontrol)
data$songfunction   <- as.factor(data$songfunction)
data$participant <- as.factor(data$participant)

data$sway_z   <- scale(data$sway_var)
data$bounce_z <- scale(data$bounce_var)


# Convert score to ordered factor
# (Assumes numeric codes -1, -0.5, 0, 0.5, 1 or similar)
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

# Fit cumulative link mixed model (ordinal mixed model)
model_pm <- clmm(score ~ tempocontrol * (sway_z + bounce_z) +
                   (1 | participant),
                 data = data)

Anova(model_pm, type = 3)

# Slope of sway_var within MUS vs BEAT
emtrends(model_pm, ~ tempocontrol, var = "sway_z")
emtrends(model_pm, ~ tempocontrol, var = "bounce_z")

summary(model_pm)
model_pm$beta

