library(readr)
library(lme4)
library(emmeans)
library(car)
library(ggplot2)
library(dplyr)
library(ordinal)
library(this.path)

# Function to read delta dataframes from all 3 samples
allowed_deltas <- c(-2, -1.66, -1.33, -1, -0.66, -0.33, 0,
                    0.33, 0.66, 1, 1.33, 1.66, 2)

read_delta <- function(path, sample_name) {
  read_csv(file.path(path, "df_delta.csv"), col_types = cols(...1 = col_skip())) %>%
    mutate(
      sample = sample_name,
      songfunction = factor(songfunction),
      PMmanip = factor(PMmanip),
      sex = factor(sex),
      caregiver = factor(caregiver),
      delta_fixed = sapply(delta, function(x) {
        allowed_deltas[which.min(abs(x - allowed_deltas))]
      }),
      delta = ordered(delta_fixed, levels = allowed_deltas)
    )
}


# --- 1. Read dataframes ---
currentdir <- this.path::here()

italy <- read_delta(paste(currentdir, "DATA/study4_rating_csv", sep = "/"),
                    "Italy")

nonwestern <- read_delta(paste(currentdir, "DATA/study5_rating_csv", sep = "/"),
                         "NonWestern")

scandinavia <- read_delta(paste(currentdir, "DATA/study6_rating_csv", sep = "/"),
                          "Scandinavia")

dat_all <- bind_rows(italy, nonwestern, scandinavia) %>%
  mutate(
    sample = factor(sample, levels = c("Italy", "NonWestern", "Scandinavia"))
  )

dat_all$participantID <- interaction(dat_all$sample,
                                     dat_all$participant,
                                     drop = TRUE)

model_4way <- clmm(
  delta ~ songfunction * PMmanip * sex * sample +
    (1 | participant),
  data = dat_all,
  Hess = TRUE
)

Anova(model_4way, type = 3)




# --- Test Scandinavia separately ---
# Fit the CLMM
model <- clmm("delta ~ songfunction * PMmanip * sex  + (1|participant)" , data = scandinavia)
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

