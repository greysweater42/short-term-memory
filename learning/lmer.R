df <- data.frame(
       person = factor(c(0, 0, 0, 1, 1, 1, 2, 2, 2, 2)),
       beers = c(2, 6, 5, 4, 7, 8, 2, 4, 3, 4),
       alcohol_concentration = c(0.3, 1.1, 0.9, 1.1, 2, 1.9, 0.7, 1.4, 1.2, 1.3)
)

# Z matrix
model.matrix(~ -1 + person, df)


m <- lm(alcohol_concentration ~ -1 + person + beers, df)
summary(m)

plot_model <- function(m) {
       colors <- c("red", "green", "blue")
       palette(colors)
       plot(alcohol_concentration ~ beers, df, col = person)
       for (p in 1:3) abline(a = coef(m)[p], b = coef(m)[4], col = colors[p])
}
plot_model(m)


# Why would we use mixed-effects model instead of linear regression?


# for unbalanced data
library(data.table)

summer_n <- 20
winter_n <- 10
df <- data.table(
       season = factor(c(rep("summer", summer_n), rep("winter", winter_n))),
       weight = c(rnorm(summer_n, 82, 2), rnorm(winter_n, 87, 3))
)
print(rbind(head(df, 3), tail(df, 3)))

plot(weight ~ season, df)

m_lr <- lm(weight ~ 1, df)
summary(m_lr)
mean(df$weight)

library(lme4)
m_mr <- lmer(weight ~ 1 + (1 | season), df)
summary(m_mr)

mw <- mean(df[season == "winter", weight])
ms <- mean(df[season == "summer", weight])
mean(c(mw, ms))

library(PBImisc)
dementia <- data.table(dementia)
head(dementia)

m_d <- lmer(demscore ~ 1 + age + sex + (1 | study), data = dementia)
summary(m_d)
mean(dementia[sex == "Male" & age == "<60", demscore])
mean(dementia[sex == "Male" & age == ">=60", demscore])
mean(dementia[sex == "Female" & age == "<60", demscore])
mean(dementia[sex == "Female" & age == ">=60", demscore])

summary(lmer(demscore ~ age * sex + (1 | study), data = dementia))
summary(lmer(demscore ~ (sex | study), data = dementia))
summary(lmer(demscore ~ age * sex + (age * sex | study), data = dementia))


colnames(iris) <- gsub("\\.", "_", tolower(colnames(iris)))
lm(sepal_length ~ petal_length * species, data = iris)

# understanding lmer:
# - understanding syntax
# colnames(iris) <- gsub("\\.", "_", tolower(colnames(iris)))
# lm(sepal_length ~ petal_length * species, data=iris)

# - understanding syntax
# m_mr <- lmer(weight ~ 1 + (1 | season), df)


# - understanding syntax
summary(lmer(demscore ~ 1 + (age | study), data = dementia))
summary(lmer(demscore ~ 1 + (study | age), data = dementia))
# which means that age's variance is "embedded" into study's variance
# I keep writing `1 + ` in formulas, because it is easy to forget abouit the intercept

var(dementia[study == "Poland Gdansk" & age == ">=60", demscore])
var(dementia[study == "Poland Gdansk" & age == "<60", demscore])
dementia[, .(var(demscore)), study]

# each observation is a sum of intercept, error, deviation from study and deviation from age+study
