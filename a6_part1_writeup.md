# Assignment 6 Part 1 - Writeup

**Name:** __Youssef Ahmed Yahia_____________  
**Date:** ___11/21/2025____________

---

## Part 1: Understanding Your Model

### Question 1: R² Score Interpretation
What does the R² score tell you about your model? What does it mean if R² is close to 1? What if it's close to 0?

**YOUR ANSWER:**

The R² score measures how much of the variance in the target variable (test scores) is explained by the model's features (hours studied). In this run the model's R² ≈ 0.9797, which means the model explains about 97.97% of the variance in scores — a very strong fit. If R² is close to 1 it means the model explains almost all the variance (very good fit). If R² is close to 0 it means the model explains almost none of the variance (poor fit).

---

### Question 2: Mean Squared Error (MSE)
What does the MSE (Mean Squared Error) mean in plain English? Why do you think we square the errors instead of just taking the average of the errors?

**YOUR ANSWER:**

MSE is the average of the squared differences between predicted and actual values. In plain English it tells us how far, on average (squared), the predictions are from the actual test scores. In this run MSE ≈ 13.58 and RMSE ≈ 3.69, so predictions are on average about 3.69 points away from the actual scores.

We square the errors so that negative and positive errors don't cancel out and to penalize larger errors more strongly than smaller ones (squaring increases the impact of larger deviations).

---

### Question 3: Model Reliability
Would you trust this model to predict a score for a student who studied 10 hours? Why or why not? Consider:
- What's the maximum hours in your dataset?
- What happens when you make predictions outside the range of your training data?

**YOUR ANSWER:**

The maximum hours in the dataset is 9.6, so predicting for 10 hours is slightly outside the observed range (extrapolation). While the model appears very accurate within the data range (high R² and low RMSE), predictions outside the range are less reliable because linear regression assumes the learned relationship continues beyond observed data. A prediction for 10 hours may be reasonable if the relationship truly remains linear, but it's safer to be cautious and not fully trust extrapolated values.

---

## Part 2: Data Analysis

### Question 4: Relationship Description
Looking at your scatter plot, describe the relationship between hours studied and test scores. Is it:
- Strong or weak?
- Linear or non-linear?
- Positive or negative?

**YOUR ANSWER:**

The relationship appears strong, approximately linear, and positive: as hours studied increases, test scores tend to increase.

---

### Question 5: Real-World Limitations
What are some real-world factors that could affect test scores that this model doesn't account for? List at least 3 factors.

**YOUR ANSWER:**
1. 
2. 
3. 

1. Prior knowledge or baseline ability of the student
2. Quality of instruction or curriculum differences
3. Sleep, stress, or test-day conditions (health, anxiety)
4. Test difficulty or grading variability

---

## Part 3: Code Reflection

### Question 6: Train/Test Split
Why do we split our data into training and testing sets? What would happen if we trained and tested on the same data?

**YOUR ANSWER:**

We split data so we can measure how well the model generalizes to unseen data. The training set is used to fit the model and the test set evaluates its performance. If we trained and tested on the same data we would get overly optimistic performance (the model could simply memorize the data), and we would not know how well it performs on new examples.

---

### Question 7: Most Challenging Part
What was the most challenging part of this assignment for you? How did you overcome it (or what help do you still need)?

**YOUR ANSWER:**

Understanding how the evaluation metrics relate to real-world error was the most challenging part. I reviewed the in-class example and compared R², MSE, and RMSE to get an intuition for how they quantify performance; plotting predictions vs actuals also helped visualize errors. If I need more help, I'd ask for feedback on interpreting model errors and when to trust extrapolated predictions.

---

## Part 4: Extending Your Learning

### Question 8: Future Applications
Describe one real-world problem you could solve with linear regression. What would be your:
- **Feature (X):** 
- **Target (Y):** 
- **Why this relationship might be linear:**

**YOUR ANSWER:**

Example: Predicting house price based on living area.
- **Feature (X):** Square footage (living area in sq ft)
- **Target (Y):** House sale price (USD)
- **Why this relationship might be linear:** Price often increases roughly proportionally with area (larger houses generally cost more), so a linear model can capture the primary relationship, though other factors (location, condition) would improve accuracy.

---

## Grading Checklist (for your reference)

Before submitting, make sure you have:
- [ ] Completed all functions in `a6_part1.py`
- [ ] Generated and saved `scatter_plot.png`
- [ ] Generated and saved `predictions_plot.png`
- [ ] Answered all questions in this writeup with thoughtful responses
- [ ] Pushed all files to GitHub (code, plots, and this writeup)

---

## Optional: Extra Credit (+2 points)

If you want to challenge yourself, modify your code to:
1. Try different train/test split ratios (60/40, 70/30, 90/10)
2. Record the R² score for each split
3. Explain below which split ratio worked best and why you think that is

**YOUR ANSWER:**
