# Online Course Engagement and Dropout Prediction

## Project Overview

This project analyzes learner activity on an online education platform and builds a machine learning model to predict which users may not complete a course.

The main objective is to support early learner-retention activities. The model can help an e-learning platform rank users by predicted dropout risk and prioritize interventions such as automated reminders, additional learning materials, or mentor support.

The project focuses on the positive class:

* `Dropout = 1` — the user did not complete the course,
* `Dropout = 0` — the user completed the course.

The model is intended as a decision-support tool rather than an automatic decision-maker.

## Dataset

The data comes from the Kaggle **Predict Online Course Engagement Dataset**.

The original dataset contains:

* 9,000 rows,
* 9 columns,
* no missing values,
* 877 duplicated records.

After removing duplicates, 8,123 observations remain.

### Variables

* `UserID` — user identifier,
* `CourseCategory` — course category,
* `TimeSpentOnCourse` — time spent on the course,
* `NumberOfVideosWatched` — number of watched videos,
* `NumberOfQuizzesTaken` — number of completed quizzes,
* `QuizScores` — quiz score,
* `CompletionRate` — percentage of course content completed,
* `DeviceType` — device used for learning,
* `CourseCompletion` — whether the course was completed.

A new target variable, `Dropout`, is created from `CourseCompletion`.

## Project Workflow

### 1. Preliminary Data Analysis

The dataset is inspected to check:

* its dimensions,
* data types,
* descriptive statistics,
* missing values,
* course-category distribution,
* duplicated records.

The duplicated rows are removed before further analysis.

After cleaning:

* approximately 56% of users did not complete the course,
* approximately 44% completed it.

### 2. Exploratory Data Analysis

The analysis examines how course completion is related to:

* time spent on the course,
* number of watched videos,
* quiz scores,
* course category.

The main observations are:

* users who spend more time on the course complete it more often,
* the completion rate increases with the number of watched videos,
* users with higher quiz scores complete the course more often,
* differences between course categories are small, with completion rates of approximately 42–45%.

These results suggest that learner behavior is more informative than course category.

### 3. Distribution and Outlier Analysis

The numerical engagement variables are analyzed using:

* descriptive statistics,
* comparisons between means and medians,
* skewness,
* histograms,
* boxplots,
* the IQR criterion.

The analyzed variables are:

* `TimeSpentOnCourse`,
* `NumberOfVideosWatched`,
* `NumberOfQuizzesTaken`,
* `QuizScores`.

Their skewness values are close to zero and their means are similar to their medians, indicating approximately symmetric distributions.

### 4. Statistical Tests

#### Mann–Whitney U tests

The Mann–Whitney U test is used to compare completers and dropouts for the four numerical engagement variables.

All four variables show statistically significant differences between the groups:

| Feature              | Completed mean | Dropout mean |
| -------------------- | -------------: | -----------: |
| Time spent on course |          56.58 |        45.93 |
| Videos watched       |          11.77 |         8.88 |
| Quizzes taken        |           6.20 |         4.36 |
| Quiz scores          |          80.03 |        71.22 |

These results show that completers have higher average engagement across all analyzed numerical variables.

#### Chi-square tests

Chi-square tests are used to check whether dropout is associated with:

* course category,
* device type.

The results do not show statistically significant associations:

* `CourseCategory`: p-value = 0.5706,
* `DeviceType`: p-value = 0.4482.

This supports the conclusion that behavioral engagement variables are more informative than these categorical variables.

## Data Leakage Consideration

`CompletionRate` is strongly related to the final course outcome.

However, it may contain information collected close to the end of the learning process. Using it for early dropout prediction could therefore introduce data leakage.

For this reason, the main model excludes `CompletionRate`.

An additional comparison is performed later to show how much model performance improves when this variable is included.

## Data Preparation

The model uses the following features.

### Categorical features

* `CourseCategory`,
* `DeviceType`.

### Numerical features

* `TimeSpentOnCourse`,
* `NumberOfVideosWatched`,
* `NumberOfQuizzesTaken`,
* `QuizScores`.

`UserID` is not used as a predictive feature. It is used only to split the data.

The train-test split is performed with `GroupShuffleSplit`, and no user appears in both sets.

### Preprocessing

* numerical variables are standardized with `StandardScaler`,
* categorical variables are transformed with `OneHotEncoder`,
* preprocessing and model training are combined in scikit-learn pipelines.

## Models

Three classification models are compared:

1. `DummyClassifier` — baseline model,
2. `LogisticRegression` — interpretable linear model,
3. `RandomForestClassifier` — non-linear ensemble model.

## Evaluation Metrics

The models are evaluated using:

* accuracy,
* dropout precision,
* dropout recall,
* dropout F1-score,
* ROC AUC,
* PR AUC.

The most important metrics are precision, recall, and PR AUC because the main objective is to identify users at risk of dropping out.

## Model Results

| Model               |  Accuracy | Precision |    Recall |        F1 |   ROC AUC |    PR AUC |
| ------------------- | --------: | --------: | --------: | --------: | --------: | --------: |
| Dummy Classifier    |     0.562 |     0.562 |     1.000 |     0.720 |     0.500 |     0.562 |
| Logistic Regression |     0.712 |     0.764 |     0.707 |     0.734 |     0.792 |     0.822 |
| Random Forest       | **0.746** | **0.807** | **0.722** | **0.762** | **0.862** | **0.872** |

Random Forest achieves the best overall performance.

At the default threshold of 0.5:

* approximately 81% of users flagged as high risk actually drop out,
* approximately 72% of all dropouts are identified.

## Threshold Analysis

The notebook evaluates thresholds from 0.10 to 0.90.

The results show the expected trade-off:

* lower thresholds identify more future dropouts but flag more users,
* higher thresholds improve precision but miss more dropouts.

For example:

* at threshold 0.40, recall is approximately 0.98 and precision is approximately 0.73,
* at threshold 0.50, recall is approximately 0.72 and precision is approximately 0.81,
* at threshold 0.60, recall is approximately 0.56 and precision is approximately 0.93.

The final threshold should depend on intervention cost, available support capacity, and the relative cost of false positives and false negatives.

## Feature Importance

Permutation importance is calculated using PR AUC as the scoring metric.

The most important variables are:

1. `NumberOfQuizzesTaken`,
2. `QuizScores`,
3. `NumberOfVideosWatched`,
4. `TimeSpentOnCourse`.

`CourseCategory` and `DeviceType` provide little additional predictive value.

Feature importance describes predictive usefulness and does not establish causality.

## Comparison with CompletionRate

A separate Logistic Regression model is trained with `CompletionRate`.

Its PR AUC increases from approximately:

* 0.82 without `CompletionRate`,
* to 0.87 with `CompletionRate`.

Although performance improves, this version is not treated as the main result because `CompletionRate` may introduce data leakage in an early-warning scenario.

## Recommendations

1. Use the model to rank users according to their predicted dropout risk.

2. Consider earlier support for high-risk learners, who are often characterized by fewer watched videos, fewer completed quizzes, lower quiz scores, and less time spent on the course.

3. Use low-cost automated reminders for a broader risk group and reserve resource-intensive mentor support for users with the highest predicted risk.

4. Select the classification threshold according to intervention capacity and the preferred balance between precision and recall.

5. Test the effectiveness of specific interventions through an A/B experiment, because the current analysis identifies associations and predictions rather than causal effects.

## Key Conclusions

* Learner engagement is more strongly associated with course completion than course category or device type.
* Users who complete courses show higher average engagement across all analyzed numerical variables.
* Random Forest provides the best predictive performance among the evaluated models.
* The model can help prioritize users for retention activities.
* `CompletionRate` improves performance but may not be suitable for early prediction because of data-leakage risk.
* The analysis is predictive and observational; it does not prove that increasing engagement directly causes course completion.

## Technologies

* Python
* pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* statsmodels
* scikit-learn
* Jupyter Notebook

## Limitations

* The data is observational, so the results do not establish causal relationships.
* The model is evaluated on a single train-test split.
* The dataset may not represent learners from other platforms or time periods.
* The effectiveness of retention interventions is not measured directly.
* Model performance may change if learner behavior, course design, or data collection changes.

## Possible Future Improvements
* repeated cross-validation,
* probability calibration,
* SHAP-based interpretation,
* time-based engagement features,
* model monitoring and data-drift detection,
* deployment as an API or dashboard,
* A/B testing of learner-support interventions.

## Author

Artur Strąg
