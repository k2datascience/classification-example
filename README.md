# Banking Classification Project

## Prompt

Client: **Briar's Community Bank**

We are having a campaign. We want people to subscribe to a term deposit. We have been recording our success rate on clients for a while. Can you tell us who we can convert based on their financial data?

We have their age, job, employees, their loans, if they have any credit defaults, when we called them, their education, etc., and we also recorded if our call was successful or not. What can we learn? What would be useful for us?

## Challenge

Predicting who who subscribe to a term deposit was a unique problem. The majority of the data we had was demographic or about the general economy. The bank-related data was mostly focused on loans individuals had taken out, for example, a mortgage. There was not many features related to savings account balances or investments.

Standard assumptions are that older, educated and individuals with families are more prone to savings. They have more responsibilites, more assets and more things to worry about than younger, unmarried individuals. Let's hope these can serve to help the model's predictability performance.

## Data

- 40,000+ individuals from a direct marketing campaign of a Portuguese banking institution
- It contains relevant demographic data, financial data as well economic indicators at the time of contact

## Exploratory Data Analysis

I explored various feaures. Below are a preview of the univariate and bivariate visualizations. Please check the accompanying Jupyter notebooks for more detailed EDA.

**Missing Data**

One of the first things you have to do with any dataset is examine missing data and decide on a removal or imputation strategy. At first glance, there were no NaN values when importing the CSV into pandas. However, on closer inspection, several columns had a large number of string values of "unknown". According to the data descriptions, another column had the value "999" when there was no previous call. These two columns were surely going to wreak havoc on any modeling. I decided to impute all these based on the median column values.

![missing data](images/missing_data.png)

There seemed to be a large number of unknowns in the column designating whether the individual had credit in default. It seemed odd that the bank would not have this information on file. Maybe they did not have permission to do a credit look-up. Another possibility for so many unknowns is that this is a sensitive financial matter. People are more likely to hide the fact that they are in financial trouble vs. reporting their marital status.

**Univariate Analysis**

There were only a select few of numerical variables, while binary yes/no or multi-category. Ages in particular had a fairly normal distribution, but has a heavy right skew. There is a large drop-off at 60, almost an inverse replica to the step up from 20 to 25 on the other-end of the age spectrum.

![histogram](images/ages.png)

**Multivariate Analysis**

There did not seem to be any significant relationships between the bank data. In the following correlation matrix, the warmer cells are mostly social and economic indicators released by the government. It would make sense that these numbers are highly correlated.

![pairplot](images/matrix.png)

## Data Preprocessing

After an inital round of EDA, I went on to make sure the data was ready to be fed into any learning algorithm. These were the steps I took:

- Median imputation for 'unknown' values
- Convert yes/no to 1/0
- Get dummy variables for month, day of week, job, marital status, education, type of contact
- Remove any features that have look-forward bias
- Brought features on the same scale

**Principal Component Analysis**

I extracted out the continuous numerical variables and decided to run PCA for two reasons. I wanted to visualize the first two components as well as the first three. Also, I wanted to see if the first few components would be able to retain most of the variance.

![pca visualization](images/pca_visual.png)

The class highlighting on the first two principal components shows no discernable difference between individuals that signed up for term deposits and those that did not.

![pca_variance](images/pca_.png)

The cumulative explained variance does provide useful insight. If this was a heavy big data problem, we would be able to remove the last 4 components and still retain 95% of the variance. We are not memory or time constrained in this project, so we will keep the variables the way they are for now.

## Modeling

Started off with a baseline model, which had a fairly high accuracy of **87.6%**. Due to the class imbalance of a lot of No's, this makes sense. Let's see how a range of classification algorithsm compared.

**Comparing cross-validated performance**

![cv](images/cv.png)

I decided to focus on the Logistic Regression classifier and further tune it as well as examine it for over- and under-fitting.

**Validation curve**

![validation](images/validation_curve
.png)

**Confusion matrix**

![confusion](images/confusion.png)

It turns out my model had an incredibly low Recall at **20.5%**. In this situation, correctly identifying the "Yes" for term deposits is the most important goal of the project. If I am unable to bring Recall up to an acceptable threshold, a "highly accurate" model is pointless.

## Lessons Learned

I learned several concepts during this project. I started to examine the nuances of linear and non-linear based algorithms. I tackled the bias-variance trade-off, but will still need to further examine how to take on the precision-recall trade-off. Even though I made solid progress on the pipeline, it opened up the door for many more questions than I originally wanted to answer.

## Further Analysis

If I had more time with the project, I would complete the following tasks:

- Construct many more features dealing with interactions between 2 variables
- Explore gradient boosting
- Play with precision-recall curve

## Code Information

Feel free to browse the Jupyter notebooks in this repository. I used standard Python packages and visualization libraries. The data set is stored on [UCI's repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#).
