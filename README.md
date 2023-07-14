# Prediction-of-Product-Sales
## This project will be a sales prediction for food items sold at various stores. The goal of this is to help the retailer understand the properties of products and outlets that play crucial roles in increasing sales.


**Ashik Sathiya**: 

### Business problem:

The retailer aims to improve sales performance for food items across their various stores. To achieve this, they seek to develop a sales prediction model that can provide valuable insights into the factors that significantly influence sales. By understanding the properties of both the products and the outlets that play crucial roles in driving sales, the retailer can make informed decisions to optimize their operations and increase revenue.



### Data:
**Variable Name**	Description

**Item_Identifier**-	Product ID

**Item_Weight**-	Weight of product

**Item_Fat_Content**-	Whether the product is low-fat or regular

**Item_Visibility**-	The percentage of total display area of all products in a store allocated to the particular product

**Item_Type**-	The category to which the product belongs

**Item_MRP**-	Maximum Retail Price (list price) of the product

**Outlet_Identifier**-	Store ID

**Outlet_Establishment_Year**-	The year in which store was established

**Outlet_Size**	The size of the store in terms of ground area covered

**Outlet_Location_Type**-	The type of area in which the store is located

**Outlet_Type**-	Whether the outlet is a grocery store or some sort of supermarket

**Item_Outlet_Sales**-	Sales of the product in the particular store. This is the target variable to be predicted.


## Methods
- Data preparation steps with explanation and justification for choices
- 

## Results

### Data Preprocessing

#### Data Cleaning Steps(Pre Split)
1) Removed duplicates: Eliminated any duplicate entries from the dataset to ensure each sales price observation is unique.

2) Resolved inconsistencies: Corrected any inconsistencies in the dataset, in this case conflicting values within the same variable.

4) Handled outliers: I examined and adjusted extreme or outlier values in the dataset, ensuring they don't significantly affect the analysis.

   
#### Column Transformer
1) Identify column types: Determine the column types in your dataset, such as ordinal, numeric, or categorical.

2) Define transformers: Create specific transformers for each column type, including SimpleImputer for missing values, StandardScaler for ordinal and numeric columns, OrdinalEncoder for ordinal columns, and OneHotEncoder for categorical columns.

3) Create a column transformer: Build a column transformer object and specify the transformers for each column type.

4) Define pipelines: Create separate pipelines for the ordinal, numeric, and categorical columns, including the appropriate transformers in each pipeline.

5) Fit and transform: Fit the column transformer on your dataset and use it to transform the columns accordingly.


### Visualizations


#### Distribution of Item Outlet Sales
<img width="704" alt="Screen Shot 2023-07-13 at 11 40 35 PM" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/4b76bb5b-ded4-49a7-9921-ecb475564ed1">

> Here we can see a histogram of the Item Outlet Sales. As we can see it is very much positively skewed with the highest count occuring to the left of the graph.


#### Item Outlet Sales by Outlet Type
<img width="714" alt="Screen Shot 2023-07-13 at 11 40 58 PM" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/2c082aad-f1fb-4d85-bb26-86215aa1cee0">

> Here we have a countplot for Item Outlet Sales by Outlet Type. This is a very insightful plot in that we are able to see which Outlet Type produces highest Sales. According to the plot Supermarket Type 3 produces the highest sales with Grocery Store performing far below that.


## Model

### Description
After evaluating three models (default linear regression, default random forest, and tuned random forest), it is evident that the tuned random forest model is the most suitable choice. It achieves the highest R^2 score on the test data, indicating superior accuracy in predicting sales prices for unseen data. Moreover, the tuned random forest strikes a balance by avoiding significant issues of overfitting or underfitting. In contrast, the default random forest model suffers from overfitting, while the default linear regression model exhibits substantial underfitting, resulting in less accurate predictions.

The final model was a Random Forest Model that was tuned using GridSearchCV. Grid search was used to fine tune the random forest model by exhaustively searching through a predefined parameter grid. The model was fitted with 5-fold cross-validation on a total of 240 candidate parameter combinations, resulting in 1200 fits. The best parameter configuration discovered by grid search includes a max_depth of 10, no restriction on max_features, 150 estimators, oob_score set to False, and warm_start set to False. These optimized parameters were determined to yield the best performance for the random forest model.



### Metrics

R-squared (R^2) is a metric that measures how well a regression model can predict the dependent variable based on other features. In our tuned random forest model, the R^2 value for the training data is 0.722, indicating that around 72.2% of the variation in the dependent variable is explained by the model. Although the R^2 value for the test data is 0.590, lower than the training data, it still suggests that the model captures a substantial portion of the relationships in the test data. However due to the R^2 value being greater in the training data, there is evidence that the model is overfitted and therefore performs poorly when exposed to unseen data. Overfitting occurs when a machine learning model performs exceptionally well on the training data but fails to perform accuratley to new, unseen data. 


The Mean Absolute Error (MAE) metric helps assess the average prediction error by measuring the absolute difference between predicted and actual values. In our model, the MAE value for the test data is 739.251 units, indicating that, on average, the model's predictions deviate from the actual values by around 739.251 units. This straightforward measure provides stakeholders with an understanding of the model's prediction accuracy, enabling them to assess reliability and make informed decisions based on the expected levels of accuracy and error.


MSE measures the average squared difference between the predicted and actual values. In the training data, the MSE is 821,866.774, indicating that, on average, the squared difference between the predicted and actual values is around 821,866.774 units. In the test data, the MSE is 1,130,000.683, suggesting a slightly higher average squared difference compared to the training data.


RMSE is the square root of the MSE and provides a measure of the average magnitude of the prediction errors. In the training data, the RMSE is 906.569, which means, on average the prediction errors have a magnitude of approximately 906.569 units. In the test data, the RMSE is 1,063.015, indicating slightly higher prediction errors compared to the training data.






## Recommendations:

Model is not yet ready for deployment, further tuning and training is required to improve overall accuracy.

## Limitations & Next Steps

In order to improve our model accuracy when exposing it to unseen data the following steps can be taken.

**Further Data Collection:** Gathering additional data or expanding the dataset to become a larger and more diverse dataset can provide the model with a newer set of patterns and relationships to learn from, and potentially improve its predictive accuracy.

**Hyperparameter Tuning:** Fine-tune the model's hyperparameters using techniques like grid search or random search. Experiment with different combinations of hyperparameters to find the optimal configuration that improves the model's performance on both training and test data. This process can help address any underfitting or overfitting issues and further enhance the model's predictive capabilities.

**Feature Engineering:** Exploring additional features or creating new meaningful features from existing features  could enhance the predictive power of the model. 




### For further information


For any additional questions, please contact **ashik.sathiya@gmail.com**
