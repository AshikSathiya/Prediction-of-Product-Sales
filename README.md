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

# Feature Importance
![feature importance](https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/78ce58a5-b813-46bf-8606-ce06d5aa4d01)
Here we can see the most important features from using our Random Forest Model. The 5 most important features are Item_MRP, Item_Visibility, Outlet_Type_Supermarket Type3, Outlet_Type_Supermarket Type1, Item_Weight


# Coeffecients
![coefficients](https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/357682d8-66a3-4377-9c88-04f92ede0194)
The retailer aims to improve sales performance for food items across their various stores. To achieve this, they seek to develop a sales prediction model that can provide valuable insights into the factors that significantly influence sales.

We found the three most impactful features. We are prediciting for the sales of the product in the particular store.

The three Outlet Type Supermarkets all had a very significant impact on Sales Production. Each of these SuperMarkets impact Sales Production positvely to varying degrees.

Outlet_Type_Supermarket Type3 increases item sales production by 3532.3 items. Outlet_Type_Supermarket Type1 increases item sales production by 1547.3 items. Outlet_Type_Supermarket Type2 increases item sales production by 1161.9 items.


# SHAP Barplot

<img width="885" alt="SHAP barplot" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/7e4fcec8-7f8c-4783-89f1-ea7978ec044f">

The most important features according to SHAP and the original feature importances differ. The top 5 most important features according to SHAP are Item_MRP, Outlet_Type_SuperMarket Type1, Outlet_Type_SuperMarket Type3, Item_Visibility, and Outlet_Establishment_Year.

According to our original important features, the most important features are Item_MRP, Item_Visibility, Outlet_Type_SuperMarket Type 3, Outlet_Type_SuperMarket Type1, and Item_Weight.

The only similiarity between the two is Item_MRP being the most important

# SHAP Dotplot

<img width="901" alt="SHAP dotplot" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/d9e6f24b-ad07-4efe-bb65-879ddd5d57fe">

Item_MRP is the Maximum Retail Price (list price) of the product. The plot is pretty consistent with there being a relatively equal amount of high and low feature values for Item_MRP. This could inidcate that a higher MRP leads to a higher impact on Sales of Items.

Outlet_Type_SuperMarket Type1 tells us that there is a higher Sales Output Items sold in Outlet_Type_SuperMarket Type1.

This is more evident Outlet_Type_SuperMarket Type3 since there is a greater difference between high and low features.

# Local SHAP Plot

I was curious to see how how much the Outlet Type affects Sales since it is one of the most important features. I first set the visibility of the product to be above average. I then compared the Outlet Type 1 to the Non OUtlet Type 1 stores. Finally I wanted to see what facotrs differed for the places that had High Sales and Low Sales.

So to summarise the 4 Groups we will be comparing are
* High Sales
   * Outlet Type 1 High Sales
   * Non Outlet Type 1 High Sales

* Low Sales
   * Outlet Type 1 Low Sales
   * Non Outlet Type 1 Low Sales

 ## Outlet Type 1 High Sales

 <img width="965" alt="Type1_HighSales" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/fcc9c946-8620-45f4-a544-6d69d288b1bb">

As you can see, Item_MRP, Outlet_Type_Supermarket Type1 are the two prominent facotrs that led to a higher Sales Price. However, Outlet_Type_Supermarket Type3 seems to be the biggest factor to lower the sales price.

## Non Outlet Type 1 High Sales
<img width="962" alt="NonType_HighSales" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/b58cf99d-9822-4f5a-9b50-13f261e21286">

The only factor that increased the sales is the Outlet Type Being Type 3. This was also the factor that lowered the above force plot. The Outlet Type 1 seems to have lowered the Sales and so this might tell us that Type 3 Supermarkets have the highest sales.


## Outlet Type 1 Low Sales
<img width="964" alt="Type 1_Low" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/d3b6b311-4040-47ab-acd2-312e7d8dd546">

According to this plot the only factor that contributed to a higher sales price is Outlet_Type_Supermaket Type 1. However the Item_MRP greatly impacted the sales, bringing it down.

## Non Outlet Type 1 Low Sales
<img width="977" alt="Non_Low" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/3d421d4a-e15c-41a6-a9dc-6dc72b4422cd">

Here, Item_MRP is the only factor that is contributing to a higher sales. However interestingly enough Outlet_Type_Supermarket Type 1 and Outlet_Type_Supermarket Type 3 being 0 were the biggest factors that brought sales down.


# LIME Plots

## Outlet Type 1 High Sales
<img width="880" alt="LIME_Type 1_High" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/4db652c8-2a18-4a08-a1c4-2ff9300c8b3e">

The primary negative factor that brings down the sales is the fact Outlet_Type Supermarket is not Type 3. In contrast Outlet_Type Supermarket Type 1 is the biggest factor that increased the sales price.

## Non Outlet Type 1 High Sales
<img width="890" alt="LIME_Non_High" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/f01d6b52-e811-4507-bff5-75fc1519d8b1">

Here, we have the exact opposite of the aboves plots primary facotrs affecting the prediciton. The OUtlet Type1 being 0 greatly hurt the sales, however the Outlet Type3 being 1 increased the Sales alot. This shows the impact these two metrics have on sales and howmuch the type of outlet affects the sales.


## Outlet Type 1 Low Sales
<img width="867" alt="LIME_Type1_Low" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/f308c4cd-4e8e-477e-bf3b-07632d746dc3">

In this plot the Item_MRP stands out as it is lower than 98.82 which brings down Sales. Once again the Outlet Type affects the sales greatly.

## Non Outlet Type 1 Low Sales
<img width="871" alt="LIME_Non_Low" src="https://github.com/AshikSathiya/Prediction-of-Product-Sales/assets/92455762/26d1a5f0-3e36-459b-9e03-4940aabbc5ea">

Here, we have an outlet type that is neither Type 1 or Type 3 and this greatly hurts the sales. The only positve factor is the Item_MRP.

### For further information

For any additional questions, please contact **ashik.sathiya@gmail.com**
