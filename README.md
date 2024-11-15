LINK FOR EC2: http://3.142.76.83:3838/AVM_MGSC410/

Kabir Mann
Taher Rastkar

Overview:
The California Real Estate Price Predictor is a Shiny app that uses an XGBoost machine learning model to estimate home prices in California. It takes into account various property, location, and neighborhood characteristics to provide an estimated home value.

Features:
Price Prediction: Predicts home prices based on user-input property features.
Area Statistics: Displays neighborhood data, including crime index and median income.
Location Overview: Shows the property's location on a map for spatial context.

How to Use:
Input Property Details: Enter information about the property, such as address, number of bedrooms and bathrooms, lot size, home type, and location-related features.
Predict Price: Click the Predict Price button to get an estimated property value.
View Results: The predicted home price, neighborhood statistics, and location map will be displayed for easy reference.

Files Used:
homes_data_final.csv: Dataset of historical housing data.
xgboost_model.rds: Pre-trained XGBoost model used for price prediction.
preprocessing_recipe.rds: Preprocessing steps applied to user inputs before model prediction.
crime_index_by_city.csv: Data on crime index by city.
median_income_zipcode.csv: Data on median income by ZIP code.

Requirements:
The app requires the following R packages:
shiny, shinydashboard, xgboost, dplyr, scales, leaflet, DT, tidygeocoder, recipes

Notes:
Make sure to input a valid California city and ZIP code for accurate predictions.
The map view and neighborhood statistics depend on the provided address and ZIP code accuracy.
