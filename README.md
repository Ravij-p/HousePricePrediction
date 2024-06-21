

---

# California Housing Price Prediction

## Overview

This project involves predicting California housing prices using an XGBoost regression model. The dataset used is the California Housing dataset from the 1990 U.S. Census, which contains information about various factors influencing house prices. The model is trained to predict the median house value for California districts.

## Dependencies

Make sure you have the following dependencies installed:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install these dependencies using pip:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Dataset

The dataset used is the California Housing dataset, which can be loaded using the `sklearn.datasets.fetch_california_housing` function. The dataset includes the following features:
- `MedInc`: median income in block group
- `HouseAge`: median house age in block group
- `AveRooms`: average number of rooms per household
- `AveBedrms`: average number of bedrooms per household
- `Population`: block group population
- `AveOccup`: average number of household members
- `Latitude`: block group latitude
- `Longitude`: block group longitude

The target variable is `MedHouseVal`, which represents the median house value for California districts.

## Steps to Run the Project

1. **Import Dependencies**: Import necessary libraries including numpy, pandas, matplotlib, seaborn, scikit-learn, and xgboost.

2. **Load the Dataset**: Load the California Housing dataset using the `fetch_california_housing` function.

3. **Prepare the Data**: 
    - Create a pandas DataFrame with the dataset.
    - Add the target variable `MedHouseVal` to the DataFrame.

4. **Exploratory Data Analysis (EDA)**:
    - Calculate the correlation matrix and visualize it using a heatmap.

5. **Split the Data**: 
    - Separate the features (X) and the target variable (y).
    - Split the data into training and testing sets using `train_test_split`.

6. **Train the Model**:
    - Initialize and train the XGBoost regressor model on the training data.

7. **Evaluate the Model**:
    - Make predictions on the training data and calculate the R-squared error and mean absolute error.
    - Plot a scatter plot of actual vs. predicted prices for the training data.
    - Make predictions on the testing data and calculate the R-squared error and mean absolute error.
    - Plot a scatter plot of actual vs. predicted prices for the testing data.

8. **Make Predictions on New Data**:
    - Preprocess the input data.
    - Use the trained model to predict the house price for new input data.

## Results

- **Training Data**:
    - R-squared error: 0.94365
    - Mean absolute error: 0.19336

- **Testing Data**:
    - R-squared error: 0.83380
    - Mean absolute error: 0.31086

## Notes

- Ensure that all dependencies are installed before running the project.
- The dataset is derived from the 1990 U.S. Census and is available from the StatLib repository.

## License

This project is open-source and free to use.

---

