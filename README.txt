# Smart Cost Estimation for Construction Planning
### CNST Group 16 - Final Project

## Project Overview
This project focuses on developing an AI-powered smart cost estimator for construction planning. 
Using a dataset containing 1500 real-world construction records from 2004–2023, we built predictive models to estimate final construction costs based on key project features like type, floor area, duration, and contractor grade.

## Features
- Data Preprocessing and Quality Checking
- Feature Engineering and Analysis
- Machine Learning Models:
  - Linear Regression
  - Random Forest
  - XGBoost
  - Neural Networks
- What-if Scenario Simulation
- Interactive Area-wise Cost Visualization
- Streamlit Web Application for User Interaction

## Setup Instructions
1. Extract the `Smart_Cost_Estimator.zip` file.
2. Navigate to the extracted folder.
3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

## Requirements
- Python 3.8 or above
- Streamlit
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- dotenv

## How to Use
- Open the Streamlit app.
- Input project details like Project Type, Area, Floors, Location, and Contractor Grade.
- Get the estimated construction cost and view area-wise cost graphs.
- Use the What-if Simulator to modify parameters and observe cost impacts dynamically.

## Project Team
CNST Group 16

## License
This project is for educational purposes only.

