import pandas as pd #import pandas library
import seaborn as sns #import seaborn library used for data visualization and plotting
import matplotlib.pyplot as plt #import matplotlib library that seaborn builds on for plotting
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #import train_test_split function from sklearn library to split data into training and testing sets
from sklearn.metrics import mean_squared_error, r2_score #import mean_squared_error and r2_score from sklearn library to evaluate the model performance
import joblib #job library library for saving and loading python objects
from pydantic import BaseModel, Field #import to define data's structure
import numpy as np #numpy for numeric arrays
from fastapi import FastAPI #imports fastAPI

model = joblib.load("house_price_model.pkl") #loads model from earlier into model object

class HouseFeatures(BaseModel): #define expected input data format
    TotalSQFT: float
    HouseAge: int = Field(..., alias="House Age")
    GarageArea: float = Field(..., alias="Garage Area")
    TotalBath: int

app = FastAPI() #create FastAPI instance
@app.get("/") #GET endpoint intended to test if API is working
def home(): #returns JSON response confirming API is live
    return {"message": "House Predictor API is live"}

@app.post("/predict") #POST endpoint '/predict' defined, receives house features and returns predicted price
def predict_price(features: HouseFeatures): 
    #convert inputs into 2D numpy array
    input_data = np.array([[features.TotalSQFT,
                            features.HouseAge,
                            features.GarageArea,
                            features.TotalBath]])
    #use model to predict sale price for input data
    prediction = model.predict(input_data)[0] # [0] gets scalar from array
    return {"predicted_sale_price": round(prediction, 2)} #JSON Response, returns prediction rounded to 2 decimal places