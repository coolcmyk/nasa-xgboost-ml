
import numpy as np
import pandas as pd
import joblib
import numpy as np
import requests
from google.cloud import functions_v1

from flask import Flask, jsonify

app = Flask(__name__)

# Load models
models = {
    'rf': joblib.load('rf_model.joblib'),
    'sw': joblib.load('sw_model.joblib'),
    'sr': joblib.load('sr_model.joblib'),
    'temp': joblib.load('temp_model.joblib')
}

fruit_models = {
    'banana': joblib.load('banana_model.joblib'),
    'mango': joblib.load('mango_model.joblib'),
    'apple': joblib.load('apple_model.joblib'),
    'orange': joblib.load('orange_model.joblib'),
    'papaya': joblib.load('papaya_model.joblib'),
    'pineapple': joblib.load('pineapple_model.joblib'),
    'watermelon': joblib.load('watermelon_model.joblib'),
    'avocado': joblib.load('avocado_model.joblib'),
    'chili': joblib.load('chili_model.joblib'),
    'onion': joblib.load('onion_model.joblib'),
    'spinach': joblib.load('spinach_model.joblib'),
    'carrot': joblib.load('carrot_model.joblib'),
    'tomato': joblib.load('tomato_model.joblib'),
    'cabbage': joblib.load('cabbage_model.joblib'),
    'potato': joblib.load('potato_model.joblib'),
    'garlic': joblib.load('garlic_model.joblib'),
    'eggplant': joblib.load('eggplant_model.joblib'),
    'cucumber': joblib.load('cucumber_model.joblib')
}

class ThresholdTester:
    def __init__(self, lower_threshold, upper_threshold):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def soilTester(self, soil_water):
        return self.lower_threshold <= soil_water <= self.upper_threshold

    def solarTester(self, solar_radiation):
        return self.lower_threshold <= solar_radiation <= self.upper_threshold

    def precipitationTester(self, precipitation):
        return self.lower_threshold <= precipitation <= self.upper_threshold

    def temperatureTester(self, temperature):
        return self.lower_threshold <= temperature <= self.upper_threshold

    def humidityTester(self, humidity):
        return self.lower_threshold <= humidity <= self.upper_threshold

    def ndviTester(self, ndvi):
        return self.lower_threshold <= ndvi <= self.upper_threshold

    def eviTester(self, evi):
        return self.lower_threshold <= evi <= self.upper_threshold

    def elevationTester(self, elevation):
        return self.lower_threshold <= elevation <= self.upper_threshold

    def slopeTester(self, slope):
        return self.lower_threshold <= slope <= self.upper_threshold

    def classifyTester(self, sw, sr, temp, humid, ndvi, evi, elev, slope, precip):
        trueCounter = 0
        falseCounter = 0

        if self.soilTester(sw):
            trueCounter += 1
        else:
            falseCounter += 1

        if self.solarTester(sr):
            trueCounter += 1
        else:
            falseCounter += 1

        if self.precipitationTester(precip):
            trueCounter += 1
        else:
            falseCounter += 1

        if self.temperatureTester(temp):
            trueCounter += 1
        else:
            falseCounter += 1

        if self.humidityTester(humid):
            trueCounter += 1
        else:
            falseCounter += 1

        if self.ndviTester(ndvi):
            trueCounter += 1
        else:
            falseCounter += 1

        if self.eviTester(evi):
            trueCounter += 1
        else:
            falseCounter += 1

        if self.elevationTester(elev):
            trueCounter += 1
        else:
            falseCounter += 1

        if self.slopeTester(slope):
            trueCounter += 1
        else:
            falseCounter += 1

        return trueCounter > falseCounter


# Example of a class that could contain your `tester` method
class PlantRecommender:
    def __init__(self, plants_data):
        self.plants_data = plants_data

    def tester(self, conditions):
        # Store scores in a dictionary
        scores = {}

        for plant_name, plant_data in self.plants_data.items():
            score = 0
            print(f"Testing for {plant_name}...")

            for param, value in conditions.items():
                param_range = plant_data.get(param)

                if param_range:
                    lower_threshold, upper_threshold = param_range
                    tester = ThresholdTester(lower_threshold, upper_threshold)

                    if param == "Soil water" and tester.soilTester(value):
                        score += 1
                    elif param == "Solar radiation" and tester.solarTester(value):
                        score += 1
                    elif param == "Precipitation" and tester.precipitationTester(value):
                        score += 1
                    elif param == "Temperature" and tester.temperatureTester(value):
                        score += 1
                    elif param == "Humidity" and tester.humidityTester(value):
                        score += 1
                    elif param == "NDVI" and tester.ndviTester(value):
                        score += 1
                    elif param == "EVI" and tester.eviTester(value):
                        score += 1
                    elif param == "Elevation" and tester.elevationTester(value):
                        score += 1
                    elif param == "Slope" and tester.slopeTester(value):
                        score += 1

            # Store the score for the plant
            scores[plant_name] = score

        # Sort plants by score in descending order and get the top 5
        top_plants = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]

        for plant, score in top_plants:
            print(f"{plant}: {score}")

        return top_plants



class EnvironmentalClassification:
    def classify_soil_water(self, soil_water_value):
        if soil_water_value > 12000:
            return "Hygroscopic"
        elif 5000 <= soil_water_value <= 12000:
            return "Capillary"
        else:
            return "Gravitational"

    def classify_solar_radiation(self, solar_radiation_value):
        if solar_radiation_value < 100:
            return "Low"
        elif 100 <= solar_radiation_value < 300:
            return "Low"
        elif 300 <= solar_radiation_value < 600:
            return "Moderate"
        elif 600 <= solar_radiation_value < 900:
            return "High"
        else:
            return "Extreme High"

    def classify_precipitation(self, precipitation_value):
        if precipitation_value < 250:
            return "Low"
        elif 250 <= precipitation_value < 500:
            return "Moderate"
        elif 500 <= precipitation_value < 1000:
            return "Sub-Humid"
        elif 1000 <= precipitation_value < 1500:
            return "Humid"
        elif 1500 <= precipitation_value < 2500:
            return "Very Humid"
        else:
            return "Excessive"

    def classify_temperature(self, temperature_value):
        if temperature_value < 5:
            return "Cold"
        elif 5 <= temperature_value < 15:
            return "Cool"
        elif 15 <= temperature_value < 25:
            return "Mild"
        elif 25 <= temperature_value < 35:
            return "Warm"
        else:
            return "Hot"

    def classify_ndvi(self, ndvi_value):
        if ndvi_value < 0:
            return "Non-Vegetated Areas"
        elif 0 <= ndvi_value < 0.2:
            return "Very Sparse Vegetation"
        elif 0.2 <= ndvi_value < 0.5:
            return "Moderate Vegetation"
        elif 0.5 <= ndvi_value < 0.7:
            return "Dense Vegetation/Healthy"
        else:
            return "Very Dense, Healthy Vegetation"



class PlantClassification:
    def __init__(self):
        self.plants_data = self.get_plant_data()

    def get_plant_data(self):
        return {
             "Mango": {
                "Soil water": (6000, 8000),
                "Solar radiation": (500, 700),
                "Precipitation": (750, 2500),
                "Temperature": (24, 30),
                "Humidity": (60, 70),
                "NDVI": (0.6, 0.8),
                "EVI": (0.5, 0.7),
                "Elevation": (0, 1200),
                "Slope": (0, 15)
            },
            "Banana": {
                "Soil water": (12000, 18000),
                "Solar radiation": (600, 800),
                "Precipitation": (1200, 2500),
                "Temperature": (26, 30),
                "Humidity": (70, 90),
                "NDVI": (0.7, 0.9),
                "EVI": (0.6, 0.9),
                "Elevation": (0, 1200),
                "Slope": (0, 10)
            },
            "Apple": {
                "Soil water": (7000, 10000),
                "Solar radiation": (500, 700),
                "Precipitation": (600, 1000),
                "Temperature": (18, 24),
                "Humidity": (50, 70),
                "NDVI": (0.6, 0.8),
                "EVI": (0.5, 0.7),
                "Elevation": (800, 2200),
                "Slope": (0, 10)
            },
            "Orange": {
                "Soil water": (6000, 8000),
                "Solar radiation": (500, 700),
                "Precipitation": (800, 1500),
                "Temperature": (22, 28),
                "Humidity": (50, 70),
                "NDVI": (0.6, 0.8),
                "EVI": (0.5, 0.7),
                "Elevation": (500, 1500),
                "Slope": (0, 12)
            },
            "Papaya": {
                "Soil water": (8000, 10000),
                "Solar radiation": (600, 800),
                "Precipitation": (1000, 2500),
                "Temperature": (25, 30),
                "Humidity": (70, 90),
                "NDVI": (0.7, 0.9),
                "EVI": (0.6, 0.8),
                "Elevation": (0, 1500),
                "Slope": (0, 10)
            },
            "Pineapple": {
                "Soil water": (7000, 10000),
                "Solar radiation": (500, 700),
                "Precipitation": (1000, 1500),
                "Temperature": (22, 28),
                "Humidity": (60, 80),
                "NDVI": (0.5, 0.7),
                "EVI": (0.4, 0.6),
                "Elevation": (0, 1000),
                "Slope": (0, 8)
            },
            "Grape": {
                "Soil water": (5000, 7000),
                "Solar radiation": (400, 600),
                "Precipitation": (500, 800),
                "Temperature": (18, 25),
                "Humidity": (50, 60),
                "NDVI": (0.5, 0.7),
                "EVI": (0.4, 0.6),
                "Elevation": (300, 1500),
                "Slope": (10, 30)
            },
            "Watermelon": {
                "Soil water": (4000, 6000),
                "Solar radiation": (600, 800),
                "Precipitation": (400, 600),
                "Temperature": (24, 30),
                "Humidity": (60, 70),
                "NDVI": (0.5, 0.7),
                "EVI": (0.5, 0.7),
                "Elevation": (0, 1000),
                "Slope": (0, 8)
            },
            "Avocado": {
                "Soil water": (6000, 8000),
                "Solar radiation": (500, 700),
                "Precipitation": (1200, 1600),
                "Temperature": (18, 25),
                "Humidity": (60, 80),
                "NDVI": (0.6, 0.8),
                "EVI": (0.5, 0.7),
                "Elevation": (300, 2100),
                "Slope": (0, 15)
            },
            "Chili": {
                "Soil water": (5000, 7000),
                "Solar radiation": (500, 700),
                "Precipitation": (600, 1200),
                "Temperature": (20, 30),
                "Humidity": (50, 70),
                "NDVI": (0.6, 0.8),
                "EVI": (0.5, 0.7),
                "Elevation": (0, 1500),
                "Slope": (0, 8)
            },
            "Onion": {
                "Soil water": (5000, 7000),
                "Solar radiation": (400, 600),
                "Precipitation": (500, 700),
                "Temperature": (13, 24),
                "Humidity": (50, 60),
                "NDVI": (0.4, 0.6),
                "EVI": (0.3, 0.5),
                "Elevation": (0, 1800),
                "Slope": (0, 5)
            },
            "Spinach": {
                "Soil water": (4000, 6000),
                "Solar radiation": (400, 600),
                "Precipitation": (600, 900),
                "Temperature": (15, 20),
                "Humidity": (60, 70),
                "NDVI": (0.5, 0.7),
                "EVI": (0.4, 0.6),
                "Elevation": (500, 2500),
                "Slope": (0, 5)
            },
            "Carrot": {
                "Soil water": (4000, 6000),
                "Solar radiation": (500, 700),
                "Precipitation": (700, 900),
                "Temperature": (16, 24),
                "Humidity": (50, 70),
                "NDVI": (0.4, 0.6),
                "EVI": (0.3, 0.5),
                "Elevation": (800, 2000),
                "Slope": (0, 5)
            },
            "Tomato": {
                "Soil water": (5000, 8000),
                "Solar radiation": (500, 700),
                "Precipitation": (600, 1200),
                "Temperature": (18, 27),
                "Humidity": (60, 80),
                "NDVI": (0.6, 0.8),
                "EVI": (0.5, 0.7),
                "Elevation": (0, 1500),
                "Slope": (0, 10)
            },
            "Cabbage": {
                "Soil water": (5000, 7000),
                "Solar radiation": (400, 600),
                "Precipitation": (700, 1000),
                "Temperature": (15, 20),
                "Humidity": (60, 70),
                "NDVI": (0.5, 0.7),
                "EVI": (0.4, 0.6),
                "Elevation": (800, 2400),
                "Slope": (0, 8)
            },
            "Potato": {
                "Soil water": (5000, 7000),
                "Solar radiation": (500, 700),
                "Precipitation": (500, 700),
                "Temperature": (15, 20),
                "Humidity": (60, 70),
                "NDVI": (0.5, 0.7),
                "EVI": (0.4, 0.6),
                "Elevation": (1000, 3000),
                "Slope": (0, 12)
            },
            "Garlic": {
                "Soil water": (5000, 7000),
                "Solar radiation": (400, 600),
                "Precipitation": (400, 600),
                "Temperature": (13, 24),
                "Humidity": (50, 60),
                "NDVI": (0.4, 0.6),
                "EVI": (0.3, 0.5),
                "Elevation": (800, 2400),
                "Slope": (0, 5)
            },
            "Eggplant": {
                "Soil water": (5000, 8000),
                "Solar radiation": (500, 700),
                "Precipitation": (600, 800),
                "Temperature": (22, 30),
                "Humidity": (60, 80),
                "NDVI": (0.6, 0.8),
                "EVI": (0.5, 0.7),
                "Elevation": (0, 1500),
                "Slope": (0, 8)
            },
            "Cucumber": {
                "Soil water": (4000, 6000),
                "Solar radiation": (500, 700),
                "Precipitation": (600, 1000),
                "Temperature": (18, 30),
                "Humidity": (70, 90),
                "NDVI": (0.6, 0.8),
                "EVI": (0.5, 0.7),
                "Elevation": (0, 1200),
                "Slope": (0, 8)
            }
        }

    def is_suitable(self, value, param_range, plant_name, param_name):
        # Print the current parameter range for debugging
        print(f"Checking suitability for {plant_name}, {param_name}: {param_range}")

        min_value, max_value = param_range

        # Adjust the returned values based on the input value
        if value < min_value:
            return max_value, max_value
        elif value > max_value:
            return min_value, min_value

        # Return the range itself if the value is within the bounds
        return min_value, max_value


    def tester(self, conditions):
        testing = ThresholdTester()
        best_plant = None
        highest_score = -1

        for plant_name, plant_data in self.plants_data.items():
          print(best_plant, highest_score)
          return best_plant, highest_score



def ranking_top5(classification_results):
    # Sort the results by score in descending order
    sorted_results = sorted(classification_results, key=lambda x: x[1], reverse=True)

    # Return the top 5 as a list of tuples (fruit_name, score, suitability)
    return sorted_results[:5]



def ranking_top5_price(price_predictions):
    sorted_results = sorted(price_predictions.items(), key=lambda x: x[1], reverse=True)
    print(sorted_results)
    return sorted_results[:5]





# def get_predictions_json(test_cases):



    
#     predictions_rf = models['rf'].astype(float).predict(test_cases)
#     predictions_sw = models['sw'].astype(float).predict(test_cases)
#     predictions_sr = models['sr'].astype(float).predict(test_cases)
#     predictions_temp = models['temp'].astype(float).predict(test_cases)



#     # Ensure all inputs are numpy arrays and get the first element if they're 2D
#     predictions_rf = np.array(predictions_rf).flatten()[0]
#     predictions_sw = np.array(predictions_sw).flatten()[0]
#     predictions_sr = np.array(predictions_sr).flatten()[0]
#     predictions_temp = np.array(predictions_temp).flatten()[0]

#     conditions = {
#     "Soil water": predictions_sw*1000000,
#     "Solar radiation": predictions_sr.flatten()[0]/100,
#     "Precipitation": predictions_rf.flatten()[0]*100,
#     "Temperature": predictions_temp.flatten()[0]-273.15,
#     }

#     classifier = EnvironmentalClassification()
#     # plant_classifier = PlantClassification()

#     soil_water = classifier.classify_soil_water(predictions_sw)
#     solar_radiation = classifier.classify_solar_radiation(predictions_sr)
#     precipitation = classifier.classify_precipitation(predictions_rf)
#     temperature = classifier.classify_temperature(predictions_temp)


#     top_plants = PlantRecommender(PlantClassification().plants_data).tester(conditions)

#     for fruit in top_plants:
#         conditions = {
#             "Soil water": predictions_sw,
#             "Solar radiation": predictions_sr,
#             "Precipitation": predictions_rf,
#             "Temperature": predictions_temp
#         }


#         predictions_df = pd.DataFrame({
#             'Longitude': test_cases[:, 0],
#             'Latitude': test_cases[:, 1],
#             # 'TOP_5H': [top_5_names] * len(test_cases),
#             # 'TOP_5H_PRICE': [top_5_price] * len(test_cases),
#             # 'TOP_5H_SCORES': price,
#             'Prediction_RF': predictions_rf,
#             'Prediction_SW': predictions_sw,
#             'Prediction_SR': predictions_sr,
#             'Prediction_Temp': predictions_temp - 273.15,  # Convert temperature to Celsius
#             'Soil_Water': [soil_water] * len(test_cases),
#             'Solar_Radiation': [solar_radiation] * len(test_cases),
#             'Precipitation': [precipitation] * len(test_cases),
#             'Temperature': [temperature] * len(test_cases),
#             # 'Top_5_Classif': [top_5_classif] * len(test_cases),
#             # 'All_Price' : [fruitAndVegetablePredictions] * len(test_cases),
#             # 'threshold' : [threshold] * len(test_cases)
#             'threshold_points': [top_plants] * len(test_cases)
#         })

#         # Convert DataFrame to JSON
#         return predictions_df


#         # return json_output


import numpy as np
import pandas as pd

def get_predictions_json(test_cases):
    # Model predictions
    predictions_rf = models['rf'].predict(test_cases).flatten()[0]
    predictions_sw = models['sw'].predict(test_cases).flatten()[0]
    predictions_sr = models['sr'].predict(test_cases).flatten()[0]
    predictions_temp = models['temp'].predict(test_cases).flatten()[0]


    # print(predictions_rf)
    # Conditions for plant recommendation
    conditions = {
        "Soil water": predictions_sw * 1000000,
        "Solar radiation": predictions_sr / 100,
        "Precipitation": predictions_rf * 100,
        "Temperature": predictions_temp - 273.15,
    }


    # print(conditions)
    

    # classifier = EnvironmentalClassification()
    # soil_water = classifier.classify_soil_water(predictions_sw)
    # solar_radiation = classifier.classify_solar_radiation(predictions_sr)
    # precipitation = classifier.classify_precipitation(predictions_rf)
    # temperature = classifier.classify_temperature(predictions_temp)

    print("cp")
    # top_plants = PlantRecommender(PlantClassification().plants_data).tester(conditions)

    # Create DataFrame for predictions and classification results
    predictions_df = pd.DataFrame({
        'Longitude': test_cases[:, 0],  # Ensure native float
        'Latitude': test_cases[:, 1],   # Ensure native float
        'Prediction_RF': [predictions_rf] * len(test_cases),
        'Prediction_SW': [predictions_sw] * len(test_cases),
        'Prediction_SR': [predictions_sr] * len(test_cases),
        'Prediction_Temp': [predictions_temp - 273.15] * len(test_cases),
        # 'Soil_Water': [soil_water] * len(test_cases),
        # 'Solar_Radiation': [solar_radiation] * len(test_cases),
        # 'Precipitation': [precipitation] * len(test_cases),
        # 'Temperature': [temperature] * len(test_cases),
        # 'threshold_points': [top_plants] * len(test_cases)
    })

    # Convert the DataFrame to JSON and ensure proper float conversion
    predictions_json = predictions_df.to_json(orient="records")
    
    return predictions_json


def get_price_json(year_cases, fruitAndVegetables):
    price_df = pd.DataFrame({
        'Year_Month': year_cases,
        'FruitAndVegetables': [fruitAndVegetables] * len(year_cases)
    })
    return price_df


import numpy as np
from flask import request, jsonify

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     if 'test_cases' not in data:
#         return jsonify({"error": "Missing 'test_cases' in request body"}), 400

#     test_cases = data['test_cases']
#     predictions_rf = models['rf'].predict(test_cases).flatten()[0]
#     predictions_sw = models['sw'].predict(test_cases).flatten()[0]
#     predictions_sr = models['sr'].predict(test_cases).flatten()[0]
#     predictions_temp = models['temp'].predict(test_cases).flatten()[0]


#     print(predictions_rf)
#     # Conditions for plant recommendation
#     conditions = {
#         "Soil water": predictions_sw * 1000000,
#         "Solar radiation": predictions_sr / 100,
#         "Precipitation": predictions_rf * 100,
#         "Temperature": predictions_temp - 273.15,
#     }

#     classifier = EnvironmentalClassification()
#     soil_water = classifier.classify_soil_water(predictions_sw)
#     solar_radiation = classifier.classify_solar_radiation(predictions_sr)
#     precipitation = classifier.classify_precipitation(predictions_rf)
#     temperature = classifier.classify_temperature(predictions_temp)

#     print(test_cases)
#     top_plants = PlantRecommender(PlantClassification().plants_data).tester(conditions)

#     # Create DataFrame for predictions and classification results
#     predictions_df = pd.DataFrame({
#         'Longitude': test_cases[:, 0],  # Ensure native float
#         'Latitude': test_cases[:, 1],   # Ensure native float
#         'Prediction_RF': [predictions_rf] * len(test_cases),
#         'Prediction_SW': [predictions_sw] * len(test_cases),
#         'Prediction_SR': [predictions_sr] * len(test_cases),
#         'Prediction_Temp': [predictions_temp - 273.15] * len(test_cases),
#         'Soil_Water': [soil_water] * len(test_cases),
#         'Solar_Radiation': [solar_radiation] * len(test_cases),
#         'Precipitation': [precipitation] * len(test_cases),
#         'Temperature': [temperature] * len(test_cases),
#         'threshold_points': [top_plants] * len(test_cases)
#     })
    
#     return jsonify(predictions_df)
    
######################################################################

# import numpy as np
# import pandas as pd
# from flask import jsonify, request

# # @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json

#     if 'test_cases' not in data:
#         return jsonify({"error": "Missing 'test_cases' in request body"}), 400

#     # Convert the test cases to a NumPy array
#     test_cases = np.array(data['test_cases'])

#     # Make predictions using your models
#     predictions_rf = models['rf'].predict(test_cases).flatten()[0]
#     predictions_sw = models['sw'].predict(test_cases).flatten()[0]
#     predictions_sr = models['sr'].predict(test_cases).flatten()[0]
#     predictions_temp = models['temp'].predict(test_cases).flatten()[0]

#     # Conditions for plant recommendation
#     conditions = {
#         "Soil water": predictions_sw * 1000000,
#         "Solar radiation": predictions_sr / 100,
#         "Precipitation": predictions_rf * 100,
#         "Temperature": predictions_temp - 273.15,
#     }

#     classifier = EnvironmentalClassification()
#     soil_water = classifier.classify_soil_water(predictions_sw)
#     solar_radiation = classifier.classify_solar_radiation(predictions_sr)
#     precipitation = classifier.classify_precipitation(predictions_rf)
#     temperature = classifier.classify_temperature(predictions_temp)

#     top_plants = PlantRecommender(PlantClassification().plants_data).tester(conditions)

#     # Create DataFrame for predictions and classification results
#     predictions_df = pd.DataFrame({
#         'Longitude': test_cases[:, 0],  # Ensure it's now a NumPy array
#         'Latitude': test_cases[:, 1],
#         'Prediction_RF': [predictions_rf] * len(test_cases),
#         'Prediction_SW': [predictions_sw] * len(test_cases),
#         'Prediction_SR': [predictions_sr] * len(test_cases),
#         'Prediction_Temp': [predictions_temp - 273.15] * len(test_cases),
#         'Soil_Water': [soil_water] * len(test_cases),
#         'Solar_Radiation': [solar_radiation] * len(test_cases),
#         'Precipitation': [precipitation] * len(test_cases),
#         'Temperature': [temperature] * len(test_cases),
#         'threshold_points': [top_plants] * len(test_cases)
#     })
    
#     # Convert DataFrame to JSON and return the response
#     return predictions_df.to_json(orient='records')



# # @app.route('/recommend', methods=['POST'])
# def recommend_plant():
#     year_cases = request.json['year_cases']
    
    
#     fruit_and_vegetable_predictions = {
#         fruit: model.predict(year_cases).tolist()
#         for fruit, model in fruit_models.items()
#     }

#     output = {
#         'Year_Month': year_cases,
#         'FruitAndVegetables': [fruit_and_vegetable_predictions]*len(year_cases)
#     }

#     return jsonify(output)




# import numpy as np
# import pandas as pd
# import json

# # Function to iterate over latitude and longitude and save to a single JSON file
# def generate_lat_long_predictions():
#     lat_range = np.arange(180, -180, -1)  # Latitude from 180 to -180
#     long_range = np.arange(180, -180, -1)  # Longitude from 180 to -180
#     all_predictions = []

#     for lat in lat_range:
#         for lon in long_range:
#             # Prepare test case with the current latitude and longitude
#             test_case = [[float(lon), float(lat)]]  # Ensure values are float, which is JSON serializable
#             data = {'test_cases': test_case}

#             # Simulate the predict request
#             with app.test_client() as client:
#                 response = 
#                     try:
#                         predictions = response.get_json()
#                         if predictions:
#                             all_predictions.extend(predictions)  # Append predictions if valid
#                         else:
#                             print(f"No predictions returned for lat: {lat}, lon: {lon}")
#                     except Exception as e:
#                         print(f"Error parsing JSON for lat: {lat}, lon: {lon} - {e}")
#                 else:
#                     print(f"Request failed with status code {response.status_code} for lat: {lat}, lon: {lon}")

#     # Save all predictions to a single JSON file
#     with open('predictions.json', 'w') as f:
#         json.dump(all_predictions, f, indent=4)



import numpy as np
import pandas as pd
import json

# Assuming `predict` is the function to call directly without Flask
def predict(test_cases):
    # Here is the logic from your Flask route, adapted to work as a regular function
    if not test_cases:
        return {"error": "Missing 'test_cases' in request body"}

    # Convert the test cases to a NumPy array
    test_cases = np.array(test_cases)

    # Make predictions using your models
    predictions_rf = models['rf'].predict(test_cases).flatten()[0]
    predictions_sw = models['sw'].predict(test_cases).flatten()[0]
    predictions_sr = models['sr'].predict(test_cases).flatten()[0]
    predictions_temp = models['temp'].predict(test_cases).flatten()[0]

    # Conditions for plant recommendation
    conditions = {
        "Soil water": predictions_sw * 1000000,
        "Solar radiation": predictions_sr / 100,
        "Precipitation": predictions_rf * 100,
        "Temperature": predictions_temp - 273.15,
    }

    classifier = EnvironmentalClassification()
    soil_water = classifier.classify_soil_water(predictions_sw)
    solar_radiation = classifier.classify_solar_radiation(predictions_sr)
    precipitation = classifier.classify_precipitation(predictions_rf)
    temperature = classifier.classify_temperature(predictions_temp)
    top_plants = PlantRecommender(PlantClassification().plants_data).tester(conditions)

    # Create DataFrame for predictions and classification results
    predictions_df = pd.DataFrame({
        'Longitude': test_cases[:, 0],  # Ensure it's a NumPy array
        'Latitude': test_cases[:, 1],
        'Prediction_RF': [predictions_rf] * len(test_cases),
        'Prediction_SW': [predictions_sw] * len(test_cases),
        'Prediction_SR': [predictions_sr] * len(test_cases),
        'Prediction_Temp': [predictions_temp - 273.15] * len(test_cases),
        'Soil_Water': [soil_water] * len(test_cases),
        'Solar_Radiation': [solar_radiation] * len(test_cases),
        'Precipitation': [precipitation] * len(test_cases),
        'Temperature': [temperature] * len(test_cases),
        'threshold_points': [top_plants] * len(test_cases)
    })

    # Return the result as a dictionary
    return predictions_df.to_dict(orient='records')


# Function to iterate over latitude and longitude and save to a single JSON file
def generate_lat_long_predictions():
    lat_range = np.arange(10, -10, -1)  # Latitude from 180 to -180
    long_range = np.arange(200, 210, -1)  # Longitude from 180 to -180
    all_predictions = []

    for lat in lat_range:
        for lon in long_range:
            # Prepare test case with the current latitude and longitude
            test_case = [[float(lon), float(lat)]]  # Ensure values are float
            predictions = predict(test_case)
            
            if isinstance(predictions, list):
                all_predictions.extend(predictions)  # Append predictions if valid
            else:
                print(f"Error in predictions for lat: {lat}, lon: {lon} - {predictions}")

    # Save all predictions to a single JSON file
    with open('predictions.json', 'w') as f:
        json.dump(all_predictions, f, indent=4)

# Run the lat-long prediction generation
generate_lat_long_predictions()


