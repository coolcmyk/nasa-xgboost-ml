
import json 


import numpy as np
import pandas as pd
import json

# Function to iterate over latitude and longitude and save to a single JSON file
def generate_lat_long_predictions():
    lat_range = np.arange(180, -180, -1)  # Latitude from 180 to -180
    long_range = np.arange(180, -180, -1)  # Longitude from 180 to -180
    all_predictions = []

    for lat in lat_range:
        for lon in long_range:
            # Prepare test case with the current latitude and longitude
            test_case = [[float(lon), float(lat)]]  # Ensure values are float, which is JSON serializable
            data = {'test_cases': test_case}

            # Simulate the predict request
            with app.test_client() as client:
                response = client.post('/predict', json=data)
                if response.status_code == 200:
                    try:
                        predictions = response.get_json()
                        if predictions:
                            all_predictions.extend(predictions)  # Append predictions if valid
                        else:
                            print(f"No predictions returned for lat: {lat}, lon: {lon}")
                    except Exception as e:
                        print(f"Error parsing JSON for lat: {lat}, lon: {lon} - {e}")
                else:
                    print(f"Request failed with status code {response.status_code} for lat: {lat}, lon: {lon}")

    # Save all predictions to a single JSON file
    with open('predictions.json', 'w') as f:
        json.dump(all_predictions, f, indent=4)

# Run the lat-long prediction generation
generate_lat_long_predictions()