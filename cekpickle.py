import pickle

# Open the pickle file in read-binary mode
with open('face_mask_detection.pkl', 'rb') as file:
    data = pickle.load(file)

# Inspect the loaded data
if isinstance(data, dict):
    # If the data is a dictionary
    for key, value in data.items():
        print(key, ":", value)
elif isinstance(data, list):
    # If the data is a list
    for item in data:
        print(item)
else:
    # Handle other data types or structures
    print(data)
