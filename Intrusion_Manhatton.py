import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import socket
import subprocess
import openai
import os

# Load train and test data
# this data is from https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection
train_data = pd.read_csv("Train_data.csv")
test_data = pd.read_csv("Test_data.csv")

# Identify categorical and numeric columns
categorical_cols = train_data.select_dtypes(include=['object']).columns
numeric_cols = train_data.select_dtypes(include=['int64', 'float64']).columns

# One Hot Encode categorical data
train_data_encoded = pd.get_dummies(train_data, columns=categorical_cols)

# Scale numeric columns
scaler = StandardScaler()
train_data_encoded[numeric_cols] = scaler.fit_transform(train_data_encoded[numeric_cols])

# Split data
X = train_data_encoded.drop(columns=['class'], errors='ignore')
y = train_data['class'] if 'class' in train_data.columns and train_data['class'].dtype == 'object' else train_data_encoded['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Define preprocess function
def preprocess(data):
    # Debug: Print incoming data format
    print("Raw data received:", data)
    
    # Attempt to split the incoming data into a list
    data_list = data.split(',')
    print("Split data:", data_list)

    # Check if data_list has the correct number of elements
    if len(data_list) != len(X.columns):
        raise ValueError(f"Expected {len(X.columns)} features, but received {len(data_list)}.")

    # Create DataFrame
    data_df = pd.DataFrame([data_list], columns=X.columns)

    # Handle categorical data: One-hot encode
    data_encoded = pd.get_dummies(data_df)
    
    # Ensure all expected columns are present
    missing_cols = set(X.columns) - set(data_encoded.columns)
    for c in missing_cols:
        data_encoded[c] = 0
    data_encoded = data_encoded[X.columns]

    # Scale numeric features
    data_encoded[numeric_cols] = scaler.transform(data_encoded[numeric_cols])
    
    return data_encoded

# Define action mapping based on predictions
action_mapping = {
    'safe': "echo 'No action required'"  # Action for safe prediction
}

# Define OpenAI action generation function
def generate_action(prediction):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Create a prompt for the OpenAI model
    prompt = f"The AI model predicted '{prediction}'. What action should be taken in response to this prediction?"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )

    action = response.choices[0].text.strip()
    return action

# Define action execution function
def execute_action(action):
    # Execute the command
    try:
        result = subprocess.run(action, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Action executed: {action}")
        print(f"Output: {result.stdout.decode('utf-8')}")
        print(f"Error: {result.stderr.decode('utf-8')}")
        
        # Handling the results
        if result.returncode == 0:
            print("Command executed successfully")
        else:
            print("Command failed")
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute action: {action}")
        print(f"Error: {e}")

# Define prediction function
def predict(data):
    model = joblib.load('knn_model.pkl')
    processed_data = preprocess(data)
    prediction = model.predict(processed_data)[0]  # Get the prediction as a string
    
    # Determine the action based on the prediction
    action = action_mapping.get(prediction)
    if not action:
        action = generate_action(prediction)  # Generate action if not predefined
    execute_action(action)  # Execute the determined action
    
    return prediction

# Server code to handle incoming connections and make predictions
def start_server(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", port))
    s.listen(5)
    print(f"Listening on port {port}...")

    while True:
        clientsocket, address = s.accept()
        print(f"Connection from {address} has been established!")
        data = clientsocket.recv(1024).decode('utf-8').strip()
        prediction = predict(data)
        print(f"Received data: {data}")
        print(f"Prediction: {prediction}")
        clientsocket.send(bytes(f"Prediction: {prediction}", "utf-8"))
        clientsocket.close()

if __name__ == "__main__":
    start_server(9999)
