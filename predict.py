import io
import joblib
import requests

# 1. Define the Raw GitHub URL of your saved model
MODEL_URL = "https://raw.githubusercontent.com/Gayathri08087/MLCS_Deploying_ML_Model/main/20newsgroups_model.joblib"

print("Fetching the trained model directly from GitHub...")

try:
    # 2. Download the model file content into memory
    response = requests.get(MODEL_URL)
    response.raise_for_status() # Check for HTTP errors

    # 3. Load the model from the raw bytes buffer
    model_bytes = io.BytesIO(response.content)
    model = joblib.load(model_bytes)
    print("Model loaded successfully into memory!\n")

    # 4. Define new, unseen text samples to classify
    new_samples = [
        "The local team won the championship game in overtime last night.",
        "Scientists have discovered a new planet orbiting a distant star using the Hubble telescope.",
        "My computer's graphics card is overheating when I try to boot up the operating system."
    ]

    # 5. Use the model to predict categories
    predictions = model.predict(new_samples)

    # Map the predicted numeric IDs back to target text names if known,
    # or print out the predictions directly.
    print("--- Prediction Results ---")
    for text, pred in zip(new_samples, predictions):
        print(f"Text: '{text}'")
        print(f"Predicted Category ID: {pred}\n")

except requests.exceptions.RequestException as e:
    print(f"Error fetching model from GitHub: {e}")
except Exception as e:
    print(f"An error occurred while loading or running the model: {e}")
