import requests
from PIL import Image
import base64
import io

# === Step 1: Load image ===
image_path = r"C:\Users\hp\Desktop\research paper\satellite_image.jpg"
image = Image.open(image_path).convert("RGB")

# === Step 2: Encode image to base64 ===
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

# === Step 3: Send request to Ollama ===
url = "http://localhost:11434/api/generate"
payload = {
    "model": "llava",  # or "llava:13b"
    "prompt": "You are an expert in satellite image analysis. Describe in detail what you see in this image: terrain type, human structures, land use, and any other features.",
    "images": [image_base64],  # Must be inside a list!
    "stream": False
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

# === Step 4: Show result ===
if response.status_code == 200:
    result = response.json().get("response", "No result in response.")
    print("🔍 Result:", result)
else:
    print(f"❌ Failed with status code {response.status_code}")
