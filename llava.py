import numpy as np
from PIL import Image
import io
import requests
import base64
import sys
import os

# ======== CONFIG ========
LLAVA_URL = "http://localhost:11434/api/generate"   # LLaVA API endpoint
LLM_URL   = "http://localhost:11434/api/generate"   # LLaMA3 endpoint
IMAGE_PATH = r"C:\Users\hp\Desktop\research paper\singleplant.jpg"  # <-- Your plant image

# ======== HELPERS ========

def load_image(path):
    """Load and resize plant image"""
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256))
    return img

def is_blank_image(img):
    """Check if uploaded image is blank (no variance)"""
    arr = np.array(img)
    return np.std(arr) < 5

def image_to_base64(img):
    """Convert image to base64 for LLaVA"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def validate_image(img_base64):
    """Ask LLaVA if image is a plant & if it looks stressed"""
    validation_prompt = (
        "Does this image show a single plant? Reply only 'Yes' or 'No'. "
        "If yes, does the plant look wilted or drought-stressed?"
    )
    payload = {
        "model": "llava",
        "prompt": validation_prompt,
        "images": [img_base64],
        "stream": False
    }
    try:
        r = requests.post(LLAVA_URL, json=payload)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        else:
            return "Error validating image"
    except Exception as e:
        return f"Error: {e}"

def enhance_vegetation(img):
    """Emphasize green channel for RGVI analysis"""
    arr = np.array(img).astype(float)
    g = np.clip(arr[:, :, 1] * 1.5, 0, 255)
    r = np.clip(arr[:, :, 0] * 0.5, 0, 255)
    b = np.clip(arr[:, :, 2] * 0.5, 0, 255)
    return Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8))

def compute_rgvi(img):
    """Compute Red-Green Vegetation Index"""
    arr = np.array(img).astype(float)
    red, green = arr[:, :, 0], arr[:, :, 1]
    rgvi = (green - red) / (green + red + 1e-5)
    return np.mean(np.clip(rgvi, -1, 1))

def compute_weather_stress(temp_c, humidity):
    """Compute stress from temperature & humidity (0 healthy -> 1 high stress)"""
    temp_stress = max(0, abs(temp_c - 25) / 20)   # ideal ~25°C
    hum_stress  = max(0, abs(humidity - 65) / 50) # ideal ~65%
    return min(1, (temp_stress + hum_stress) / 2)

def soil_modifier(soil_type):
    """Soil type affects final stress"""
    s = soil_type.lower()
    if "sandy" in s: return 1.2  # drains fast → more stress
    if "clay"  in s: return 1.1  # retains → slightly less stress
    if "loam"  in s: return 0.9  # optimal soil → less stress
    return 1.0

def ask_llm(summary_prompt):
    """Ask LLaMA3 for an expert explanation"""
    payload = {
        "model": "llama3",
        "prompt": summary_prompt,
        "stream": False
    }
    try:
        r = requests.post(LLM_URL, json=payload)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        else:
            return "Error: LLM failed"
    except Exception as e:
        return f"Error: {e}"

# ======== EXTRA METRICS ========

def water_deficit_index(stress_score):
    """Estimate water deficit (0=none, 1=severe) based on stress"""
    return round(stress_score, 2)

def irrigation_recommendation(stress_score, plant_size="medium"):
    """
    Estimate irrigation need in liters based on stress and plant size.
    """
    base_water = {"small": 0.5, "medium": 1.5, "large": 3.0}
    return round(base_water.get(plant_size, 1.5) * stress_score * 2, 2)

def confidence_from_rgvi(img):
    """
    Confidence in vegetation reading: higher std → better vegetation contrast → higher confidence
    """
    arr = np.array(img)
    std_val = np.std(arr[:, :, 1])  # green channel variation
    return min(1.0, std_val / 50.0)  # normalize to 0-1

def get_accuracy_from_llm(summary_text):
    """Ask LLaMA3 to self-assess accuracy in %"""
    accuracy_prompt = f"""
You are an expert agronomist.
You just gave this plant water stress analysis:

\"\"\"{summary_text}\"\"\"

On a scale of 0 to 100, how confident are you that this diagnosis and irrigation recommendation is correct? 
Reply with ONLY the number (0 to 100) without any explanation.
"""
    result = ask_llm(accuracy_prompt)
    # Extract numeric confidence
    try:
        confidence_percent = float(''.join(ch for ch in result if (ch.isdigit() or ch == '.')))
        confidence_percent = min(100.0, max(0.0, confidence_percent))  # clamp 0-100
        return confidence_percent
    except:
        return None

# ======== MAIN ========

# 1️⃣ Load image
if not os.path.isfile(IMAGE_PATH):
    print(f"❌ Image not found: {IMAGE_PATH}")
    sys.exit()

image = load_image(IMAGE_PATH)
if is_blank_image(image):
    print("❌ The uploaded image appears blank!")
    sys.exit()

# 2️⃣ Validate with LLaVA
img_base64 = image_to_base64(image)
llava_result = validate_image(img_base64)
print(f"🔍 LLaVA says: {llava_result}")
if "no" in llava_result.lower():
    print("❌ Not a valid plant image. Exiting.")
    sys.exit()

# 3️⃣ Get WEATHER + SOIL details
print("\n🌦 Please enter the current weather for this plant:")
weather_desc = input("Weather description (e.g., clear sky): ").strip()
temp_c = float(input("Temperature in °C: ").strip())
humidity = float(input("Humidity %: ").strip())
soil_type = input("Soil type (sandy, clay, loamy, etc.): ").strip()

# 4️⃣ Compute vegetation stress
enhanced_img = enhance_vegetation(image)
rgvi_score = compute_rgvi(enhanced_img)
ndvi_stress = 1 - max(0, min(1, (rgvi_score + 1) / 2)) # invert: low RGVI → high stress

# 5️⃣ Weather stress
weather_stress = compute_weather_stress(temp_c, humidity)

# 6️⃣ Combine RGVI & Weather (50%-50%) + Soil modifier
combined_stress = (0.5 * ndvi_stress + 0.5 * weather_stress) * soil_modifier(soil_type)
combined_stress = min(1, combined_stress)

if combined_stress < 0.3:
    stress_status = "Low water stress – plant looks healthy."
elif combined_stress < 0.6:
    stress_status = "Moderate water stress – some irrigation may help."
else:
    stress_status = "High water stress – urgent watering required."

# ✅ New metrics
wdi = water_deficit_index(combined_stress)
confidence_score = confidence_from_rgvi(enhanced_img)
irrigation_need = irrigation_recommendation(combined_stress, plant_size="medium")

# 7️⃣ Ask LLaMA3 for expert summary
summary_prompt = f"""
You are an agricultural expert analyzing a SINGLE PLANT.

- LLaVA says: {llava_result}
- RGVI Score: {rgvi_score:.2f} (0 unhealthy, 1 healthy)
- Weather: {weather_desc}, {temp_c}°C, {humidity}% humidity
- Soil type: {soil_type}
- Combined water stress score (0 healthy → 1 very stressed): {combined_stress:.2f}
- Water Deficit Index (0 none → 1 severe): {wdi}
- Confidence in vegetation reading: {confidence_score:.2f}
- Estimated irrigation need: {irrigation_need} liters
- Final interpretation: {stress_status}

Explain in 3-5 sentences WHY this plant has this water stress level,
mention how weather, soil, and vegetation index contribute,
and give a short irrigation recommendation.
"""

expert_summary = ask_llm(summary_prompt)

# 8️⃣ Ask LLaMA3 for self-rated accuracy
accuracy_percent = get_accuracy_from_llm(expert_summary)

# 9️⃣ Final report
print("Single Plant Water Stress Report")
print(f"RGVI Score             : {rgvi_score:.2f}")
print(f"Weather Stress         : {weather_stress:.2f}")
print(f"Combined Stress        : {combined_stress:.2f}")
print(f"Water Deficit Index    : {wdi}")
print(f"Confidence Score       : {confidence_score:.2f}")
print(f"Estimated Irrigation   : {irrigation_need} liters")
if accuracy_percent is not None:
    print(f"metric score   : {accuracy_percent:.1f}%")
else:
    print("metric score    : Could not determine")
print(f"Soil Type              : {soil_type}")
print(f"Status                 : {stress_status}")
print("\nExpert Summary:")
print(expert_summary)

