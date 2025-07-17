import numpy as np
from PIL import Image
import io, base64, os, sys
import google.generativeai as genai

# ===== CONFIG =====
GEMINI_API_KEY = "AIzaSyAp-xGpVAdK6eqbdtGz2V5TriXEE8Z3VOQ"  # 👉 Replace with your working key from https://aistudio.google.com/app/apikey
genai.configure(api_key=GEMINI_API_KEY)

IMAGE_PATH = r"C:\Users\hp\Desktop\research paper\singleplant.jpg"

# ===== HELPERS =====
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256))
    return img

def is_blank_image(img):
    arr = np.array(img)
    return np.std(arr) < 5

def enhance_vegetation(img):
    arr = np.array(img).astype(float)
    g = np.clip(arr[:, :, 1] * 1.5, 0, 255)
    r = np.clip(arr[:, :, 0] * 0.5, 0, 255)
    b = np.clip(arr[:, :, 2] * 0.5, 0, 255)
    return Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8))

def compute_rgvi(img):
    arr = np.array(img).astype(float)
    red, green = arr[:, :, 0], arr[:, :, 1]
    rgvi = (green - red) / (green + red + 1e-5)
    return np.mean(np.clip(rgvi, -1, 1))

def compute_weather_stress(temp_c, humidity):
    temp_stress = max(0, abs(temp_c - 25) / 20)   # ideal ~25°C
    hum_stress  = max(0, abs(humidity - 65) / 50) # ideal ~65%
    return min(1, (temp_stress + hum_stress) / 2)

def soil_modifier(soil_type):
    s = soil_type.lower()
    if "sandy" in s: return 1.2
    if "clay"  in s: return 1.1
    if "loam"  in s: return 0.9
    return 1.0

def irrigation_recommendation(stress_score, plant_size="medium"):
    base_water = {"small": 0.5, "medium": 1.5, "large": 3.0}
    return round(base_water.get(plant_size, 1.5) * stress_score * 2, 2)

def confidence_from_rgvi(img):
    arr = np.array(img)
    std_val = np.std(arr[:, :, 1])  # green channel stddev
    return min(1.0, std_val / 50.0)

def water_deficit_index(stress_score):
    return round(stress_score, 2)

# ===== GEMINI VISION =====
def validate_with_gemini(image_path):
    model = genai.GenerativeModel("gemini-1.5-flash")
    img = Image.open(image_path)
    prompt = "Does this image show a single plant? Reply Yes or No. If yes, does it look wilted or drought-stressed?"
    response = model.generate_content([prompt, img])
    return response.text.strip()

def expert_summary_with_gemini(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def get_accuracy_from_gemini(summary_text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are an expert agronomist.
You just gave this plant water stress analysis:

\"\"\"{summary_text}\"\"\"

How confident are you that this analysis and irrigation recommendation is correct? 
Reply ONLY with a number between 0 and 100 (no % sign, no explanation).
"""
    response = model.generate_content(prompt)
    text = response.text.strip()
    try:
        # Extract only digits and parse
        number = ''.join(ch for ch in text if ch.isdigit() or ch == '.')
        confidence_percent = float(number)
        confidence_percent = max(0, min(confidence_percent, 100))  # clamp between 0-100
        return confidence_percent
    except:
        return None

# ===== MAIN =====
if not os.path.isfile(IMAGE_PATH):
    print(f"❌ Image not found: {IMAGE_PATH}")
    sys.exit()

image = load_image(IMAGE_PATH)
if is_blank_image(image):
    print("❌ Blank image detected!")
    sys.exit()

# ✅ Validate with Gemini Vision
print("🔍 Validating image with Gemini 1.5 Vision...")
validation = validate_with_gemini(IMAGE_PATH)
print(f"🌿 Gemini says: {validation}")
if "no" in validation.lower():
    print("❌ Not a valid single plant image!")
    sys.exit()

# ✅ Get weather + soil inputs
print("\n🌦 Enter weather conditions:")
weather_desc = input("Weather (clear, cloudy...): ")
temp_c = float(input("Temperature °C: "))
humidity = float(input("Humidity %: "))
soil_type = input("Soil type (sandy, clay, loamy...): ")

# ✅ Compute vegetation stress
enhanced_img = enhance_vegetation(image)
rgvi_score = compute_rgvi(enhanced_img)
ndvi_stress = 1 - max(0, min(1, (rgvi_score + 1) / 2))
weather_stress = compute_weather_stress(temp_c, humidity)
combined_stress = (0.5 * ndvi_stress + 0.5 * weather_stress) * soil_modifier(soil_type)
combined_stress = min(1, combined_stress)

if combined_stress < 0.3:
    stress_status = "Low water stress – plant looks healthy."
elif combined_stress < 0.6:
    stress_status = "Moderate water stress – some irrigation may help."
else:
    stress_status = "High water stress – urgent watering required."

# ✅ Extra metrics
irrigation_need = irrigation_recommendation(combined_stress)
confidence_score = confidence_from_rgvi(enhanced_img)
wdi = water_deficit_index(combined_stress)

# ✅ Ask Gemini for expert reasoning
prompt = f"""
You are an agricultural expert analyzing a SINGLE PLANT.

- Gemini Vision says: {validation}
- RGVI Score: {rgvi_score:.2f} (0 unhealthy, 1 healthy)
- Weather: {weather_desc}, {temp_c}°C, {humidity}% humidity
- Soil type: {soil_type}
- Combined water stress score: {combined_stress:.2f}
- Water Deficit Index (0 none → 1 severe): {wdi}
- Confidence in vegetation reading: {confidence_score:.2f}
- Estimated irrigation need: {irrigation_need} liters
- Final interpretation: {stress_status}

Explain in 3-5 sentences WHY this plant has this water stress level,
mention how weather, soil, and vegetation index contribute,
and give a short irrigation recommendation.
"""
expert_summary = expert_summary_with_gemini(prompt)

# ✅ Ask Gemini for its self-rated accuracy
accuracy_percent = get_accuracy_from_gemini(expert_summary)

# ✅ Final report
print("\n=== 🌱 Plant Water Stress Report ===")
print(f"RGVI Score           : {rgvi_score:.2f}")
print(f"Weather Stress       : {weather_stress:.2f}")
print(f"Combined Stress      : {combined_stress:.2f}")
print(f"Water Deficit Index  : {wdi}")
print(f"Confidence Score     : {confidence_score:.2f}")
print(f"Irrigation Need      : {irrigation_need} liters")
print(f"Soil Type            : {soil_type}")
print(f"Status               : {stress_status}")

print("\nExpert Summary:")
print(expert_summary)

if accuracy_percent is not None:
    print(f"\n🔎 Gemini Self-rated Accuracy: {accuracy_percent:.1f}%")
else:
    print("\n🔎 Gemini could not determine confidence score.")
