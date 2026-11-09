# %%
from joblib import load
from pathlib import Path
import cv2
from homeassistant_api import Client

model = load('svm_model.joblib')
scaler = load('scaled.joblib')

print("running model prediction")
# %%
label_coding = {0: "other", 1: "ellipse", 2: "triangle", 3: "rectangle"}

# %%
def preprocess_image(img_path):
    source_folder = Path('live_images')
    img_path = source_folder / img_path
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = img.flatten()
        img = img / 255.0
    return img

# %%
img = 'cropped_drawing.png'
process_img = preprocess_image(img)

process_img = scaler.transform([process_img])


# %%
pred = model.predict(process_img)
pred_shape = (label_coding[pred[0]])

print(f'The predicted shape is "{pred_shape}": {pred}')

"""
homeassistant_ip = "http://localhost:8123/api/"
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1ZjIwYzg5Nzk5ZDg0YTQ3OTUxMmFiYmYyODMzNWRjNCIsImlhdCI6MTczNTUyMzI4MywiZXhwIjoyMDUwODgzMjgzfQ.dfKgirQ7VVsHXqtX0X5cyYfv20H8w3E86BSefOw8f80"

with Client(
    homeassistant_ip,
    token
) as client:

    media_player = client.get_domain("media_player")

    media_player.turn_on(entity_id="media_player.family_room")"""