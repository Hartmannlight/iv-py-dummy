from ultralytics import YOLO
from time import sleep
from PIL import Image
import numpy as np
import os

generate = True
send = False
export = True

sourceFile = 'C:/Users/Nathaniel/Desktop/yt/short.mp4'
exportPath = 'C:/Users/Nathaniel/Desktop/yt/'

jsons = []
images = []

if generate:
    model = YOLO('yolov8n.pt')
    results = model(source=sourceFile, show=True, max_det=1, line_width=2)

    for i, frame in enumerate(results):
        bgr_img = frame.plot()
        rgb_img = bgr_img[..., ::-1]  # Kanäle von BGR zu RGB umordnen
        pil_img = Image.fromarray(rgb_img.astype(np.uint8))
        images.append(pil_img)
        jsons.append(frame.tojson())

else:
    for filename in os.listdir(exportPath):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(exportPath, filename)
            img = Image.open(img_path)
            images.append(img)

    with open(f"{exportPath}/labels.json", 'r') as f:
        jsons = f.read()

if export:
    for i, img in enumerate(images):
        img.save(f"{exportPath}/frame_{i}.jpg")

    with open(f"{exportPath}/labels.json", 'w') as f:
        f.write(str(jsons))

if send:
    print("grpc send")
    sleep(0.033)
