from flask import Flask, render_template, request, redirect
import os
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models.detection as detection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

app = Flask(__name__)
# Загрузка предварительно обученных моделей для обнаружения оружия и лиц
weapon_detection_model = detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
weapon_detection_model.eval()
face_detection_model = detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
face_detection_model.eval()

UPLOAD_FOLDER = 'pythonProject'
app.config['pythonProject'] = UPLOAD_FOLDER


# Определение функции process_image, как в предыдущем ответе
def process_image(image_path):
    image = cv2.imread(image_path)
    image_tensor = transforms.ToTensor()(image)

    with torch.no_grad():
        weapon_predictions = weapon_detection_model([image_tensor])

    if len(weapon_predictions[0]['scores']) > 0:
        with torch.no_grad():
            face_predictions = face_detection_model([image_tensor])

        if len(face_predictions[0]['scores']) > 0:
            return "Оружие обнаружено и лицо распознано."

        else:
            return "Оружие обнаружено, но лицо не распознано."

    else:
        return "Оружие не обнаружено."


def run_app():
    app.run(host='0.0.0.0', port=5000)


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = os.path.join(app.config['pythonProject'], file.filename)
            file.save(filename)
            result = process_image(filename)

    return render_template('index1.html', result=result)


if __name__ == '__main__':
    run_app()
