import io
from PIL import Image
import cv2
from flask import Flask, render_template, request, Response
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('best1.pt')  # Load model once at the start

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/gellary")
def gellary():
    return render_template("gellary.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict_img", methods=["POST"])
def predict_img():
    if 'file' in request.files:
        f = request.files['file']
        in_memory_file = io.BytesIO()
        f.save(in_memory_file)
        in_memory_file.seek(0)

        img = Image.open(in_memory_file).convert('RGB')  # Ensure RGB format
        results = model(img)
        res_plotted = results[0].plot()

        # Directly encode to JPEG without color conversion
        _, img_encoded = cv2.imencode('.jpg', res_plotted)
        response = img_encoded.tobytes()

        return Response(response, mimetype='image/jpeg')

    return "No image uploaded or unsupported format."

@app.route("/predict_video", methods=["POST"])
def predict_video():
    if 'file' in request.files:
        f = request.files['file']
        in_memory_file = io.BytesIO()
        f.save(in_memory_file)
        in_memory_file.seek(0)

        video_array = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)
        cap = cv2.VideoCapture(cv2.CAP_FFMPEG)
        cap.open(io.BytesIO(video_array))

        def generate_frames():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                res_plotted = results[0].plot()

                # No color conversion
                _, buffer = cv2.imencode('.jpg', res_plotted)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            cap.release()

        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return "No video uploaded or unsupported format."

@app.route("/webcam_feed")
def webcam_feed():
    cap = cv2.VideoCapture(0)

    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            res_plotted = results[0].plot()

            # No color conversion
            _, buffer = cv2.imencode('.jpg', res_plotted)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
