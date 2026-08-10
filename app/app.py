import base64
import io

import requests
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

ROBOFLOW_API_KEY = "6PiC3j6Kkj67AYQklopn"
ROBOFLOW_MODEL_ENDPOINT = "safety-helmet-detection-e4zru"
ROBOFLOW_MODEL_VERSION = "1"
CONFIDENCE_THRESHOLD = 40 

API_URL = (
    f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ENDPOINT}/{ROBOFLOW_MODEL_VERSION}"
    f"?api_key={ROBOFLOW_API_KEY}&confidence={CONFIDENCE_THRESHOLD}"
)

BOX_COLORS = {
    "person": "#3b82f6",
    "helmet": "#22c55e",
    "without_helmet": "#ef4444",
}


def predict(image: Image.Image):
    if image is None:
        return None, "Please upload an image first."

    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = requests.post(
        API_URL,
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    if response.status_code != 200:
        return image, f"Error calling model: {response.status_code} - {response.text}"

    result = response.json()
    predictions = result.get("predictions", [])

    draw_image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(draw_image)

    summary_lines = []
    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        cls = pred["class"]
        conf = pred["confidence"]

        left, top = x - w / 2, y - h / 2
        right, bottom = x + w / 2, y + h / 2

        color = BOX_COLORS.get(cls, "#facc15")
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        label = f"{cls} {conf:.2f}"
        draw.rectangle([left, top - 18, left + len(label) * 7, top], fill=color)
        draw.text((left + 2, top - 17), label, fill="white")

        summary_lines.append(f"- {cls}  (confidence: {conf:.2f})")

    summary = (
        "\n".join(summary_lines) if summary_lines else "No objects detected above the confidence threshold."
    )
    return draw_image, summary


with gr.Blocks(title="Safety Helmet Detection - YOLOv8") as demo:
    gr.Markdown(
        """
        # 🪖 Safety Helmet Detection (YOLOv8)
        Upload an image to detect **Person**, **Helmet**, and **Without Helmet**.
        This demo calls the trained model hosted on Roboflow.
        """
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Detect", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Detections")
            output_text = gr.Textbox(label="Summary", lines=6)

    submit_btn.click(fn=predict, inputs=input_image, outputs=[output_image, output_text])

if __name__ == "__main__":
    demo.launch()
