# flask_app.py
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from rembg import remove
import tempfile
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create upload directory if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Check if files are present
        if 'content' not in request.files or 'style' not in request.files:
            return jsonify({"error": "Missing files"}), 400
            
        content_file = request.files['content']
        style_file = request.files['style']

        # Validate files
        if not (content_file.filename and style_file.filename):
            return jsonify({"error": "No selected files"}), 400
            
        if not (allowed_file(content_file.filename) and allowed_file(style_file.filename)):
            return jsonify({"error": "Invalid file type"}), 400

        # Save uploaded files
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content.png')
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style.png')
        content_file.save(content_path)
        style_file.save(style_path)

        # Process images
        content_img = cv2.imread(content_path)
        output = remove(content_img)  # Remove background
        foreground = output[:, :, :3]
        mask = output[:, :, 3]

        # Style transfer placeholder
        styled_bg = cv2.imread(style_path)

        # Resize and composite
        h, w = foreground.shape[:2]
        styled_bg = cv2.resize(styled_bg, (w, h))
        mask = cv2.resize(mask, (w, h))

        # Blend images
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=-1)
        result = (foreground * mask) + (styled_bg * (1 - mask))
        result = result.astype(np.uint8)

        # Save result
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
        cv2.imwrite(result_path, result)

        return send_file(result_path, mimetype='image/png')

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup temporary files
        for path in [content_path, style_path]:
            if path and os.path.exists(path):
                os.remove(path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)