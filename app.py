import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import torch
import colorizers
import numpy as np
import skimage.color as color
import skimage.io as io
import skimage.transform

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def colorize_image(input_path, output_path):
    colorizer = colorizers.eccv16().eval()
    img = io.imread(input_path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img_lab = color.rgb2lab(img)
    img_l = img_lab[:,:,0]
    (H_orig,W_orig) = img_l.shape
    img_l_rs = skimage.transform.resize(img_l, (256,256))
    img_l_rs = img_l_rs[np.newaxis,np.newaxis,:,:]
    img_l_rs = torch.from_numpy(img_l_rs).float()
    with torch.no_grad():
        out_ab = colorizer(img_l_rs)
    out_ab = out_ab.cpu().numpy()
    out_ab = out_ab[0].transpose((1,2,0))
    out_ab = skimage.transform.resize(out_ab, (H_orig, W_orig))
    img_lab_out = np.concatenate((img_l[:,:,np.newaxis], out_ab), axis=2)
    img_rgb_out = np.clip(color.lab2rgb(img_lab_out), 0, 1)
    io.imsave(output_path, (img_rgb_out*255).astype(np.uint8))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            file.save(input_path)
            colorize_image(input_path, output_path)
            return redirect(url_for('result', filename=filename))
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=5000)
    app.run(debug=True) 