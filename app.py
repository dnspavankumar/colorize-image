import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
from werkzeug.utils import secure_filename
import torch
import colorizers
import numpy as np
import skimage.color as color
import skimage.io as io
import skimage.transform
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///colorize.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class ImageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    output = db.Column(db.Boolean, default=False)  # True if in outputs folder

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

def cleanup_old_files():
    expire_time = datetime.utcnow() - timedelta(hours=1)
    old_records = ImageRecord.query.filter(ImageRecord.upload_time < expire_time).all()
    for record in old_records:
        folder = OUTPUT_FOLDER if record.output else UPLOAD_FOLDER
        file_path = os.path.join(folder, record.filename)
        try:
            os.remove(file_path)
        except Exception:
            pass
        db.session.delete(record)
    db.session.commit()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        cleanup_old_files()
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
            # Save upload record
            db.session.add(ImageRecord(filename=filename, upload_time=datetime.utcnow(), output=False))
            db.session.commit()
            colorize_image(input_path, output_path)
            # Save output record
            db.session.add(ImageRecord(filename=filename, upload_time=datetime.utcnow(), output=True))
            db.session.commit()
            return redirect(url_for('result', filename=filename))
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    record = ImageRecord.query.filter_by(filename=filename, output=True).first()
    if not record:
        abort(404)
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

# Minimal health check route for Render
@app.route('/health')
def health():
    return "ok", 200

if __name__ == '__main__':
    print("Starting app...")
    with app.app_context():
        print("Creating database...")
        db.create_all()
        print("Database created.")
    port = int(os.environ.get("PORT", 10000))
    print(f"About to start server on port {port}")
    from waitress import serve
    serve(app, host="0.0.0.0", port=port)
    print("Server started.") 