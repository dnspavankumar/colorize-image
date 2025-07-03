import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import colorizers
import skimage.color as color
import skimage.io as io
import skimage.transform

# Load the colorizer model once at startup
print("Loading model...")
try:
    colorizer = colorizers.eccv16().eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Colorization logic
def colorize_image(input_path, output_path):
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
    return output_path

class ColorizationApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Colorization App")
        master.geometry("600x500")

        self.input_path = None
        self.output_path = None
        self.input_img = None
        self.output_img = None

        self.label = tk.Label(master, text="Select a grayscale image to colorize:")
        self.label.pack(pady=10)

        self.select_button = tk.Button(master, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=5)

        self.colorize_button = tk.Button(master, text="Colorize Image", command=self.colorize, state=tk.DISABLED)
        self.colorize_button.pack(pady=5)

        self.save_button = tk.Button(master, text="Save Colorized Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        self.img_panel = tk.Label(master)
        self.img_panel.pack(pady=10)

    def select_image(self):
        filetypes = (('Image files', '*.png *.jpg *.jpeg'), ('All files', '*.*'))
        path = filedialog.askopenfilename(title='Open image', filetypes=filetypes)
        if path:
            self.input_path = path
            self.display_image(self.input_path)
            self.colorize_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)

    def colorize(self):
        if not self.input_path:
            messagebox.showerror("Error", "No image selected.")
            return
        output_path = os.path.join(os.path.dirname(self.input_path), "colorized_" + os.path.basename(self.input_path))
        try:
            colorize_image(self.input_path, output_path)
            self.output_path = output_path
            self.display_image(self.output_path)
            self.save_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to colorize image: {e}")

    def save_image(self):
        if not self.output_path:
            messagebox.showerror("Error", "No colorized image to save.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension='.jpg', filetypes=[('JPEG', '*.jpg'), ('PNG', '*.png')])
        if save_path:
            try:
                img = Image.open(self.output_path)
                img.save(save_path)
                messagebox.showinfo("Saved", f"Image saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")

    def display_image(self, path):
        img = Image.open(path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.img_panel.configure(image=img_tk)
        self.img_panel.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorizationApp(root)
    root.mainloop() 