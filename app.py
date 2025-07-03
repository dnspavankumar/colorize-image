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
        master.geometry("900x600")
        master.configure(bg="#23272f")

        self.input_path = None
        self.output_path = None
        self.input_img = None
        self.output_img = None

        # Title
        title = tk.Label(master, text="Image Colorization App", font=("Segoe UI", 24, "bold"), fg="#00bfff", bg="#23272f")
        title.pack(pady=(20, 5))

        # Instructions
        instructions = tk.Label(master, text="Select a grayscale image to colorize. The colorized result will appear on the right.", font=("Segoe UI", 12), fg="#b0b3b8", bg="#23272f")
        instructions.pack(pady=(0, 10))

        # Controls frame (just below instructions)
        controls_frame = tk.Frame(master, bg="#23272f")
        controls_frame.pack(pady=(0, 20))

        self.select_button = tk.Button(controls_frame, text="Select Image", command=self.select_image, font=("Segoe UI", 11), bg="#00bfff", fg="#fff", activebackground="#009acd", activeforeground="#fff", relief=tk.FLAT, padx=20, pady=8)
        self.select_button.grid(row=0, column=0, padx=10)

        self.colorize_button = tk.Button(controls_frame, text="Colorize Image", command=self.colorize, state=tk.DISABLED, font=("Segoe UI", 11), bg="#4caf50", fg="#fff", activebackground="#388e3c", activeforeground="#fff", relief=tk.FLAT, padx=20, pady=8)
        self.colorize_button.grid(row=0, column=1, padx=10)

        self.save_button = tk.Button(controls_frame, text="Save Colorized Image", command=self.save_image, state=tk.DISABLED, font=("Segoe UI", 11), bg="#ff9800", fg="#fff", activebackground="#f57c00", activeforeground="#fff", relief=tk.FLAT, padx=20, pady=8)
        self.save_button.grid(row=0, column=2, padx=10)

        # Main frame for images
        main_frame = tk.Frame(master, bg="#23272f")
        main_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        # Centering container for image frames
        center_frame = tk.Frame(main_frame, bg="#23272f")
        center_frame.pack(expand=True)

        # Image frames with centered labels
        img_frame = tk.Frame(center_frame, bg="#23272f")
        img_frame.pack(anchor="center")

        # Labels row
        orig_label = tk.Label(img_frame, text="Original", font=("Segoe UI", 13, "bold"), fg="#b0b3b8", bg="#23272f")
        orig_label.grid(row=0, column=0, pady=(0, 10), padx=40, sticky="n")
        color_label = tk.Label(img_frame, text="Colorized", font=("Segoe UI", 13, "bold"), fg="#b0b3b8", bg="#23272f")
        color_label.grid(row=0, column=1, pady=(0, 10), padx=40, sticky="n")

        # Image panels
        self.orig_img_panel = tk.Label(img_frame, bg="#2c2f36", relief=tk.RIDGE, bd=2)
        self.orig_img_panel.grid(row=1, column=0, padx=40, pady=5)
        self.color_img_panel = tk.Label(img_frame, bg="#2c2f36", relief=tk.RIDGE, bd=2)
        self.color_img_panel.grid(row=1, column=1, padx=40, pady=5)

        # Status label
        self.status_label = tk.Label(master, text="Ready", font=("Segoe UI", 11), fg="#b0b3b8", bg="#23272f")
        self.status_label.pack(pady=(0, 10))

    def select_image(self):
        filetypes = (('Image files', '*.png *.jpg *.jpeg'), ('All files', '*.*'))
        path = filedialog.askopenfilename(title='Open image', filetypes=filetypes)
        if path:
            self.input_path = path
            self.display_image(self.input_path, panel="orig")
            self.colorize_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)
            self.status_label.config(text="Ready")
            self.display_image(None, panel="color")  # Clear colorized panel

    def colorize(self):
        if not self.input_path:
            messagebox.showerror("Error", "No image selected.")
            return
        output_path = os.path.join(os.path.dirname(self.input_path), "colorized_" + os.path.basename(self.input_path))
        try:
            self.status_label.config(text="Colorizing... Please wait.")
            self.master.update_idletasks()
            colorize_image(self.input_path, output_path)
            self.output_path = output_path
            self.display_image(self.output_path, panel="color")
            self.save_button.config(state=tk.NORMAL)
            self.status_label.config(text="Done!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to colorize image: {e}")
            self.status_label.config(text="Error during colorization.")

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
                self.status_label.config(text=f"Image saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")
                self.status_label.config(text="Error saving image.")

    def display_image(self, path, panel="orig"):
        if path is None:
            if panel == "orig":
                self.orig_img_panel.configure(image="")
                setattr(self.orig_img_panel, "image", None)
            else:
                self.color_img_panel.configure(image="")
                setattr(self.color_img_panel, "image", None)
            return
        img = Image.open(path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        if panel == "orig":
            self.orig_img_panel.configure(image=img_tk)
            setattr(self.orig_img_panel, "image", img_tk)
        else:
            self.color_img_panel.configure(image=img_tk)
            setattr(self.color_img_panel, "image", img_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorizationApp(root)
    root.mainloop() 