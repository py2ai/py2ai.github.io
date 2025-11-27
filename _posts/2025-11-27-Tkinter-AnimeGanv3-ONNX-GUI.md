---
description: Step-by-step beginner-friendly tutorial to use AnimeGANv3 ONNX with a Tkinter GUI for transforming images into anime style.
featured-img: 20251127-animegan-gui/20251127-animegan-gui
keywords:
- Python
- AnimeGANv3
- ONNX
- Tkinter
- GUI
- image processing
- beginner
- tutorial
layout: post
mathjax: false
tags:
- python
- animegan
- onnx
- gui
- beginner
- tutorial
title: AnimeGANv3 ONNX GUI – Complete Beginner's Guide
---

## Introduction

AnimeGANv3 allows you to transform images into anime-style artwork using pre-trained ONNX models.  
In this tutorial, we will build a **Tkinter GUI application** that:

- Automatically installs required Python packages
- Lets you select an image
- Converts it into anime style using ONNX models
- Allows you to view, save, or revert to the original image

No prior experience with ONNX or GUI programming is needed.

---

## Understanding the Requirements

### Required Packages:

- `Pillow>=10` – for image processing
- `numpy` – numerical operations
- `opencv-contrib-python` – image manipulation
- `onnxruntime` – run ONNX models
- `tkinter` – GUI (usually comes with Python)

Our app will:

- Ensure required packages are installed automatically
- Download AnimeGANv3 models if missing
- Resize images for GUI display
- Transform images to anime style
- Save results to your disk

---

## Step 1: Setting Up the Environment
It is recommended to use Python >=3.8 and <=3.11. Open your terminal and install packages manually (optional, automatic installer included in code):

```bash
pip install Pillow>=10 numpy opencv-contrib-python onnxruntime
```

---

## Step 2: Automatic Package Installer

Our code automatically installs missing packages:

```python
import sys
import subprocess

required_packages = [
    "Pillow>=10",
    "numpy",
    "opencv-contrib-python",
    "onnxruntime"
]

def install_package(pkg):
    print(f"[INFO] Installing {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])

for pkg in required_packages:
    pkg_name = pkg.split('>=')[0]
    try:
        __import__(pkg_name)
    except ImportError:
        install_package(pkg)
```

---

## Step 3: Loading and Downloading ONNX Models

AnimeGANv3 comes with two styles: **Hayao** and **Shinkai**.

```python
import os
import urllib.request

STYLE_MODELS = {
    "Hayao": "AnimeGANv3_Hayao_36.onnx",
    "Shinkai": "AnimeGANv3_Shinkai_37.onnx",
}

BASE_URL = "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/"
MAX_DISPLAY_SIZE = 512

def get_model_path(style_name):
    fname = STYLE_MODELS.get(style_name)
    if fname is None:
        return None
    return os.path.join("models", fname)

def ensure_model(style_name):
    model_path = get_model_path(style_name)
    if model_path is None:
        return None
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        url = BASE_URL + STYLE_MODELS[style_name]
        urllib.request.urlretrieve(url, model_path)
        print(f"[INFO] Downloaded {style_name} model to {model_path}")
    return model_path
```

---

## Step 4: ONNXAnime Wrapper Class

This class handles image transformation:

```python
import cv2
import numpy as np
import onnxruntime as ort

class ONNXAnime:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape

    def transform(self, img_bgr_or_rgba):
        img = img_bgr_or_rgba
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        img_resized = cv2.resize(img, (target_w, target_h))
        img_resized = img_resized.astype(np.float32)/127.5 - 1.0
        tensor = np.transpose(img_resized, (2,0,1))[None, :, :, :]
        out = self.session.run(None, {self.input_name: tensor})[0]
        out_img = np.clip((out[0].transpose(1,2,0)+1.0)*127.5, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
```

---

## Step 5: Building the Tkinter GUI

```python
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

class AnimeApp:
    def __init__(self, master):
        self.master = master
        master.title("PyShine Anime GUI")
        master.geometry("450x700")

        tk.Label(master, text="Select image → choose style → Convert").pack(pady=8)
        frame = tk.Frame(master)
        frame.pack(pady=5)

        tk.Button(frame, text="Select Image", command=self.load_image).pack(side=tk.LEFT, padx=8)
        self.style_var = tk.StringVar(value="Hayao")
        ttk.OptionMenu(frame, self.style_var, self.style_var.get(), "Hayao", "Shinkai").pack(side=tk.LEFT)
        tk.Button(frame, text="Convert", command=self.convert).pack(side=tk.LEFT)
        tk.Button(frame, text="Save", command=self.save).pack(side=tk.LEFT)
        tk.Button(frame, text="Original", command=self.show_original).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(master, width=MAX_DISPLAY_SIZE, height=MAX_DISPLAY_SIZE, bg="gray")
        self.canvas.pack(pady=10)

        self.input_path = None
        self.output_img = None
        self.orig_img_pil = None
```

---

## Step 6: Loading and Displaying Images

```python
def load_image(self):
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not path: return
    self.input_path = path
    img = Image.open(path).convert("RGB")
    self.orig_img_pil = img.copy()
    display_w = int(img.width*MAX_DISPLAY_SIZE/max(img.width,img.height))
    display_h = int(img.height*MAX_DISPLAY_SIZE/max(img.width,img.height))
    img_display = img.resize((display_w, display_h), Image.Resampling.LANCZOS)
    self.tk_img = ImageTk.PhotoImage(img_display)
    self.canvas.config(width=display_w, height=display_h)
    self.canvas.create_image(display_w//2, display_h//2, image=self.tk_img)
```

---

## Step 7: Converting Images

```python
def convert(self):
    if not self.input_path: return
    style = self.style_var.get()
    model_path = ensure_model(style)
    img = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)
    anime = ONNXAnime(model_path)
    out = anime.transform(img)
    self.output_img = out
    out_display = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    out_display = out_display.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.Resampling.LANCZOS)
    self.tk_img = ImageTk.PhotoImage(out_display)
    self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, image=self.tk_img)
```

---

## Step 8: Saving and Showing Original Image

```python
def save(self):
    if self.output_img is None: return
    path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
    if not path: return
    cv2.imwrite(path, self.output_img)
    messagebox.showinfo("Saved", f"Saved to {path}")

def show_original(self):
    if self.orig_img_pil is None: return
    img_display = self.orig_img_pil.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.Resampling.LANCZOS)
    self.tk_img = ImageTk.PhotoImage(img_display)
    self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, image=self.tk_img)
```

---

## Step 9: Running the App

```python
if __name__ == "__main__":
    root = tk.Tk()
    AnimeApp(root)
    root.mainloop()
```

---
## Sample Images
![image1]({{"assets/img/posts/samples/image1.png" | absolute_url}} )
![image2]({{"assets/img/posts/samples/image2.png" | absolute_url}} )
![image3]({{"assets/img/posts/samples/image3.png" | absolute_url}} )

## Complete Code

{% include codeHeader.html %}
```python
import sys
import subprocess

# Fail-safe automatic installer for bare env
required_packages = [
    "Pillow>=10",
    "numpy",
    "opencv-contrib-python",
    "onnxruntime"
]

def install_package(pkg):
    print(f"[INFO] Installing {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])

for pkg in required_packages:
    pkg_name = pkg.split('>=')[0]
    try:
        __import__(pkg_name)
    except ImportError:
        install_package(pkg)


# Import all after installation
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import onnxruntime as ort
import os
import urllib.request

# Ensure Pillow has Resampling.LANCZOS
try:
    _ = Image.Resampling.LANCZOS
except AttributeError:
    print("[INFO] Pillow version is too old, upgrading...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "Pillow>=10"])
    from PIL import Image, ImageTk

print("[INFO] All required packages are installed and imported successfully!")


# AnimeGANv3 ONNX Model Config
STYLE_MODELS = {
    "Hayao": "AnimeGANv3_Hayao_36.onnx",
    "Shinkai": "AnimeGANv3_Shinkai_37.onnx",
}

BASE_URL = "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/"

MAX_DISPLAY_SIZE = 512  # Max width/height for GUI display

def get_model_path(style_name):
    fname = STYLE_MODELS.get(style_name)
    if fname is None:
        return None
    return os.path.join("models", fname)

def ensure_model(style_name):
    model_path = get_model_path(style_name)
    if model_path is None:
        return None
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        url = BASE_URL + STYLE_MODELS[style_name]
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"[INFO] Downloaded {style_name} model to {model_path}")
        except Exception as e:
            messagebox.showerror("Download error", f"Could not download {style_name} model:\n{e}")
            return None
    return model_path

def safe_dim(d, default=512):
    try:
        return int(d)
    except (TypeError, ValueError):
        return default


# ONNX AnimeGANv3 Wrapper
class ONNXAnime:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        print("[INFO] Model expects input shape:", self.input_shape)

    def transform(self, img_bgr_or_rgba):
        img = img_bgr_or_rgba
        if img is None:
            raise ValueError("Input image is None")

        orig_h, orig_w = img.shape[:2]
        print(f"[DEBUG] Original input image shape: {img.shape}")

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(self.input_shape) != 4:
            raise ValueError(f"Unsupported model input shape: {self.input_shape}")

        # Determine target height/width
        if self.input_shape[1] == 3:  # NCHW
            target_h = safe_dim(self.input_shape[2])
            target_w = safe_dim(self.input_shape[3])
        elif self.input_shape[3] == 3:  # NHWC
            target_h = safe_dim(self.input_shape[1])
            target_w = safe_dim(self.input_shape[2])
        else:
            raise ValueError(f"Unsupported model input shape: {self.input_shape}")

        print(f"[DEBUG] Resizing input for model: ({target_h}, {target_w})")
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img_resized = img_resized.astype(np.float32) / 127.5 - 1.0

        if self.input_shape[1] == 3:  # NCHW
            tensor = np.transpose(img_resized, (2,0,1))[None, :, :, :]
        else:  # NHWC
            tensor = img_resized[None, :, :, :]

        out = self.session.run(None, {self.input_name: tensor})[0]
        print(f"[DEBUG] Raw model output shape: {out.shape}")

        if out.ndim == 4 and out.shape[1] == 3:  # NCHW
            out_img = out[0].transpose(1,2,0)
        elif out.ndim == 4 and out.shape[3] == 3:  # NHWC
            out_img = out[0]
        else:
            raise ValueError(f"Unsupported model output shape: {out.shape}")

        out_img = np.clip((out_img + 1.0) * 127.5, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        print(f"[DEBUG] Converted output image shape (before GUI resize): {out_bgr.shape}")
        return out_bgr


# GUI App
class AnimeApp:
    def __init__(self, master):
        self.master = master
        master.title("PyShine Anime GUI")
        master.geometry("450x700")

        tk.Label(master, text="Select image → choose style → Convert").pack(pady=8)

        frame = tk.Frame(master)
        frame.pack(pady=5)

        
        # OptionMenu for style selection 
        self.style_var = tk.StringVar()
        self.style_var.set("Hayao")  # default value
        ttk.OptionMenu(frame, self.style_var, self.style_var.get(), *STYLE_MODELS.keys()).pack(side=tk.LEFT, padx=0)
        tk.Button(frame, text="Select", command=self.load_image).pack(side=tk.LEFT, padx=0)
        tk.Button(frame, text="Convert", command=self.convert).pack(side=tk.LEFT, padx=0)
        tk.Button(frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=0)
        tk.Button(frame, text="Original", command=self.show_original).pack(side=tk.LEFT, padx=0)

        # Canvas for image display 
        self.canvas = tk.Canvas(master, width=MAX_DISPLAY_SIZE, height=MAX_DISPLAY_SIZE, bg="gray")
        self.canvas.pack(pady=10)

        # Variables 
        self.input_path = None
        self.output_img = None
        self.orig_img_pil = None  # original PIL image for showing anytime
        self.orig_w = 512
        self.orig_h = 512
        self.display_scale = 1.0
        self.tk_img = None  # reference to Tkinter image


    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self.input_path = path
        img = Image.open(path).convert("RGB")
        self.orig_img_pil = img.copy()  # save original for Show Original
        self.orig_w, self.orig_h = img.size

        scale = min(MAX_DISPLAY_SIZE/self.orig_w, MAX_DISPLAY_SIZE/self.orig_h, 1.0)
        self.display_scale = scale
        display_w = int(self.orig_w * scale)
        display_h = int(self.orig_h * scale)

        img_display = img.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_display)
        self.canvas.config(width=display_w, height=display_h)
        self.canvas.create_image(display_w//2, display_h//2, image=self.tk_img)
        print(f"[DEBUG] GUI displaying input image shape: {display_w}x{display_h}")


    def show_original(self):
        if self.orig_img_pil is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return

        display_w = int(self.orig_w * self.display_scale)
        display_h = int(self.orig_h * self.display_scale)
        img_display = self.orig_img_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(img_display)
        self.canvas.config(width=display_w, height=display_h)
        self.canvas.create_image(display_w//2, display_h//2, image=self.tk_img)
        print(f"[DEBUG] GUI displaying original input image shape: {display_w}x{display_h}")


    def convert(self):
        if not self.input_path:
            messagebox.showwarning("No image", "Please select an image first.")
            return
        style = self.style_var.get()
        model_path = ensure_model(style)
        if not model_path:
            return

        try:
            img = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)
            anime = ONNXAnime(model_path)
            out = anime.transform(img)
            self.output_img = out

            # Resize to original image for saving
            out_display = cv2.resize(out, (self.orig_w, self.orig_h), interpolation=cv2.INTER_AREA)

            # Resize for GUI display
            display_w = int(self.orig_w * self.display_scale)
            display_h = int(self.orig_h * self.display_scale)
            out_display_pil = Image.fromarray(cv2.cvtColor(out_display, cv2.COLOR_BGR2RGB))
            out_display_pil = out_display_pil.resize((display_w, display_h), Image.Resampling.LANCZOS)

            self.tk_img = ImageTk.PhotoImage(out_display_pil)
            self.canvas.config(width=display_w, height=display_h)
            self.canvas.create_image(display_w//2, display_h//2, image=self.tk_img)

            print(f"[DEBUG] GUI displaying output image shape: {display_w}x{display_h}")

        except Exception as e:
            messagebox.showerror("Failed to convert image", f"{e}")


    def save(self):
        if self.output_img is None:
            messagebox.showwarning("No output", "No anime image to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
        if not path:
            return

        save_img = cv2.resize(self.output_img, (self.orig_w, self.orig_h), interpolation=cv2.INTER_AREA)
        print(f"[DEBUG] Saving output image resized shape: {save_img.shape}")
        cv2.imwrite(path, save_img)
        messagebox.showinfo("Saved", f"Saved to {path}")


# Run App
if __name__ == "__main__":
    root = tk.Tk()
    # Define a style for the OptionMenu
    style = ttk.Style(root)
    style.theme_use('clam')  # or 'default', 'alt'
    style.configure('TMenubutton', foreground='black', background='white', font=('Arial', 12))

    style_var = tk.StringVar(value="Hayao")
    option = ttk.OptionMenu(root, style_var, style_var.get(), "Hayao", "Shinkai")
    # option.pack(padx=20, pady=20)
    AnimeApp(root)
    root.mainloop()

```

## Conclusion

You now have a **fully functional AnimeGANv3 ONNX GUI**!  
You learned:

- Automatic package installation
- Downloading ONNX models
- Transforming images with ONNX
- Displaying images in Tkinter
- Saving results and viewing original image

Experiment with different styles and images to create your own anime artwork!

---

**Website:** https://www.pyshine.com  
**Author:** PyShine

