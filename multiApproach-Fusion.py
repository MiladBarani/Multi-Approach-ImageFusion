"""
Medical Image Fusion GUI using Tkinter, OpenCV, and PyWavelets
Author: Milad Jafari Barani
Date: July 2025

Description:
------------
This Python script provides a graphical user interface (GUI) for fusing brain CT and MRI images 
using various image fusion methods including average, max, minimum, weighted average, wavelet-based fusion, 
and Laplacian pyramid fusion.

The interface automatically loads paired CT and MRI images from a dataset folder structure:
    Dataset/images
        ├── trainA/   ← CT images
        └── trainB/   ← MRI images

Features:
---------
- Loads and displays paired medical images (MRI & CT)
- Supports six fusion techniques
- Uses PyWavelets for wavelet-based fusion
- Supports interactive navigation (next/previous sample)
- Automatically resizes and colorizes the output
- Saves fused images with a single click

Requirements:
-------------
- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- PyWavelets
- PIL (Pillow)
- tkinter (built-in)

Usage:
------
1. Place CT and MRI images into 'Dataset/trainA' and 'Dataset/trainB' respectively.
2. Run this script.
3. Use the GUI to browse, fuse, and save results.

Note:
-----
This script assumes that images are paired by order (not filename matching).
"""



import os
import cv2
import numpy as np
import pywt
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import random

# Paths to CT and MRI folders
CT_PATH = "ImageFusion/Dataset/images/trainA"
MRI_PATH = "ImageFusion/Dataset/images/trainB"

# Load filenames
ct_files = sorted(os.listdir(CT_PATH))
mri_files = sorted(os.listdir(MRI_PATH))
paired_samples = list(zip(ct_files, mri_files))
random.shuffle(paired_samples)

# Globals
current_index = 0
mri = None
ct = None
fused_image = None

def load_current_images():
    global mri, ct
    ct_file, mri_file = paired_samples[current_index]
    ct = cv2.imread(os.path.join(CT_PATH, ct_file), cv2.IMREAD_GRAYSCALE)
    mri = cv2.imread(os.path.join(MRI_PATH, mri_file), cv2.IMREAD_GRAYSCALE)
    if mri.shape != ct.shape:
        mri = cv2.resize(mri, (ct.shape[1], ct.shape[0]))
    update_images()

def fuse_images(method, wavelet):
    if method == "Average":
        fused = (mri + ct) / 2
    elif method == "Max":
        fused = np.maximum(mri, ct)
    elif method == "Wavelet":
        coeffs_mri = pywt.dwt2(mri, wavelet)
        coeffs_ct = pywt.dwt2(ct, wavelet)
        ca_fused = (coeffs_mri[0] + coeffs_ct[0]) / 2
        ch_fused = np.maximum(coeffs_mri[1][0], coeffs_ct[1][0])
        cv_fused = np.maximum(coeffs_mri[1][1], coeffs_ct[1][1])
        cd_fused = np.maximum(coeffs_mri[1][2], coeffs_ct[1][2])
        fused = pywt.idwt2((ca_fused, (ch_fused, cv_fused, cd_fused)), wavelet)
        fused = np.clip(fused, 0, 255)
    elif method == "Minimum":
        fused = np.minimum(mri, ct)
    elif method == "Weighted Average":
        alpha = 0.6
        fused = alpha * mri + (1 - alpha) * ct
    elif method == "Laplacian Pyramid":
        fused = laplacian_fusion(mri, ct)
    else:
        fused = mri
    return np.uint8(fused)

def laplacian_fusion(img1, img2):
    gp1, gp2 = [img1], [img2]
    for i in range(4):
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
        gp1.append(img1)
        gp2.append(img2)
    lp1 = [gp1[3]]
    lp2 = [gp2[3]]
    for i in range(3, 0, -1):
        ge1 = cv2.pyrUp(gp1[i])
        ge2 = cv2.pyrUp(gp2[i])
        lp1.append(gp1[i - 1] - ge1)
        lp2.append(gp2[i - 1] - ge2)
    lp_fused = [np.maximum(l1, l2) for l1, l2 in zip(lp1, lp2)]
    fused = lp_fused[0]
    for i in range(1, 4):
        fused = cv2.pyrUp(fused) + lp_fused[i]
    return np.clip(fused, 0, 255)

def update_images(*args):
    global fused_image
    method = fusion_method.get()
    wavelet = wavelet_type.get()
    fused_image = fuse_images(method, wavelet)
    display_images()

def display_images():
    mri_resized = cv2.resize(mri, (256, 256))
    ct_resized = cv2.resize(ct, (256, 256))
    fused_resized = cv2.resize(fused_image, (256, 256))

    mri_img = ImageTk.PhotoImage(Image.fromarray(mri_resized))
    ct_img = ImageTk.PhotoImage(Image.fromarray(ct_resized))

    fused_colored = plt.cm.jet(fused_resized / 255.0)[:, :, :3] * 255
    fused_colored = np.uint8(fused_colored)
    fused_img = ImageTk.PhotoImage(Image.fromarray(fused_colored))

    mri_label.config(image=mri_img)
    mri_label.image = mri_img

    ct_label.config(image=ct_img)
    ct_label.image = ct_img

    fused_label.config(image=fused_img)
    fused_label.image = fused_img

    status_label.config(text=f"Sample {current_index + 1} / {len(paired_samples)}")

def next_sample():
    global current_index
    if current_index < len(paired_samples) - 1:
        current_index += 1
        load_current_images()

def prev_sample():
    global current_index
    if current_index > 0:
        current_index -= 1
        load_current_images()

def save_fused_image():
    if fused_image is not None:
        filename = f"fused_{current_index + 1}.png"
        cv2.imwrite(filename, fused_image)
        print(f"Saved: {filename}")

# GUI
root = tk.Tk()
root.title("Medical Image Fusion (Dataset GUI)")

fusion_method = tk.StringVar(value="Average")
wavelet_type = tk.StringVar(value="haar")

methods = ["Average", "Max", "Wavelet", "Minimum", "Weighted Average", "Laplacian Pyramid"]
wavelets = pywt.wavelist(kind='discrete')

# Bind changes to update
fusion_method.trace("w", update_images)
wavelet_type.trace("w", update_images)

mri_label = tk.Label(root, text="MRI", relief="solid")
mri_label.grid(row=0, column=0, padx=5, pady=5)

ct_label = tk.Label(root, text="CT", relief="solid")
ct_label.grid(row=0, column=1, padx=5, pady=5)

fused_label = tk.Label(root)
fused_label.grid(row=0, column=2, padx=5, pady=5)

ttk.Combobox(root, textvariable=fusion_method, values=methods, state="readonly").grid(row=1, column=0, padx=10, pady=10)
ttk.Combobox(root, textvariable=wavelet_type, values=wavelets, state="readonly").grid(row=1, column=1, padx=10, pady=10)
tk.Button(root, text="Save", command=save_fused_image).grid(row=1, column=2, padx=10, pady=10)

tk.Button(root, text="<< Previous", command=prev_sample).grid(row=2, column=0, pady=5)
tk.Label(root, text="").grid(row=2, column=1)
tk.Button(root, text="Next >>", command=next_sample).grid(row=2, column=2, pady=5)

status_label = tk.Label(root, text="")
status_label.grid(row=3, column=0, columnspan=3)

# Load and show first pair
load_current_images()
root.mainloop()
