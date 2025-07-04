# Multi-Approach-ImageFusion
Interactive GUI for fusing CT and MRI brain scans using multiple image fusion methods including wavelet and Laplacian pyramid. Built with Python, OpenCV, Tkinter, and PyWavelets.

**Author:** Milad Jafari Barani 

**Date:** July 2025

## Description

This project provides an interactive graphical user interface (GUI) for fusing brain CT and MRI images using multiple image fusion techniques including wavelet transform and Laplacian pyramid methods. It leverages Python libraries such as OpenCV, PyWavelets, Tkinter, and Matplotlib to perform image processing and visualization.

## Features

- Loads paired CT and MRI brain images from a predefined dataset structure.
- Supports several fusion methods:
  - Average
  - Maximum
  - Minimum
  - Weighted Average
  - Wavelet-based fusion
  - Laplacian Pyramid fusion
- Interactive navigation through image pairs (Next / Previous).
- Real-time image fusion updates upon method or parameter changes.
- Save fused images with one click.
- Automatically handles resizing and color mapping of fused images for visualization.

## Dataset Structure

Place your dataset inside the project folder with the following structure:

Dataset/

├── trainA/ # CT images (grayscale)

└── trainB/ # MRI images (grayscale)


The program assumes images in `trainA` and `trainB` are paired by order (first image in `trainA` matches first image in `trainB`, and so on).

## Installation

Make sure you have Python 3.x installed. Then install the required packages:

bash
pip install opencv-python numpy matplotlib pywavelets pillow

## Notes
If images have different sizes, MRI images will be resized to match CT images automatically.

The fusion methods can be selected via dropdown menus.

The GUI shows the current sample index out of the total available pairs.

Feel free to open issues or submit pull requests to improve the project.

