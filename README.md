RoboDraw

A fully automated pipeline that converts any image into hand-drawn style sketches using a custom-built CNC plotter powered by Arduino + GRBL + Servo control.


🚀 Project Overview

This project is a complete end-to-end drawing system that transforms digital images into physical sketches using a custom-built CNC machine.



It combines:

Computer vision

Generative stroke algorithms

G-code generation

Embedded systems (Arduino + GRBL)

Mechanical design (3D printed plotter)

The result is a system that produces organic, hand-drawn style outputs instead of robotic straight-line plots.



🧩 System Architecture

Image (PNG/JPG) -> Image Processing Pipeline (Python) -> Stroke Generation (Contours + Hatching + Noise) -> SVG (virtual representation) -> G-code Generation -> GRBL Controller (Arduino) -> CNC Plotter + Servo Pen ->  Physical Drawing


🛠️ Hardware Setup

Arduino Uno

CNC Shield (A4988 drivers)

NEMA 17 Stepper Motors (X/Y)

Servo Motor (Pen Up/Down)

GRBL Firmware (modified / miGRBL for servo control)

Custom 3D Printed Frame (A4 working area)

Linear rods + LM8UU bearings


💻 Software Stack

Python
 
OpenCV (edge detection)

PIL (image processing)

svgpathtools (SVG parsing)

GRBL (motion control firmware)

Universal G-code Sender (UGS)


✨ Key Features

🎨 1. Hand-Drawn Style Rendering

Uses Perlin noise to introduce natural wobble

Eliminates rigid robotic motion

Produces sketch-like aesthetic

🔍 2. Intelligent Edge Detection

Extracts contours using:

Sobel filters (fallback)

OpenCV Canny (primary)

Captures important structural features of the image


🧱 3. Adaptive Hatching Algorithm

Converts grayscale intensity into stroke density

Simulates shading using layered hatch lines

Adjustable density via hatch_size


⚙️ 4. Stroke Optimization

Reorders paths to minimize travel distance

Reduces drawing time significantly

Improves machine efficiency


🔄 6. Unified Pipeline (No Manual Steps)

Single script:

image → strokes → SVG (in-memory) → G-code

No need to manually run multiple files


📂 Usage
Basic:
python pipeline.py input.png --gcode output.gcode

With preview:
python pipeline.py input.png --gcode output.gcode --preview preview.png

With tuning:
python pipeline.py input.png 
  --gcode output.gcode 
  --contour_simplify 2 
  --hatch_size 12
  
🎛️ Parameters Explained

contour_simplify - Controls edge detail:

1 → highly detailed, sketchy

2 → balanced

3 → simplified, clean

hatch_size - Controls shading density:

8 → dense shading (dark)

16 → balanced

32 → light shading

📸 Example Outputs# Robo-Draw
