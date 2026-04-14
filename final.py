#!/usr/bin/env python3
from __future__ import annotations

# Exact-logic combined pipeline without needing to keep the SVG on disk:
# - Stage 1 logic matches the original image -> SVG code
# - Stage 2 logic matches the original SVG -> G-code code
# - The SVG is generated in memory and, only if needed for parsing, written to a
#   temporary file internally and deleted automatically unless --keep-svg is used.

from random import *
import math
import argparse
import tempfile
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageOps

import cv2
import numpy as np
from svgpathtools import svg2paths2

Point = Tuple[float, float]

# -----------------------------
# Original helper code: filters.py
# -----------------------------
F_Blur = {
    (-2,-2):2,(-1,-2):4,(0,-2):5,(1,-2):4,(2,-2):2,
    (-2,-1):4,(-1,-1):9,(0,-1):12,(1,-1):9,(2,-1):4,
    (-2,0):5,(-1,0):12,(0,0):15,(1,0):12,(2,0):5,
    (-2,1):4,(-1,1):9,(0,1):12,(1,1):9,(2,1):4,
    (-2,2):2,(-1,2):4,(0,2):5,(1,2):4,(2,2):2,
}
F_SobelX = {(-1,-1):1,(0,-1):0,(1,-1):-1,(-1,0):2,(0,0):0,(1,0):-2,(-1,1):1,(0,1):0,(1,1):-1}
F_SobelY = {(-1,-1):1,(0,-1):2,(1,-1):1,(-1,0):0,(0,0):0,(1,0):0,(-1,1):-1,(0,1):-2,(1,1):-1}

def appmask(IM,masks):
    PX = IM.load()
    w,h = IM.size
    NPX = {}
    for x in range(0,w):
        for y in range(0,h):
            a = [0]*len(masks)
            for i in range(len(masks)):
                for p in masks[i].keys():
                    if 0<x+p[0]<w and 0<y+p[1]<h:
                        a[i] += PX[x+p[0],y+p[1]] * masks[i][p]
                if sum(masks[i].values())!=0:
                    a[i] = a[i] / sum(masks[i].values())
            NPX[x,y]=int(sum([v**2 for v in a])**0.5)
    for x in range(0,w):
        for y in range(0,h):
            PX[x,y] = NPX[x,y]

# -----------------------------
# Original helper code: perlin.py
# -----------------------------
PERLIN_YWRAPB = 4
PERLIN_YWRAP = 1<<PERLIN_YWRAPB
PERLIN_ZWRAPB = 8
PERLIN_ZWRAP = 1<<PERLIN_ZWRAPB
PERLIN_SIZE = 4095

perlin_octaves = 4
perlin_amp_falloff = 0.5

def scaled_cosine(i):
    return 0.5*(1.0-math.cos(i*math.pi))

perlin = None

def perlin_noise(x,y=0,z=0):
    global perlin
    import random
    if perlin == None:
        perlin = []
        for i in range(0,PERLIN_SIZE+1):
            perlin.append(random.random())
    if x<0:x=-x
    if y<0:y=-y
    if z<0:z=-z

    xi,yi,zi = int(x),int(y),int(z)
    xf = x-xi
    yf = y-yi
    zf = z-zi
    rxf = ryf = None

    r = 0
    ampl = 0.5

    n1 = n2 = n3 = None
    for o in range(0,perlin_octaves):
        of=xi+(yi<<PERLIN_YWRAPB)+(zi<<PERLIN_ZWRAPB)

        rxf = scaled_cosine(xf)
        ryf = scaled_cosine(yf)

        n1  = perlin[of&PERLIN_SIZE]
        n1 += rxf*(perlin[(of+1)&PERLIN_SIZE]-n1)
        n2  = perlin[(of+PERLIN_YWRAP)&PERLIN_SIZE]
        n2 += rxf*(perlin[(of+PERLIN_YWRAP+1)&PERLIN_SIZE]-n2)
        n1 += ryf*(n2-n1)

        of += PERLIN_ZWRAP
        n2  = perlin[of&PERLIN_SIZE]
        n2 += rxf*(perlin[(of+1)&PERLIN_SIZE]-n2)
        n3  = perlin[(of+PERLIN_YWRAP)&PERLIN_SIZE]
        n3 += rxf*(perlin[(of+PERLIN_YWRAP+1)&PERLIN_SIZE]-n3)
        n2 += ryf*(n3-n2)

        n1 += scaled_cosine(zf)*(n2-n1)

        r += n1*ampl
        ampl *= perlin_amp_falloff
        xi<<=1
        xf*=2
        yi<<=1
        yf*=2
        zi<<=1
        zf*=2

        if (xf>=1.0): xi+=1; xf-=1
        if (yf>=1.0): yi+=1; yf-=1
        if (zf>=1.0): zi+=1; zf-=1
    return r

# -----------------------------
# Original helper code: util.py
# -----------------------------
def midpt(*args):
    xs,ys = 0,0
    for p in args:
        xs += p[0]
        ys += p[1]
    return xs/len(args),ys/len(args)

def distsum(*args):
    return sum([ ((args[i][0]-args[i-1][0])**2 + (args[i][1]-args[i-1][1])**2)**0.5 for i in range(1,len(args))])

# -----------------------------
# Original helper code: strokesort.py
# -----------------------------
def sortlines(lines):
    print("optimizing stroke sequence...")
    clines = lines[:]
    slines = [clines.pop(0)]
    while clines != []:
        x,s,r = None,1000000,False
        for l in clines:
            d = distsum(l[0],slines[-1][-1])
            dr = distsum(l[-1],slines[-1][-1])
            if d < s:
                x,s,r = l[:],d,False
            if dr < s:
                x,s,r = l[:],s,True

        clines.remove(x)
        if r == True:
            x = x[::-1]
        slines.append(x)
    return slines

# -----------------------------
# Original main stage-1 code
# -----------------------------
no_cv = False
draw_contours = True
draw_hatch = True
show_bitmap = False
resolution = 1024
hatch_size = 16
contour_simplify = 2

def find_edges(IM):
    print("finding edges...")
    if no_cv:
        appmask(IM,[F_SobelX,F_SobelY])
    else:
        im = np.array(IM)
        im = cv2.GaussianBlur(im,(3,3),0)
        im = cv2.Canny(im,100,200)
        IM = Image.fromarray(im)
    return IM.point(lambda p: p > 128 and 255)

def getdots(IM):
    print("getting contour points...")
    PX = IM.load()
    dots = []
    w,h = IM.size
    for y in range(h-1):
        row = []
        for x in range(1,w):
            if PX[x,y] == 255:
                if len(row) > 0:
                    if x-row[-1][0] == row[-1][-1]+1:
                        row[-1] = (row[-1][0],row[-1][-1]+1)
                    else:
                        row.append((x,0))
                else:
                    row.append((x,0))
        dots.append(row)
    return dots

def connectdots(dots):
    print("connecting contour points...")
    contours = []
    for y in range(len(dots)):
        for x,v in dots[y]:
            if v > -1:
                if y == 0:
                    contours.append([(x,y)])
                else:
                    closest = -1
                    cdist = 100
                    for x0,v0 in dots[y-1]:
                        if abs(x0-x) < cdist:
                            cdist = abs(x0-x)
                            closest = x0

                    if cdist > 3:
                        contours.append([(x,y)])
                    else:
                        found = 0
                        for i in range(len(contours)):
                            if contours[i][-1] == (closest,y-1):
                                contours[i].append((x,y,))
                                found = 1
                                break
                        if found == 0:
                            contours.append([(x,y)])
        for c in contours[:]:
            if c[-1][1] < y-1 and len(c)<4:
                contours.remove(c)
    return contours

def getcontours(IM,sc=2):
    print("generating contours...")
    IM = find_edges(IM)
    IM1 = IM.copy()
    IM2 = IM.rotate(-90,expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    dots1 = getdots(IM1)
    contours1 = connectdots(dots1)
    dots2 = getdots(IM2)
    contours2 = connectdots(dots2)

    for i in range(len(contours2)):
        contours2[i] = [(c[1],c[0]) for c in contours2[i]]
    contours = contours1+contours2

    for i in range(len(contours)):
        for j in range(len(contours)):
            if len(contours[i]) > 0 and len(contours[j])>0:
                if distsum(contours[j][0],contours[i][-1]) < 8:
                    contours[i] = contours[i]+contours[j]
                    contours[j] = []

    for i in range(len(contours)):
        contours[i] = [contours[i][j] for j in range(0,len(contours[i]),8)]

    contours = [c for c in contours if len(c) > 1]

    for i in range(0,len(contours)):
        contours[i] = [(v[0]*sc,v[1]*sc) for v in contours[i]]

    for i in range(0,len(contours)):
        for j in range(0,len(contours[i])):
            contours[i][j] = int(contours[i][j][0]+10*perlin_noise(i*0.5,j*0.1,1)),int(contours[i][j][1]+10*perlin_noise(i*0.5,j*0.1,2))

    return contours

def hatch(IM,sc=16):
    print("hatching...")
    PX = IM.load()
    w,h = IM.size
    lg1 = []
    lg2 = []
    for x0 in range(w):
        for y0 in range(h):
            x = x0*sc
            y = y0*sc
            if PX[x0,y0] > 144:
                pass

            elif PX[x0,y0] > 64:
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
            elif PX[x0,y0] > 16:
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
                lg2.append([(x+sc,y),(x,y+sc)])

            else:
                lg1.append([(x,y+sc/4),(x+sc,y+sc/4)])
                lg1.append([(x,y+sc/2+sc/4),(x+sc,y+sc/2+sc/4)])
                lg2.append([(x+sc,y),(x,y+sc)])

    lines = [lg1,lg2]
    for k in range(0,len(lines)):
        for i in range(0,len(lines[k])):
            for j in range(0,len(lines[k])):
                if lines[k][i] != [] and lines[k][j] != []:
                    if lines[k][i][-1] == lines[k][j][0]:
                        lines[k][i] = lines[k][i]+lines[k][j][1:]
                        lines[k][j] = []
        lines[k] = [l for l in lines[k] if len(l) > 0]
    lines = lines[0]+lines[1]

    for i in range(0,len(lines)):
        for j in range(0,len(lines[i])):
            lines[i][j] = int(lines[i][j][0]+sc*perlin_noise(i*0.5,j*0.1,1)),int(lines[i][j][1]+sc*perlin_noise(i*0.5,j*0.1,2))-j
    return lines

def makesvg(lines):
    print("generating svg file...")
    out = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1">'
    for l in lines:
        l = ",".join([str(p[0]*0.5)+","+str(p[1]*0.5) for p in l])
        out += '<polyline points="'+l+'" stroke="black" stroke-width="2" fill="none" />\n'
    out += '</svg>'
    return out

def image_to_svg_text_exact(path: str):
    IM = None
    possible = [path,"images/"+path,"images/"+path+".jpg","images/"+path+".png","images/"+path+".tif"]
    for p in possible:
        try:
            IM = Image.open(p)
            break
        except FileNotFoundError:
            pass
    if IM is None:
        raise FileNotFoundError("The Input File wasn't found. Check Path")

    w,h = IM.size
    IM = IM.convert("L")
    IM=ImageOps.autocontrast(IM,10)

    lines = []
    if draw_contours:
        lines += getcontours(IM.resize((resolution//contour_simplify,resolution//contour_simplify*h//w)),contour_simplify)
    if draw_hatch:
        lines += hatch(IM.resize((resolution//hatch_size,resolution//hatch_size*h//w)),hatch_size)

    lines = sortlines(lines)
    if show_bitmap:
        disp = Image.new("RGB",(resolution,resolution*h//w),(255,255,255))
        draw = ImageDraw.Draw(disp)
        for l in lines:
            draw.line(l,(0,0,0),5)
        disp.show()

    svg_text = makesvg(lines)
    print(len(lines),"strokes.")
    print("done.")
    return lines, svg_text

# -----------------------------
# Original stage-2 code
# -----------------------------
def sample_svg_paths(svg_path: Path, sample_step: float = 1.0) -> List[List[Point]]:
    paths, _attributes, _svg_attributes = svg2paths2(str(svg_path))
    polylines: List[List[Point]] = []

    for path in paths:
        if len(path) == 0:
            continue

        pts: List[Point] = []

        for segment in path:
            try:
                seg_len = max(float(segment.length(error=1e-4)), 0.001)
            except Exception:
                seg_len = 1.0

            num_samples = max(2, int(math.ceil(seg_len / sample_step)))

            for i in range(num_samples):
                t = i / (num_samples - 1)
                p = segment.point(t)
                pts.append((float(p.real), float(p.imag)))

        cleaned: List[Point] = []
        for p in pts:
            if not cleaned:
                cleaned.append(p)
            else:
                x1, y1 = cleaned[-1]
                x2, y2 = p
                if abs(x2 - x1) > 1e-9 or abs(y2 - y1) > 1e-9:
                    cleaned.append(p)

        if len(cleaned) >= 2:
            polylines.append(cleaned)

    return polylines

def get_bounds(polylines: List[List[Point]]) -> Tuple[float, float, float, float]:
    xs = [x for poly in polylines for x, _ in poly]
    ys = [y for poly in polylines for _, y in poly]
    return min(xs), min(ys), max(xs), max(ys)

def scale_polylines(
    polylines: List[List[Point]],
    width_mm: float,
    height_mm: float,
    margin_mm: float,
    keep_aspect: bool = True,
) -> List[List[Point]]:
    if not polylines:
        return []

    min_x, min_y, max_x, max_y = get_bounds(polylines)
    src_w = max_x - min_x
    src_h = max_y - min_y

    if src_w == 0 or src_h == 0:
        raise ValueError("SVG bounds are invalid.")

    usable_w = width_mm - 2 * margin_mm
    usable_h = height_mm - 2 * margin_mm

    if keep_aspect:
        scale = min(usable_w / src_w, usable_h / src_h)
        scaled_w = src_w * scale
        scaled_h = src_h * scale
        offset_x = margin_mm + (usable_w - scaled_w) / 2.0
        offset_y = margin_mm + (usable_h - scaled_h) / 2.0
    else:
        scale_x = usable_w / src_w
        scale_y = usable_h / src_h
        offset_x = margin_mm
        offset_y = margin_mm

    result: List[List[Point]] = []
    for poly in polylines:
        new_poly: List[Point] = []
        for x, y in poly:
            x0 = x - min_x
            y0 = y - min_y

            if keep_aspect:
                x_mm = offset_x + x0 * scale
                y_mm = offset_y + y0 * scale
            else:
                x_mm = offset_x + x0 * scale_x
                y_mm = offset_y + y0 * scale_y

            y_mm = height_mm - y_mm
            new_poly.append((x_mm, y_mm))
        result.append(new_poly)

    return result

def sort_polylines(polylines: List[List[Point]]) -> List[List[Point]]:
    if not polylines:
        return []

    remaining = [list(poly) for poly in polylines]
    ordered = [remaining.pop(0)]

    while remaining:
        last_x, last_y = ordered[-1][-1]

        best_idx = 0
        best_dist = float("inf")
        best_reverse = False

        for i, poly in enumerate(remaining):
            sx, sy = poly[0]
            ex, ey = poly[-1]

            d_start = (sx - last_x) ** 2 + (sy - last_y) ** 2
            d_end = (ex - last_x) ** 2 + (ey - last_y) ** 2

            if d_start < best_dist:
                best_dist = d_start
                best_idx = i
                best_reverse = False

            if d_end < best_dist:
                best_dist = d_end
                best_idx = i
                best_reverse = True

        chosen = remaining.pop(best_idx)
        if best_reverse:
            chosen.reverse()
        ordered.append(chosen)

    return ordered

def generate_gcode(
    polylines: List[List[Point]],
    draw_feed: int = 1500,
    travel_feed: int = 3000,
    pen_up_cmd: str = "M5",
    pen_down_cmd: str = "M3 S30",
    close_paths: bool = False,
) -> str:
    lines: List[str] = []

    lines.append("; Generated by svg_to_gcode.py")
    lines.append("G21 ; mm units")
    lines.append("G90 ; absolute positioning")
    lines.append(pen_up_cmd)
    lines.append(f"G0 F{travel_feed} X0 Y0")
    lines.append("")

    for poly in polylines:
        if len(poly) < 2:
            continue

        start_x, start_y = poly[0]
        lines.append(pen_up_cmd)
        lines.append(f"G0 F{travel_feed} X{start_x:.3f} Y{start_y:.3f}")
        lines.append(pen_down_cmd)

        for x, y in poly[1:]:
            lines.append(f"G1 F{draw_feed} X{x:.3f} Y{y:.3f}")

        if close_paths:
            first_x, first_y = poly[0]
            last_x, last_y = poly[-1]
            if abs(first_x - last_x) > 1e-9 or abs(first_y - last_y) > 1e-9:
                lines.append(f"G1 F{draw_feed} X{first_x:.3f} Y{first_y:.3f}")

        lines.append(pen_up_cmd)
        lines.append("")

    lines.append(pen_up_cmd)
    lines.append(f"G0 F{travel_feed} X0 Y0")
    lines.append("M2")

    return "\n".join(lines)

def save_preview(
    polylines: List[List[Point]],
    width_mm: float,
    height_mm: float,
    output_path: Path,
) -> None:
    scale = 4
    w = int(width_mm * scale)
    h = int(height_mm * scale)
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    for poly in polylines:
        pts = []
        for x_mm, y_mm in poly:
            x_px = int(round(x_mm * scale))
            y_px = int(round((height_mm - y_mm) * scale))
            pts.append([x_px, y_px])

        if len(pts) >= 2:
            arr = np.array([pts], dtype=np.int32)
            cv2.polylines(img, [arr], isClosed=False, color=(0, 0, 0), thickness=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)

# -----------------------------
# Combined CLI
# -----------------------------
def main() -> None:
    global draw_hatch, draw_contours, hatch_size, contour_simplify, show_bitmap, no_cv

    parser = argparse.ArgumentParser(description="Exact-logic combined pipeline without needing to keep the SVG file")
    parser.add_argument("input_image", type=str, help="Input image path or basename")
    parser.add_argument("--gcode", type=Path, required=True, help="Output G-code path")
    parser.add_argument("--preview", type=Path, default=None, help="Optional preview PNG path")
    parser.add_argument("--keep-svg", type=Path, default=None, help="Optional path to also save the SVG")

    # original stage-1 args
    parser.add_argument("-b","--show_bitmap", dest="show_bitmap",
        const=True, default=False, action="store_const",
        help="Display bitmap preview.")
    parser.add_argument("-nc","--no_contour", dest="no_contour",
        const=True, default=False, action="store_const",
        help="Don't draw contours.")
    parser.add_argument("-nh","--no_hatch", dest="no_hatch",
        const=True, default=False, action="store_const",
        help="Disable hatching.")
    parser.add_argument("--no_cv", dest="no_cv",
        const=True, default=False, action="store_const",
        help="Don't use openCV.")
    parser.add_argument("--hatch_size", dest="hatch_size",
        default=hatch_size, action="store", nargs="?", type=int,
        help="Patch size of hatches. eg. 8, 16, 32")
    parser.add_argument("--contour_simplify", dest="contour_simplify",
        default=contour_simplify, action="store", nargs="?", type=int,
        help="Level of contour simplification. eg. 1, 2, 3")

    # original stage-2 args
    parser.add_argument("--width", type=float, default=210.0, help="Canvas width in mm")
    parser.add_argument("--height", type=float, default=297.0, help="Canvas height in mm")
    parser.add_argument("--margin", type=float, default=10.0, help="Margin in mm")
    parser.add_argument("--sample-step", type=float, default=1.0, help="Curve sampling step in SVG units")
    parser.add_argument("--draw-feed", type=int, default=1500, help="Drawing feed rate")
    parser.add_argument("--travel-feed", type=int, default=3000, help="Travel feed rate")
    parser.add_argument("--pen-up", type=str, default="M5", help="Pen up command")
    parser.add_argument("--pen-down", type=str, default="M3 S30", help="Pen down command")
    parser.add_argument("--close-paths", action="store_true", help="Close each polyline back to start")
    parser.add_argument("--no-sort", action="store_true", help="Disable path sorting")
    parser.add_argument("--stretch", action="store_true", help="Stretch to fill area instead of keeping aspect ratio")

    args = parser.parse_args()

    draw_hatch = not args.no_hatch
    draw_contours = not args.no_contour
    hatch_size = args.hatch_size
    contour_simplify = args.contour_simplify
    show_bitmap = args.show_bitmap
    no_cv = args.no_cv

    print("=" * 68)
    print(" exact_logic_image_gcode_pipeline_no_saved_svg")
    print("=" * 68)

    _lines, svg_text = image_to_svg_text_exact(args.input_image)

    if args.keep_svg is not None:
        args.keep_svg.parent.mkdir(parents=True, exist_ok=True)
        args.keep_svg.write_text(svg_text, encoding="utf-8")
        print(f"Saved SVG to: {args.keep_svg}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".svg", delete=False, encoding="utf-8") as tmp:
        tmp.write(svg_text)
        tmp_svg_path = Path(tmp.name)

    try:
        polylines = sample_svg_paths(tmp_svg_path, sample_step=args.sample_step)
    finally:
        try:
            tmp_svg_path.unlink(missing_ok=True)
        except Exception:
            pass

    print(f"SVG polylines found: {len(polylines)}")
    if not polylines:
        raise ValueError("No drawable SVG paths found.")

    polylines = scale_polylines(
        polylines,
        width_mm=args.width,
        height_mm=args.height,
        margin_mm=args.margin,
        keep_aspect=not args.stretch,
    )

    if not args.no_sort:
        polylines = sort_polylines(polylines)

    gcode = generate_gcode(
        polylines,
        draw_feed=args.draw_feed,
        travel_feed=args.travel_feed,
        pen_up_cmd=args.pen_up,
        pen_down_cmd=args.pen_down,
        close_paths=args.close_paths,
    )

    args.gcode.parent.mkdir(parents=True, exist_ok=True)
    args.gcode.write_text(gcode, encoding="utf-8")
    print(f"Saved G-code to: {args.gcode}")

    if args.preview is not None:
        save_preview(polylines, args.width, args.height, args.preview)
        print(f"Saved preview to: {args.preview}")

if __name__ == "__main__":
    main()
