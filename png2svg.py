#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import math
from collections import deque
from io import StringIO
from optparse import OptionParser

from PIL import Image


logging.basicConfig()
log = logging.getLogger("png2svg")


def add_tuple(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub_tuple(a, b):
    return tuple(x - y for x, y in zip(a, b))


def direction(edge):
    return sub_tuple(edge[1], edge[0])


def magnitude(a):
    return int(math.sqrt(a[0] ** 2 + a[1] ** 2))


def normalize(a):
    mag = magnitude(a)
    assert mag > 0, "Cannot normalize a zero-length vector"
    return tuple(x / mag for x in a)


def svg_header(width, height):
    return """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="%d" height="%d"
     xmlns="http://www.w3.org/2000/svg" version="1.1">
""" % (width, height)


def rgba_image_to_svg_pixels(im, opaque=None):
    s = StringIO()
    s.write(svg_header(*im.size))

    width, height = im.size
    for x in range(width):
        for y in range(height):
            here = (x, y)
            rgba = im.getpixel(here)
            if opaque and not rgba[3]:
                continue
            s.write(
                '  <rect x="%d" y="%d" width="1" height="1" '
                'style="fill:rgb%s; fill-opacity:%.3f; stroke:none;" />\n'
                % (x, y, rgba[0:3], float(rgba[3]) / 255)
            )

    s.write("</svg>\n")
    return s.getvalue()


def joined_edges(assorted_edges, keep_every_point=False):
    pieces = []
    piece = []
    directions = deque([
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
    ])

    while assorted_edges:
        if not piece:
            piece.append(assorted_edges.pop())

        current_direction = normalize(direction(piece[-1]))
        while current_direction != directions[2]:
            directions.rotate()

        for i in range(1, 4):
            next_end = add_tuple(piece[-1][1], directions[i])
            next_edge = (piece[-1][1], next_end)
            if next_edge in assorted_edges:
                assorted_edges.remove(next_edge)
                if i == 2 and not keep_every_point:
                    piece[-1] = (piece[-1][0], next_edge[1])
                else:
                    piece.append(next_edge)

                if piece[0][0] == piece[-1][1]:
                    if (
                        not keep_every_point
                        and normalize(direction(piece[0]))
                        == normalize(direction(piece[-1]))
                    ):
                        piece[-1] = (piece[-1][0], piece.pop(0)[1])
                    pieces.append(piece)
                    piece = []
                break
        else:
            raise Exception("Failed to find connecting edge")

    return pieces


def rgba_image_to_svg_contiguous(im, opaque=None, keep_every_point=False):
    adjacent = ((1, 0), (0, 1), (-1, 0), (0, -1))
    visited = Image.new("1", im.size, 0)
    color_pixel_lists = {}

    width, height = im.size
    for x in range(width):
        for y in range(height):
            here = (x, y)
            if visited.getpixel(here):
                continue

            rgba = im.getpixel((x, y))
            if opaque and not rgba[3]:
                continue

            piece = []
            queue = [here]
            visited.putpixel(here, 1)

            while queue:
                here = queue.pop()
                for offset in adjacent:
                    neighbour = add_tuple(here, offset)
                    if not (0 <= neighbour[0] < width) or not (0 <= neighbour[1] < height):
                        continue
                    if visited.getpixel(neighbour):
                        continue
                    neighbour_rgba = im.getpixel(neighbour)
                    if neighbour_rgba != rgba:
                        continue
                    queue.append(neighbour)
                    visited.putpixel(neighbour, 1)
                piece.append(here)

            if rgba not in color_pixel_lists:
                color_pixel_lists[rgba] = []
            color_pixel_lists[rgba].append(piece)

    edges = {
        (-1, 0): ((0, 0), (0, 1)),
        (0, 1): ((0, 1), (1, 1)),
        (1, 0): ((1, 1), (1, 0)),
        (0, -1): ((1, 0), (0, 0)),
    }

    color_edge_lists = {}
    for rgba, pieces in color_pixel_lists.items():
        for piece_pixel_list in pieces:
            piece_pixel_set = set(piece_pixel_list)
            edge_set = set()
            for coord in piece_pixel_list:
                for offset, (start_offset, end_offset) in edges.items():
                    neighbour = add_tuple(coord, offset)
                    start = add_tuple(coord, start_offset)
                    end = add_tuple(coord, end_offset)
                    edge = (start, end)
                    if neighbour in piece_pixel_set:
                        continue
                    edge_set.add(edge)

            if rgba not in color_edge_lists:
                color_edge_lists[rgba] = []
            color_edge_lists[rgba].append(edge_set)

    color_joined_pieces = {}
    for color, pieces in color_edge_lists.items():
        color_joined_pieces[color] = []
        for assorted_edges in pieces:
            color_joined_pieces[color].append(
                joined_edges(set(assorted_edges), keep_every_point)
            )

    s = StringIO()
    s.write(svg_header(*im.size))

    for color, shapes in color_joined_pieces.items():
        for shape in shapes:
            s.write(' <path d="')
            for sub_shape in shape:
                sub_shape_copy = list(sub_shape)
                here = sub_shape_copy.pop(0)[0]
                s.write(" M %d,%d" % here)
                for edge in sub_shape_copy:
                    here = edge[0]
                    s.write(" L %d,%d" % here)
                s.write(" Z")
            s.write(
                '" style="fill:rgb%s; fill-opacity:%.3f; stroke:none;" />\n'
                % (color[0:3], float(color[3]) / 255)
            )

    s.write("</svg>\n")
    return s.getvalue()


def png_to_svg(filename, contiguous=None, opaque=None, keep_every_point=None):
    try:
        im = Image.open(filename)
    except IOError:
        sys.stderr.write("%s: Could not open as image file\n" % filename)
        sys.exit(1)

    im_rgba = im.convert("RGBA")

    if contiguous:
        return rgba_image_to_svg_contiguous(im_rgba, opaque, keep_every_point)
    return rgba_image_to_svg_pixels(im_rgba, opaque)


def default_output_path(input_path):
    base, _ = os.path.splitext(input_path)
    return base + ".svg"


def save_svg(input_path, output_path=None, contiguous=None, opaque=None, keep_every_point=None):
    svg_text = png_to_svg(
        input_path,
        contiguous=contiguous,
        opaque=opaque,
        keep_every_point=keep_every_point,
    )
    output_path = output_path or default_output_path(input_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg_text)
    return output_path


if __name__ == "__main__":
    parser = OptionParser(usage="%prog [options] FILE")
    parser.add_option(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbosity",
        help="Print verbose information for debugging",
        default=None,
    )
    parser.add_option(
        "-q",
        "--quiet",
        action="store_false",
        dest="verbosity",
        help="Suppress warnings",
        default=None,
    )
    parser.add_option(
        "-p",
        "--pixels",
        action="store_false",
        dest="contiguous",
        help="Generate a separate shape for each pixel; do not group pixels into contiguous areas of the same colour",
        default=True,
    )
    parser.add_option(
        "-o",
        "--opaque",
        action="store_true",
        dest="opaque",
        help="Opaque only; do not create shapes for fully transparent pixels.",
        default=None,
    )
    parser.add_option(
        "-1",
        "--one",
        action="store_true",
        dest="keep_every_point",
        help="1-pixel-width edges on contiguous shapes; default is to remove intermediate points on straight line edges.",
        default=None,
    )
    parser.add_option(
        "-f",
        "--output",
        dest="output_file",
        help="Output SVG file path. Defaults to the same name as the input with a .svg extension.",
        default=None,
    )

    (options, args) = parser.parse_args()

    if options.verbosity is True:
        log.setLevel(logging.DEBUG)
    elif options.verbosity is False:
        log.setLevel(logging.ERROR)

    if len(args) != 1:
        parser.error("exactly one input image file is required")

    output_path = save_svg(
        args[0],
        output_path=options.output_file,
        contiguous=options.contiguous,
        opaque=options.opaque,
        keep_every_point=options.keep_every_point,
    )
    print(f"Saved SVG to: {output_path}")
