# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data import generate_data, smoothen
from model import generate_model, train_model, test_model

import freetype as ft
import scipy.ndimage

from PIL import Image, ImageDraw, ImageFont
import sys

def str_to_img(text, size=11, zmin=1.0, zmax=1.0, add_kerning=False):
    """
    Generate a noisy bitmap string of text using different fonts

    Parameters
    ==========

    text: string
        Text to be displayed

    size: int
        Font size to use to generate text (default 20)

    zmin: float
        Minimal horizontal distortion

    zmax: float
        Maximal horizontal distortion

    Returns
    =======

    Tuple of numpy array (Z,I)

       Z is the bitmap string array

       I is a unidimensional numpy array that indicates the corresponding
       character for each column of Z
    """

    # Load fonts
    # fonts = ["../data/font/FreeMono.ttf", "../data/font/FreeMono_Italic.ttf", "../data/font/FreeMono_Bold.ttf", "../data/font/FreeMono_Bold_Italic.ttf"]
    fonts = ["../data/font/Inconsolata-Regular.ttf"]
    faces = [ft.Face(filename) for filename in fonts]
    for face in faces:
        face.set_char_size(size*64)
    slots = [face.glyph for face in faces]

    # Find baseline and height (maximum)
    baseline, height = 0, 0
    for face in faces:
        ascender = face.size.ascender >> 6
        descender = face.size.descender >> 6
        height = max(height, ascender-descender)
        baseline = max(baseline, -descender)


    # Set individual character font and zoom level
    font_index = np.random.randint(0, len(faces), len(text))
    zoom_level = np.random.uniform(zmin, zmax, len(text))

    # First pass to compute bounding box
    width = 0
    previous = 0
    for i,c in enumerate(text):
        index = font_index[i]
        zoom = zoom_level[i]
        face, slot = faces[index], slots[index]
        face.load_char(c, ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT)

        bitmap = slot.bitmap
        kerning = face.get_kerning(previous, c).x >> 6
        kerning = int(round(zoom*kerning))
        advance = slot.advance.x >> 6
        advance = int(round(zoom*advance))

        if i == len(text)-1:
            width += max(advance, int(round(zoom*bitmap.width)))
        else:
            width += advance + kerning
        previous = c



    # Allocate arrays for storing data
    Z = np.zeros((height,width), dtype=np.ubyte)
    I = np.zeros(width, dtype=np.int) + ord(' ')

    # Second pass for actual rendering
    x, y = 0, 0
    previous = 0
    for i,c in enumerate(text):
        index = font_index[i]
        zoom = zoom_level[i]
        face, slot = faces[index], slots[index]
        face.load_char(c, ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT)

        bitmap = slot.bitmap
        top, left = slot.bitmap_top, slot.bitmap_left
        w,h = bitmap.width, bitmap.rows
        y = height - baseline - top
        kerning = 0
        if(add_kerning):
            kerning = face.get_kerning(previous, c).x >> 6
            kerning = int(round(zoom*kerning))

        advance = slot.advance.x >> 6
        advance = int(round(zoom*advance))

        glyph = np.array(bitmap.buffer, dtype='ubyte').reshape(h,w)
        glyph = scipy.ndimage.zoom(glyph, (1, zoom), order=3)
        #print(np.max(np.float64(glyph) * 255 / np.max(glyph)))
        #print(glyph.shape)
        if glyph.shape[0]!= 0 and glyph.shape[1]!= 0:
            glyph = np.uint8(np.float64(glyph) * 255 / np.max(glyph))
        w = glyph.shape[1]

        x += kerning
        Z[y:y+h,x:x+w] += glyph
        I[x:x+w] = ord(c)
        x += advance
        previous = c

    return Z/255.0, I

def values_to_str(values, c=""):
    s = ""
    for v in values[:-1]:
        s+="{:.0f}{:s}".format(v, c)
    s+="{:.0f}".format(v)
    return s

def convert_data(data_, size):
    s = values_to_str(data_["input"][:, 0])
    Z, I = str_to_img(s, size = size)
    data = np.zeros(Z.shape[1], dtype = [ ("input",  float, (1 + Z.shape[0],)),
                                    ("output", float, (    n_gate,))])
    data["input"][:, :-1] = Z.T
    data["input"][:,-1] = np.repeat(data_["input"][:, 1], Z.shape[1]//len(s))
    data["output"][:, 0] = np.repeat(data_["output"], Z.shape[1]//len(s))/10
    return data

if __name__ == '__main__':
    
    # Random generator initialization
    np.random.seed(1)
    
    # Build memoryticks
    n_gate = 1
    size = 11

    # Training data
    n = 10000
    values = np.random.randint(0, 10, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.1
    train_data_ = generate_data(values, ticks)
    train_data = convert_data(train_data_, size)


    # Testing data
    n = 50
    values = np.random.randint(0, 10, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.1
    test_data_ = generate_data(values, ticks, last = train_data_["output"][-1])
    test_data = convert_data(test_data_, size)

    # Model
    model = generate_model(shape=(train_data["input"].shape[1],1000,n_gate), sparsity=0.5, radius=0.1,
                        scaling=0.25, leak=1.0, noise=0.0001)
    
    error = train_model(model, train_data)
    print("Training error : {0}".format(error))
    
    error = test_model(model, test_data)
    print("Testing error : {0}".format(error))


    # Display
    size_char = 1
    size_curves = 2
    gs = gridspec.GridSpec(2*size_curves+size_char, 1)
    fig = plt.figure(figsize=(10,5))
    fig.patch.set_alpha(0.0)

    data = test_data

    ax1 = fig.add_subplot(gs[:size_char])
    Z = test_data["input"][:, :].T
    RGBA = Z.reshape((Z.shape[0], Z.shape[1], 1))
    RGBA = np.repeat(RGBA, 4, axis=2)
    RGBA[:,:,:3] = 0
    ax1.get_yaxis().set_visible(False)
    ax1.imshow(RGBA, interpolation='nearest', origin='upper')

    ax2 = fig.add_subplot(gs[size_char:size_char+size_curves], sharex=ax1)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.plot(data["output"],  color='0.75', lw=1.0)
    ax2.plot(model["output"], color='0.00', lw=1.5)
    X, Y = np.arange(len(data)), np.ones(len(data))
    C = np.zeros((len(data),4))
    C[:,3] = data["input"][:,-1]
    ax2.scatter(X, 0.1*Y, s=1, facecolors=C, edgecolors=None)
    ax2.text(5, 0.1, "Triggers:",
             fontsize=8, transform=ax2.transData,
             horizontalalignment="right", verticalalignment="center")
    ax2.set_ylim(-0.1,1.1)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Input & Output")
    ax2.text(0.01, 0.9, "A",
             fontsize=16, fontweight="bold", transform=ax2.transAxes,
             horizontalalignment="left", verticalalignment="top")


    ax3 = fig.add_subplot(gs[size_char+size_curves:], sharex=ax1)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax3.plot(model["output"]-data["output"],  color='red', lw=1.0)
    ax3.set_ylim(-0.11, +0.11)
    ax3.yaxis.tick_right()
    ax3.axhline(0, color='.75', lw=.5)
    ax3.set_ylabel("Output error")
    ax3.text(0.01, 0.9, "B",
             fontsize=16, fontweight="bold", transform=ax3.transAxes,
             horizontalalignment="left", verticalalignment="top")


    plt.tight_layout()
    plt.savefig("../img/WM-one-gate_character.pdf")
    plt.show()
