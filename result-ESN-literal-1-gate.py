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
import sys

def str_to_bmp(text, size=11, zmin=1.0, zmax=1.0, add_kerning=False):
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
    fonts = ["./Inconsolata-Regular.ttf"]
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
        w = glyph.shape[1]
        x += kerning
        # left = 0
        Z[y:y+h,x+left:x+left+w] += glyph
        I[x:x+w] = ord(c)
        x += advance
        previous = c

    return Z/255.0, I


def convert_data(data_, size, noise = 0.):
    values = (data_["input"][:, 0]).astype(int)
    text = [chr(ord("0")+i) for i in values]
    Z, I = str_to_bmp(text, size = size)
    Z = Z [3:-3]
    
    # Z *= np.random.uniform(0.9,1.1,Z.shape)
    # Z = np.maximum(np.minimum(Z,1),0)
    
    data = np.zeros(Z.shape[1], dtype = [ ("input",  float, (1 + Z.shape[0],)),
                                          ("output", float, (    n_gate,))])
    data["input"][:, :-1] = Z.T + noise*np.random.uniform(-1,1, size = Z.T.shape)
    n = Z.shape[1]//len(text)
    data["input"][:,-1] = np.repeat(data_["input"][:, 1], n)
    data["output"][:, 0] = np.repeat(data_["output"], n) / 10
    return data




if __name__ == '__main__':
    
    # Random generator initialization
    np.random.seed(3)
    
    # Build memoryticks
    n_gate = 1
    size = 11

    # Training data
    n = 25000
    values = np.random.randint(0, 10, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.1
    train_data_ = generate_data(values, ticks)
    train_data = convert_data(train_data_, size, noise = 0.)


    # Testing data
    n = 50
    values = np.random.randint(0, 10, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.1
    test_data_ = generate_data(values, ticks, last = train_data_["output"][-1])
    test_data = convert_data(test_data_, size, noise = 0.)

    # Model
    model = generate_model(shape=(train_data["input"].shape[1],1000,n_gate),
                           sparsity=0.5, radius=0.1,
                           scaling=0.25, leak=1.0, noise= 0.0001)
    
    error = train_model(model, train_data)
    print("Training error : {0}".format(error))
    
    error = test_model(model, test_data)
    print("Testing error : {0}".format(error))


    # Display
    fig = plt.figure(figsize=(10,4))
    fig.patch.set_alpha(0.0)

    data = test_data

    ax1 = plt.subplot(2,1,1)

    Z = test_data["input"][:, :-1].T
    ax1.imshow(Z, interpolation='nearest', origin='upper', cmap="gray_r",
               extent=[0,len(data),1.225,1.4], aspect='auto')

    
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.plot(data["output"],  color='0.75', lw=1.0)
    ax1.plot(model["output"], color='0.00', lw=1.5)
    X, Y = np.arange(len(data)), np.ones(len(data))
    C = np.zeros((len(data),4))
    C[:,3] = data["input"][:,-1]

    ax1.scatter(X, 1.1*Y, s=1, facecolors=C, edgecolors=None)
    ax1.text(5, 1.1, "Triggers:",
             fontsize=8, transform=ax1.transData,
             horizontalalignment="right", verticalalignment="center")

    ax1.yaxis.tick_right()
    ax1.set_ylabel("Input & Output")
    ax1.text(0.01, 0.95, "A",
             fontsize=16, fontweight="bold", transform=ax1.transAxes,
             horizontalalignment="left", verticalalignment="top")

    ax1.set_ylim(-0.1,1.5)
    ax1.set_xlim(-15, 315)
    ax1.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    
    ax2 = plt.subplot(2,1,2)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.plot(model["output"]-data["output"],  color='red', lw=1.0, zorder=10)
    ax2.set_ylim(-0.11, +0.11)
    ax2.yaxis.tick_right()
    ax2.axhline(0, color='.75', lw=.5)
    ax2.set_ylabel("Output error")
    ax2.text(0.01, 0.95, "B",
             fontsize=16, fontweight="bold", transform=ax2.transAxes,
             horizontalalignment="left", verticalalignment="top")


    plt.tight_layout()
    plt.savefig("result-ESN-literal-1-gate.pdf")
    plt.show()
