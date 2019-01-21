# -----------------------------------------------------------------------------
#  Copyright (c) 2019 Nicolas P. Rougier
#  Distributed under the terms of the new BSD license.
# -----------------------------------------------------------------------------
import numpy as np
import freetype as ft
import scipy.ndimage


def generate_data(text, size=20, zmin=1.0, zmax=1.0, kerning=True):
    """
    Generate a noisy bitmap string of text using different fonts

    Parameters
    ==========

    text: string
        Text to be displayed

    size: int
        Font size to use to generate text (default 20)

    kerning: bool
        Whether to use kerning

    zmin: float
        Minimal horizontal zoom (shrink)

    zmax: float
        Maximal horizontal zoom (expansion)

    Returns
    =======

    Tuple of numpy array (Z,I)

       Z is the bitmap string array

       I is a unidimensional numpy array that indicates the corresponding
       character for each column of Z
    """

    # Load fonts
    fonts = [
        'Inconsolata-Regular.ttf'
        # 'data/FreeMono.ttf',
        # 'data/FreeMonoBold.ttf',
        # 'data/FreeMonoOblique.ttf',
        # 'data/FreeMonoBoldOblique.ttf'
    ]
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
    use_kerning = kerning
    
    for i,c in enumerate(text):
        index = font_index[i]
        zoom = zoom_level[i]
        face, slot = faces[index], slots[index]
        face.load_char(c, ft.FT_LOAD_RENDER | ft.FT_LOAD_FORCE_AUTOHINT)
        
        bitmap = slot.bitmap
        if use_kerning:
            kerning = face.get_kerning(previous, c).x >> 6
            kerning = int(round(zoom*kerning))
        else:
            kerning = 0
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

        if use_kerning:
            kerning = face.get_kerning(previous, c).x >> 6
            kerning = int(round(zoom*kerning))
        else:
            kerning = 0
        advance = slot.advance.x >> 6
        advance = int(round(zoom*advance))

        glyph = np.array(bitmap.buffer, dtype='ubyte').reshape(h,w)
        glyph = scipy.ndimage.zoom(glyph, (1, zoom), order=3)
        w = glyph.shape[1]
        
        x += kerning
        Z[y:y+h,x+left:x+left+w] += glyph
        I[x:x+advance] = ord(c)
        x += advance
        previous = c

    return Z/255.0, I
        


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import warnings
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['toolbar'] = 'None'
    
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    # Initialize the pseudo-random number generator
    np.random.seed(1)
    
    text = "Hello world!"
    text = [chr(ord("0") +i) for i in np.random.randint(0,10,50)]
    Z, I = generate_data(text, 11, zmin=1.0, zmax=1.0, kerning=False)
    
    # Mean character size
    print("Mean character size {0:.2f}x{1:.2f}".format(
                      Z.shape[1] / len(text), Z.shape[0]))
    
    # Display
    filename = "output.png"
    RGBA = Z.reshape((Z.shape[0], Z.shape[1], 1))
    RGBA = np.repeat(RGBA, 4, axis=2)
    RGBA[:,:,:3] = 0

    aspect = Z.shape[0]/Z.shape[1]
    size = 0.75
    fig = plt.figure(figsize=(size/aspect, size))
    ax = fig.add_axes([0,0,1,1], frameon=False)
    ax.imshow(RGBA, interpolation='nearest', origin='upper')
    plt.savefig(filename, transparent=True)

    # OSX/iTerm specific
    from subprocess import call
    call(["imgcat", filename])

    # System agnostic
    plt.savefig("textual-1-gate-task.pdf")
    # plt.show()
