import cv2
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def clahe_demo_gui(im):
    if im is None:
        im = cv2.imread(cv2.samples.findFile("pc.jpg"))
    elif isinstance(im, str):
        im = cv2.imread(im)
    else:
        im = np.array(im)

    h = buildGUI(im)
    return h


def onChange(h):
    alg = h['pop'].get()
    clipLimit = h['slid'][0].get()
    tileSize = int(h['slid'][1].get())
    h['txt'][0].config(text=f'Clip Limit: {clipLimit:.2f}')
    h['txt'][1].config(text=f'Tile Size: {tileSize}x{tileSize}')

    opts = {'ClipLimit': clipLimit, 'TileGridSize': (tileSize, tileSize)}
    if alg == 2:
        out = h['src']
    elif h['src'].ndim == 2:
        if alg == 0:
            out = cv2.createCLAHE(clipLimit, (tileSize, tileSize)).apply(h['src'])
        elif alg == 1:
            out = cv2.equalizeHist(h['src'])
    else:
        lab = cv2.cvtColor(h['src'], cv2.COLOR_BGR2Lab)
        if alg == 0:
            lab[:, :, 0] = cv2.createCLAHE(clipLimit, (tileSize, tileSize)).apply(lab[:, :, 0])
        elif alg == 1:
            lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    h['img'].set_data(out)
    h['canvas'].draw()


def buildGUI(img):
    clipLimit = 2.0
    tileSize = 8
    sz = img.shape
    sz = (sz[1], max(sz[0], 300))

    h = {}
    h['src'] = img
    h['fig'] = Figure(figsize=(sz[0] / 100, sz[1] / 100))
    h['ax'] = h['fig'].add_axes([0, 0.09, 1, 0.91])
    h['canvas'] = FigureCanvasTkAgg(h['fig'], master=tk.Tk())
    h['canvas'].get_tk_widget().pack()

    h['img'] = h['ax'].imshow(img)
    h['txt'] = []
    h['slid'] = []

    h['txt'].append(tk.Label(master=tk.Tk(), text=f'Clip Limit: {clipLimit}', font=('Arial', 11)))
    h['txt'].append(tk.Label(master=tk.Tk(), text=f'Tile Size: {tileSize}x{tileSize}', font=('Arial', 11)))
    h['txt'].append(tk.Label(master=tk.Tk(), text='Algorithm:', font=('Arial', 11)))

    h['txt'][0].pack()
    h['txt'][1].pack()
    h['txt'][2].pack()

    h['slid'].append(tk.Scale(master=tk.Tk(), label='Clip Limit', from_=0, to=40, resolution=0.1, orient='horizontal'))
    h['slid'].append(tk.Scale(master=tk.Tk(), label='Tile Size', from_=1, to=32, orient='horizontal'))
    h['slid'][0].set(clipLimit)
    h['slid'][1].set(tileSize)

    h['slid'][0].pack()
    h['slid'][1].pack()

    h['pop'] = tk.StringVar(master=tk.Tk())
    h['pop'].set('CLAHE')
    h['menu'] = tk.OptionMenu(tk.Tk(), h['pop'], 'CLAHE', 'equalizeHist', '-None-')
    h['menu'].pack()

    h['slid'][0].config(command=lambda event: onChange(h))
    h['slid'][1].config(command=lambda event: onChange(h))
    h['pop'].trace('w', lambda *args: onChange(h))

    onChange(h)
    tk.mainloop()
    return h

# Example usage
clahe_demo_gui(None)
