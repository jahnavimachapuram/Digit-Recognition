"""
Small Tkinter scratch-pad that lets you draw a digit with mouse and
immediately see the CNN’s prediction.
Run:  python draw_digit_gui.py
"""

import tkinter as tk
from PIL import Image, ImageDraw
from utils import predict_digit

CANVAS_SIZE = 280     # pixels (10× MNIST resolution)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Digit Recogniser")
        self.resizable(0,0)

        self.canvas = tk.Canvas(self, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack()

        self.label = tk.Label(self, text="Draw a digit (0-9)", font=("Helvetica", 16))
        self.label.pack(pady=10)

        btn_frame = tk.Frame(self)
        btn_frame.pack()
        tk.Button(btn_frame, text="Predict", command=self.on_predict).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Clear",   command=self.clear_canvas).grid(row=0, column=1, padx=5)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.img)

    def paint(self, event):
        x, y = event.x, event.y
        r = 8                                    # brush radius
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def on_predict(self):
        digit = predict_digit(self.img)
        self.label.config(text=f"↳ I think it’s a **{digit}**")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,CANVAS_SIZE,CANVAS_SIZE], fill=255)
        self.label.config(text="Draw a digit (0-9)")

if __name__ == "__main__":
    App().mainloop()
