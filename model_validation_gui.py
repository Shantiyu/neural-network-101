"""MNIST 手写验证 GUI。

用鼠标在 28×28 的画板上写一个数字，然后让训练好的模型预测。
画板每个像素显示为 5×5 个屏幕像素，所以整个画板是 140×140。

用法：
    python model_validation_gui.py
    python model_validation_gui.py --model models/mnist_conv_01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

from neural_network import network

GRID = 28
PIXEL_PX = 5
CANVAS_PX = GRID * PIXEL_PX
FONT = "Microsoft YaHei UI"


def center_by_mass(image: np.ndarray) -> np.ndarray:
    """按非零像素的质心做整数平移，让手写数字更接近 MNIST 的中心位置。"""
    ys, xs = np.nonzero(image)
    if len(ys) == 0:
        return image

    dy = int(round((GRID - 1) / 2 - ys.mean()))
    dx = int(round((GRID - 1) / 2 - xs.mean()))
    if dy == 0 and dx == 0:
        return image

    shifted = np.zeros_like(image)
    sy0, sy1 = max(0, -dy), min(GRID, GRID - dy)
    sx0, sx1 = max(0, -dx), min(GRID, GRID - dx)
    shifted[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = image[sy0:sy1, sx0:sx1]
    return shifted


class ValidationGUI:
    def __init__(self, root: tk.Tk, model_path: Path) -> None:
        self.root = root
        self.model_path = model_path
        self.net = network.deserialize(str(model_path))

        self.pixels = np.zeros((GRID, GRID), dtype=np.uint8)
        self.cells: dict[tuple[int, int], int] = {}
        self.drawing = False
        self.erasing = False
        self.last_cell: tuple[int, int] | None = None

        self.brush_size = tk.IntVar(value=2)
        self.auto_center = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar()

        self._build_ui()
        self._create_cells()
        self._reset_result()
        self.status_var.set(f"模型已加载：{model_path.name}  |  左键绘制，右键擦除")

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.root.title(f"MNIST 手写验证 - {self.model_path.name}")
        self.root.resizable(False, False)

        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="n")
        ttk.Label(left, text="28 × 28 画板（每格 5px）", font=(FONT, 10)).pack(pady=(0, 6))

        border = tk.Frame(left, bg="#888888", padx=1, pady=1)
        border.pack()
        self.canvas = tk.Canvas(
            border,
            width=CANVAS_PX,
            height=CANVAS_PX,
            bg="#000000",
            highlightthickness=0,
            bd=0,
            cursor="pencil",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_press)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)

        ttk.Label(
            left,
            text="左键绘制 · 右键擦除",
            foreground="#666666",
            font=(FONT, 9),
        ).pack(pady=(6, 0))

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="n", padx=(18, 0))

        ttk.Label(right, text="预测结果", font=(FONT, 10, "bold")).pack(anchor="w")
        self.digit_label = tk.Label(
            right,
            text="-",
            font=(FONT, 54, "bold"),
            fg="#2c3e50",
            width=3,
        )
        self.digit_label.pack(anchor="w")
        self.conf_label = ttk.Label(right, text="置信度 --", font=(FONT, 11))
        self.conf_label.pack(anchor="w")
        self.top3_label = ttk.Label(right, text="", foreground="#666666", font=(FONT, 9))
        self.top3_label.pack(anchor="w", pady=(2, 8))

        self.bar_canvas = tk.Canvas(
            right,
            width=270,
            height=176,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#dddddd",
        )
        self.bar_canvas.pack()
        self._build_bars()

        controls = ttk.Frame(right)
        controls.pack(fill="x", pady=(10, 0))

        ttk.Label(controls, text="笔刷").pack(side="left")
        spin = ttk.Spinbox(
            controls,
            from_=1,
            to=5,
            width=3,
            textvariable=self.brush_size,
        )
        spin.pack(side="left", padx=(4, 12))

        ttk.Checkbutton(
            controls,
            text="自动居中",
            variable=self.auto_center,
        ).pack(side="left")

        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="验证 (Enter)", command=self.predict).pack(side="left")
        ttk.Button(buttons, text="清空 (C)", command=self.clear).pack(side="left", padx=(8, 0))

        self.status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            relief="sunken",
            padding=(8, 3),
        )
        self.status.grid(row=1, column=0, sticky="ew")

        self.root.bind("<Return>", lambda _event: self.predict())
        self.root.bind("<KeyPress-c>", lambda _event: self.clear())
        self.root.bind("<KeyPress-C>", lambda _event: self.clear())

    def _create_cells(self) -> None:
        for y in range(GRID):
            for x in range(GRID):
                rect = self.canvas.create_rectangle(
                    x * PIXEL_PX,
                    y * PIXEL_PX,
                    (x + 1) * PIXEL_PX,
                    (y + 1) * PIXEL_PX,
                    fill="#000000",
                    outline="#000000",
                )
                self.cells[(x, y)] = rect

    # ------------------------------------------------------------- drawing
    def _event_cell(self, event: tk.Event) -> tuple[int, int]:
        x = min(max(int(event.x) // PIXEL_PX, 0), GRID - 1)
        y = min(max(int(event.y) // PIXEL_PX, 0), GRID - 1)
        return x, y

    def _set_pixel(self, x: int, y: int, on: bool) -> None:
        if not (0 <= x < GRID and 0 <= y < GRID):
            return
        value = 255 if on else 0
        if self.pixels[y, x] == value:
            return
        self.pixels[y, x] = value
        self.canvas.itemconfigure(self.cells[(x, y)], fill="#ffffff" if on else "#000000")

    def _stamp(self, cx: int, cy: int, erase: bool) -> None:
        size = max(1, int(self.brush_size.get()))
        start = -(size // 2)
        for oy in range(start, start + size):
            for ox in range(start, start + size):
                self._set_pixel(cx + ox, cy + oy, not erase)

    def _draw_line(self, x0: int, y0: int, x1: int, y1: int, erase: bool) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0:
            self._stamp(x1, y1, erase)
            return
        for i in range(steps + 1):
            x = int(round(x0 + (x1 - x0) * i / steps))
            y = int(round(y0 + (y1 - y0) * i / steps))
            self._stamp(x, y, erase)

    def _on_left_press(self, event: tk.Event) -> None:
        self.drawing = True
        self.erasing = False
        self.last_cell = self._event_cell(event)
        self._stamp(*self.last_cell, erase=False)
        self.status_var.set("绘制中…… 画完后点击“验证”")

    def _on_left_drag(self, event: tk.Event) -> None:
        if not self.drawing or self.erasing:
            return
        cell = self._event_cell(event)
        if self.last_cell is not None:
            self._draw_line(*self.last_cell, *cell, erase=False)
        else:
            self._stamp(*cell, erase=False)
        self.last_cell = cell
        self.status_var.set("绘制中…… 画完后点击“验证”")

    def _on_right_press(self, event: tk.Event) -> None:
        self.drawing = True
        self.erasing = True
        self.last_cell = self._event_cell(event)
        self._stamp(*self.last_cell, erase=True)
        self.status_var.set("擦除中……")

    def _on_right_drag(self, event: tk.Event) -> None:
        if not self.drawing or not self.erasing:
            return
        cell = self._event_cell(event)
        if self.last_cell is not None:
            self._draw_line(*self.last_cell, *cell, erase=True)
        else:
            self._stamp(*cell, erase=True)
        self.last_cell = cell
        self.status_var.set("擦除中……")

    def _on_release(self, _event: tk.Event) -> None:
        self.drawing = False
        self.erasing = False
        self.last_cell = None

    # ------------------------------------------------------------ predict
    def _preprocess(self) -> np.ndarray:
        image = self.pixels.astype(np.float32) / 255.0
        if self.auto_center.get():
            image = center_by_mass(image)
        return image.reshape(1, GRID * GRID)

    def predict(self) -> None:
        if not np.any(self.pixels):
            self._reset_result()
            self.status_var.set("画板是空的，先写一个数字。")
            return

        probabilities = self.net.forward(self._preprocess())[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction])
        top3 = np.argsort(probabilities)[::-1][:3]

        self.digit_label.configure(text=str(prediction))
        self.conf_label.configure(text=f"置信度 {confidence * 100:.1f}%")
        self.top3_label.configure(
            text="Top-3：" + "   ".join(
                f"{int(d)} {probabilities[d] * 100:.1f}%" for d in top3
            )
        )
        self._update_bars(probabilities, prediction)
        self.status_var.set(f"模型认为你写的是 {prediction}，置信度 {confidence * 100:.1f}%")

    def clear(self) -> None:
        self.pixels.fill(0)
        for (x, y), rect in self.cells.items():
            self.canvas.itemconfigure(rect, fill="#000000")
        self._reset_result()
        self.status_var.set("画板已清空。")

    # ------------------------------------------------------------- result
    def _reset_result(self) -> None:
        self.digit_label.configure(text="-")
        self.conf_label.configure(text="置信度 --")
        self.top3_label.configure(text="")
        for digit, (bar, pct) in enumerate(zip(self._bar_items, self._pct_items)):
            y = 14 + digit * 16
            self.bar_canvas.coords(bar, 30, y - 6, 30, y + 6)
            self.bar_canvas.itemconfigure(bar, fill="#5b8ff9")
            self.bar_canvas.itemconfigure(pct, text="")

    def _build_bars(self) -> None:
        self._bar_items = []
        self._pct_items = []
        for digit in range(10):
            y = 14 + digit * 16
            self.bar_canvas.create_text(
                8, y, text=str(digit), anchor="w", font=(FONT, 7)
            )
            self.bar_canvas.create_rectangle(
                30, y - 6, 220, y + 6, fill="#f0f0f0", outline=""
            )
            bar = self.bar_canvas.create_rectangle(
                30, y - 6, 30, y + 6, fill="#5b8ff9", outline=""
            )
            pct = self.bar_canvas.create_text(
                226, y, text="", anchor="w", font=(FONT, 7), fill="#555555"
            )
            self._bar_items.append(bar)
            self._pct_items.append(pct)

    def _update_bars(self, probabilities: np.ndarray, prediction: int) -> None:
        for digit in range(10):
            y = 14 + digit * 16
            prob = float(probabilities[digit])
            width = 190 * prob
            bar = self._bar_items[digit]
            self.bar_canvas.coords(bar, 30, y - 6, 30 + width, y + 6)
            self.bar_canvas.itemconfigure(
                bar, fill="#e74c3c" if digit == prediction else "#5b8ff9"
            )
            self.bar_canvas.itemconfigure(self._pct_items[digit], text=f"{prob * 100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a digit and validate it with a trained MNIST model.")
    parser.add_argument("--model", default="models/mnist_conv_01", help="path to the pickled model")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"找不到模型文件：{model_path}")

    root = tk.Tk()

    try:
        app = ValidationGUI(root, model_path)
    except Exception as exc:  # 模型加载或界面构建失败时给出可读提示
        root.withdraw()
        messagebox.showerror("启动失败", f"无法加载模型或创建界面：\n{exc}")
        root.destroy()
        raise SystemExit(1)

    app._reset_result()

    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() - width) // 2
    y = (root.winfo_screenheight() - height) // 2
    root.geometry(f"+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    main()
