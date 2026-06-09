import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector


LAYOUT_ITEMS = [
    ("scene", "Simulation / top-down view"),
    ("camera", "Robot camera"),
    ("map", "Lidar map"),
    ("log", "LLM status and result log"),
]


class LayoutSelector:
    def __init__(self):
        self.index = 0
        self.rectangles = {}
        self.patches = {}
        self.output_path = Path(__file__).with_name("dashboard_layout.json")

        self.fig, self.ax = plt.subplots(figsize=(12, 7.2))
        self.fig.canvas.manager.set_window_title("Dashboard Layout Selector")
        self.ax.set_title(self._title())
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_aspect("auto")
        self.ax.grid(True, linestyle="--", alpha=0.35)
        self.ax.set_xlabel("normalized x")
        self.ax.set_ylabel("normalized y")

        self.selector = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],
            minspanx=0.02,
            minspany=0.02,
            spancoords="data",
            interactive=True,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw_help()

    def _title(self):
        key, label = LAYOUT_ITEMS[self.index]
        return f"Draw region {self.index + 1}/{len(LAYOUT_ITEMS)}: {label} ({key})"

    def _draw_help(self):
        help_text = (
            "Drag left mouse button to draw current region.\n"
            "Keys: 1/2/3/4 switch target, Enter save, Backspace delete current, r reset, q quit."
        )
        self.ax.text(
            0.01,
            0.01,
            help_text,
            transform=self.ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )

    def _current_key(self):
        return LAYOUT_ITEMS[self.index][0]

    def _on_select(self, eclick, erelease):
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        x0, x1 = max(0.0, x0), min(1.0, x1)
        y0, y1 = max(0.0, y0), min(1.0, y1)

        key = self._current_key()
        self.rectangles[key] = {
            "left": round(float(x0), 4),
            "bottom": round(float(y0), 4),
            "width": round(float(x1 - x0), 4),
            "height": round(float(y1 - y0), 4),
        }
        self._redraw_rectangles()

        if self.index < len(LAYOUT_ITEMS) - 1:
            self.index += 1
            self.ax.set_title(self._title())
            self.fig.canvas.draw_idle()

    def _redraw_rectangles(self):
        for patch in self.patches.values():
            patch.remove()
        self.patches.clear()

        colors = {
            "scene": "tab:blue",
            "camera": "tab:green",
            "map": "tab:orange",
            "log": "tab:purple",
        }
        for key, rect in self.rectangles.items():
            patch = Rectangle(
                (rect["left"], rect["bottom"]),
                rect["width"],
                rect["height"],
                fill=False,
                linewidth=2.0,
                edgecolor=colors.get(key, "tab:red"),
            )
            self.ax.add_patch(patch)
            self.ax.text(
                rect["left"] + 0.01,
                rect["bottom"] + rect["height"] - 0.03,
                key,
                color=colors.get(key, "tab:red"),
                fontsize=11,
                fontweight="bold",
                va="top",
            )
            self.patches[key] = patch
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in {"1", "2", "3", "4"}:
            self.index = int(event.key) - 1
            self.ax.set_title(self._title())
            self.fig.canvas.draw_idle()
        elif event.key in {"enter", "return"}:
            self._save()
        elif event.key == "backspace":
            self.rectangles.pop(self._current_key(), None)
            self._redraw_rectangles()
        elif event.key == "r":
            self.rectangles.clear()
            self._redraw_rectangles()
        elif event.key == "q":
            plt.close(self.fig)

    def _save(self):
        missing = [key for key, _ in LAYOUT_ITEMS if key not in self.rectangles]
        if missing:
            print(f"Missing regions: {missing}")
            return

        layout = {
            "figure_size": [12.0, 7.2],
            "axes": self.rectangles,
        }
        self.output_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")
        print(json.dumps(layout, indent=2))
        print(f"Saved layout to: {self.output_path}")
        plt.close(self.fig)

    def show(self):
        plt.show()


def main():
    LayoutSelector().show()


if __name__ == "__main__":
    main()
