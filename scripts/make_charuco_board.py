#!/usr/bin/env python3
"""Print-ready ChArUco board at exact physical scale.

  python scripts/make_charuco_board.py --board configs/charuco_board.yaml --paper A3 --out board_A3.pdf

Print at 100 % / "actual size" (no "fit to page"), glue to a flat rigid plate (foam board,
acrylic, glass), then measure the squares and write the measured size into the YAML.
The page carries a 100 mm scale bar to check the printer scaling.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from metricvision.calibration.charuco import BoardSpec  # noqa: E402

PAPER_MM = {"A4": (297, 210), "A3": (420, 297), "A2": (594, 420), "letter": (279.4, 215.9), "tabloid": (431.8, 279.4)}  # landscape (w, h)
MM_PER_IN = 25.4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default="configs/charuco_board.yaml")
    ap.add_argument("--paper", default="A3", choices=list(PAPER_MM))
    ap.add_argument("--out", default="charuco_board.pdf")
    ap.add_argument("--fx", type=float, default=430.0, help="approx focal length (px) for the range table")
    a = ap.parse_args()
    spec = BoardSpec.from_yaml(a.board)
    bw, bh = (1000 * x for x in spec.size_m)
    pw, ph = PAPER_MM[a.paper]
    margin = 12.0
    if bw + 2 * margin > pw or bh + 2 * margin + 18 > ph:
        sys.exit(f"board {bw:.0f}x{bh:.0f} mm does not fit {a.paper} ({pw}x{ph} mm) with margins")

    img = spec.render(px_per_square=200, margin_px=0, legacy=spec.legacy == "true")
    fig = plt.figure(figsize=(pw / MM_PER_IN, ph / MM_PER_IN))
    x0, y0 = (pw - bw) / 2, (ph - bh) / 2 + 6
    ax = fig.add_axes([x0 / pw, y0 / ph, bw / pw, bh / ph])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="auto")
    ax.set_axis_off()
    # 100 mm scale bar + text
    sb = fig.add_axes([0, 0, 1, 1], facecolor="none")
    sb.set_xlim(0, pw)
    sb.set_ylim(0, ph)
    sb.set_axis_off()
    sb.plot([x0, x0 + 100], [y0 - 8, y0 - 8], color="black", lw=1.2)
    for xx in (x0, x0 + 100):
        sb.plot([xx, xx], [y0 - 10, y0 - 6], color="black", lw=1.2)
    sb.text(x0 + 102, y0 - 8, "100 mm", va="center", fontsize=7)
    sb.text(
        x0,
        y0 + bh + 3,
        f"ChArUco {spec.squares_x}x{spec.squares_y}, square {1000 * spec.square_length:.1f} mm, marker "
        f"{1000 * spec.marker_length:.1f} mm, {spec.dictionary}, legacy={spec.legacy}. Print at 100 % - "
        f"measure {spec.squares_x - 1} squares = {1000 * spec.square_length * (spec.squares_x - 1):.1f} mm",
        fontsize=6.5,
        va="bottom",
    )
    fig.savefig(a.out)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=150)
    print(f"wrote {a.out} ({a.paper}, board {bw:.0f} x {bh:.0f} mm)")
    bits = int(spec.dictionary.split("_")[1].split("X")[0]) + 2  # marker cells incl. black border
    print("\nworking-distance guide (one marker cell in pixels; reliable detection needs >= ~2 px):")
    for d in (0.3, 0.5, 0.75, 1.0, 1.25, 1.5):
        px = a.fx * spec.marker_length / d / bits
        print(f"  {d:4.2f} m : {px:4.1f} px/cell  {'ok' if px >= 2.0 else 'too small'}")


if __name__ == "__main__":
    main()
