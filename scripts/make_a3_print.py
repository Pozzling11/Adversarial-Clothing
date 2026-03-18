"""
Generate A3 300 DPI print-ready files for specified iterations.
Patch is upscaled to 150mm x 150mm and centred on a white A3 canvas.
"""
from PIL import Image
from pathlib import Path

# A3 at 300 DPI: 297mm x 420mm
A3_W = int(297 / 25.4 * 300)   # 3508 px
A3_H = int(420 / 25.4 * 300)   # 4961 px

# Fill the page: scale patch to full A3 width (297mm), centred vertically
PRINT_PX = A3_W   # 3507 px = 297mm — fills the full width

ROOT = Path(__file__).parent.parent

iterations = [
    ROOT / "patterns/iterations/iteration_15/patch_160_uniform.png",
    ROOT / "patterns/iterations/iteration_17/patch_160_uniform.png",
]

for src in iterations:
    patch = Image.open(src).convert("RGB")
    patch_print = patch.resize((PRINT_PX, PRINT_PX), Image.NEAREST)

    canvas = Image.new("RGB", (A3_W, A3_H), (255, 255, 255))
    x = (A3_W - PRINT_PX) // 2
    y = (A3_H - PRINT_PX) // 2
    canvas.paste(patch_print, (x, y))

    out = src.parent / "patch_A3_print_300dpi.png"
    canvas.save(out, dpi=(300, 300))
    print(f"Saved {out.relative_to(ROOT)}  "
          f"({PRINT_PX}x{PRINT_PX}px patch on {A3_W}x{A3_H}px canvas)")
