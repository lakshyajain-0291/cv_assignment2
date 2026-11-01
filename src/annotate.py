"""Polygon mask annotator for the captured photos.

Runs on the already-resized 256x256 PNGs, as Section 1c requires the mask to be
drawn on the resized photo. Displayed at 3x so edges can be placed accurately;
clicks are mapped back to 256x256 coordinates. Writes masks with the object at
255 and the background at 0.

Controls
  left click    add a polygon vertex
  u / backspace undo last vertex
  c             close the current polygon and add it to the mask
  e             toggle erase mode (next closed polygon is subtracted)
  r             reset this image
  n             save and go to next image
  p             previous image
  q             save and quit
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Qt ships no fonts in the OpenCV wheel; the window itself is fine and all text
# is drawn by cv2.putText (Hershey), so silence the cosmetic warnings.
os.environ.setdefault("QT_LOGGING_RULES", "*=false")
import numpy as np
import cv2

ZOOM = 3
IMG_DIR = "captured/images_256"
MSK_DIR = "captured/masks_256"


def run():
    os.makedirs(MSK_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    if not files:
        print(f"no images in {IMG_DIR}"); return
    i = 0
    cv2.namedWindow("annotate", cv2.WINDOW_AUTOSIZE)
    state = {"pts": [], "erase": False}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append((x // ZOOM, y // ZOOM))
    cv2.setMouseCallback("annotate", on_mouse)

    while 0 <= i < len(files):
        path = files[i]
        base = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        mpath = os.path.join(MSK_DIR, base)
        mask = (cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                if os.path.exists(mpath) else np.zeros((256, 256), np.uint8))
        if mask is None:
            mask = np.zeros((256, 256), np.uint8)
        state["pts"] = []

        while True:
            vis = img.copy()
            tint = np.zeros_like(vis); tint[:, :, 2] = 255
            m3 = (mask > 127)[:, :, None]
            vis = np.where(m3, (0.55 * vis + 0.45 * tint).astype(np.uint8), vis)
            big = cv2.resize(vis, (256 * ZOOM, 256 * ZOOM),
                             interpolation=cv2.INTER_NEAREST)
            pts = state["pts"]
            if pts:
                p = (np.array(pts) * ZOOM).astype(np.int32)
                cv2.polylines(big, [p], False, (0, 255, 255), 2)
                for q in p:
                    cv2.circle(big, tuple(q), 4, (0, 255, 0), -1)
            hdr = (f"[{i+1}/{len(files)}] {base}   "
                   f"{'ERASE' if state['erase'] else 'DRAW'}   "
                   f"pts={len(pts)}   c=close u=undo e=erase r=reset n=next q=quit")
            cv2.rectangle(big, (0, 0), (256 * ZOOM, 26), (0, 0, 0), -1)
            cv2.putText(big, hdr, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
            cv2.imshow("annotate", big)
            k = cv2.waitKey(20) & 0xFF

            if k in (ord('u'), 8) and pts:
                pts.pop()
            elif k == ord('c') and len(pts) >= 3:
                poly = np.array(pts, np.int32)
                tmp = np.zeros((256, 256), np.uint8)
                cv2.fillPoly(tmp, [poly], 255)
                mask = (np.where(tmp > 0, 0, mask) if state["erase"]
                        else np.maximum(mask, tmp)).astype(np.uint8)
                state["pts"] = []
            elif k == ord('e'):
                state["erase"] = not state["erase"]
            elif k == ord('r'):
                mask[:] = 0; state["pts"] = []
            elif k in (ord('n'), ord('q'), ord('p')):
                cv2.imwrite(mpath, (mask > 127).astype(np.uint8) * 255)
                print(f"saved {mpath}  object px = {int((mask>127).sum())}")
                if k == ord('q'):
                    cv2.destroyAllWindows(); return
                i += 1 if k == ord('n') else -1
                break
    cv2.destroyAllWindows()
    print("done")


if __name__ == "__main__":
    run()
