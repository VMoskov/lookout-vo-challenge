"""Side-by-side viewer: video frame | reference trajectory (x-y) for train clips.
Keys: n/p next/prev clip, g pick clip (terminal prompt), space pause, left/right seek +-5s, q quit.
Usage: python view_clips.py [release_dir] [clip_key]"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
import matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv) > 1 else "release")
man = json.loads((root / "release_manifest.json").read_text())
clips = [c for c in man["clips"] if c["split"] == "train"]
keys = [c["key"] for c in clips]

def fps(c):
    t = json.loads((root / c["timestamps"]).read_text())["times"]
    return (len(t) - 1) / (t[-1] - t[0])

def pick():
    for i, c in enumerate(clips):
        print(f"{i:3d}  {c['key']}  {c['camera_mode']:10s} {fps(c):6.2f} fps")
    s = input("Clip index or key: ").strip()
    return keys.index(s) if s in keys else int(s)

idx = keys.index(sys.argv[2]) if len(sys.argv) > 2 else pick()

fig, (axv, axt) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.6, 1]})
st = dict(cap=None, paused=False, t0=0.0, wall=0.0)
img = marker = trail = title = None

def load(i):
    global idx, img, marker, trail, title
    idx = i % len(clips)
    c = clips[idx]
    if st["cap"]: st["cap"].release()
    st["cap"] = cv2.VideoCapture(str(root / c["video"]))
    st["ft"] = np.array(json.loads((root / c["timestamps"]).read_text())["times"])
    ref = json.loads((root / c["reference"]).read_text())
    st["rt"] = np.array(ref["times"]); st["pos"] = np.array(ref["positions"])
    st["fi"] = -1; st["wall"] = time.time(); st["t0"] = 0.0
    axv.clear(); axt.clear()
    img = axv.imshow(np.zeros((c["calibration"]["height"], c["calibration"]["width"], 3), np.uint8)); axv.axis("off")
    p = st["pos"]
    axt.plot(p[:, 0], p[:, 1], "-", color="0.7", lw=1)
    trail, = axt.plot([], [], "b-", lw=2)
    marker, = axt.plot([], [], "ro")
    axt.set_aspect("equal"); axt.grid(alpha=.3); axt.set_xlabel("x [m]"); axt.set_ylabel("y [m]")
    title = fig.suptitle("")
    axt.set_title("reference local position (top-down)")

def step(_=None):
    cap, ft = st["cap"], st["ft"]
    if not st["paused"]:
        st["t0"] += time.time() - st["wall"]
    st["wall"] = time.time()
    t = st["t0"]
    if t > ft[-1]: st["t0"] = t = 0.0; cap.set(cv2.CAP_PROP_POS_FRAMES, 0); st["fi"] = -1
    target = min(int(np.searchsorted(ft, t, side="right")) - 1, len(ft) - 1)
    if target != st["fi"]:
        if target < st["fi"] or target > st["fi"] + 60:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target); st["fi"] = target - 1
        while st["fi"] < target - 1: cap.grab(); st["fi"] += 1
        ok, f = cap.read(); st["fi"] += 1
        if ok: img.set_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    rt, p = st["rt"], st["pos"]
    k = int(np.searchsorted(rt, t, side="right"))
    trail.set_data(p[:k, 0], p[:k, 1])
    cur = [np.interp(t, rt, p[:, j]) for j in (0, 1)]
    marker.set_data([cur[0]], [cur[1]])
    title.set_text(f"{keys[idx]} [{idx+1}/{len(clips)}] {clips[idx]['camera_mode']}  t={t:.1f}s  frame {st['fi']}/{len(ft)}"
                   f"{'  (paused)' if st['paused'] else ''}")
    fig.canvas.draw_idle()

def seek(dt): st["t0"] = max(0.0, st["t0"] + dt)

def key(e):
    global idx
    if e.key == "n": load(idx + 1)
    elif e.key == "p": load(idx - 1)
    elif e.key == " ": st["paused"] = not st["paused"]
    elif e.key == "right": seek(5)
    elif e.key == "left": seek(-5)
    elif e.key == "g": load(pick())
    elif e.key == "q": plt.close(fig)

fig.canvas.mpl_connect("key_press_event", key)
load(idx)
timer = fig.canvas.new_timer(interval=40); timer.add_callback(step); timer.start()
plt.show()
