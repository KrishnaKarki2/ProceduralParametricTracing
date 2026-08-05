# ProceduralParametricTracing
Procedural Parametric Tracing in Desmos 

https://github.com/user-attachments/assets/5014ec9b-4f58-4106-a6fc-6f7e0ef811ca

# Cell 1: install dependencies
```sh
!apt-get install -y libagg-dev libpotrace-dev pkg-config build-essential > /dev/null 2>&1
!pip install flask flask-cors pillow numpy opencv-python pypotrace pyngrok -q
print('Done installing.')`
```
# Cell 2: mount google drive
```sh
from google.colab import drive
drive.mount('/content/drive')
print('Drive mounted at /content/drive')
```
# Cell 3: configure paths and quality settings
```sh

# sort frames in order
import re

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

frames = sorted(
    [f for f in os.listdir(FRAME_DIR) if f.endswith(FILE_EXT)],
    key=extract_number
)

print(f'Found {len(frames)} frames in {FRAME_DIR}')
print('First few:', frames[:5])
print('Last few:', frames[-5:])
```
```sh
Cell 4: process frames → bezier latex expressions
import cv2, numpy as np, potrace, os
from multiprocessing import Pool, cpu_count
from time import time
# note: if a frame is missing, the processing will continue on the next available frame
# e.g. if in frames 1-4, frame 2 is missing from the google frames folder, it will process ['frame1', 'frame3', 'frame4']

def get_contours(filename, nudge=0.33):
    image = cv2.imread(filename)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    src   = cv2.bilateralFilter(gray, 5, 50, 50) if BILATERAL_FILTER else gray
    if ADAPTIVE_CANNY:
        median = max(10, min(245, np.median(src)))
        lo = int(max(0,   (1 - nudge) * median))
        hi = int(min(255, (1 + nudge) * median))
        edged = cv2.Canny(src, lo, hi, L2gradient=USE_L2_GRADIENT)
    else:
        edged = cv2.Canny(src, 30, 200)
    return edged[::-1], image.shape

def get_trace(data):
    for i in range(len(data)):
        data[i][data[i] > 1] = 1
    pixels   = data.shape[0] * data.shape[1]
    turdsize = TURDSIZE if TURDSIZE is not None else max(2, min(8, round(pixels / 40000)))
    bmp  = potrace.Bitmap(data)
    path = bmp.trace(turdsize, potrace.TURNPOLICY_MINORITY, ALPHAMAX, 1, OPTTOLERANCE)
    return path

def get_latex(filename):
    latex = []
    edged, _ = get_contours(filename)
    for curve in get_trace(edged).curves:
        start = curve.start_point
        for seg in curve.segments:
            x0, y0 = start
            if seg.is_corner:
                x1, y1 = seg.c
                x2, y2 = seg.end_point
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x0, x1, y0, y1))
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x1, x2, y1, y2))
            else:
                x1, y1 = seg.c1
                x2, y2 = seg.c2
                x3, y3 = seg.end_point
                latex.append(
                    '((1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)),'
                    '(1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)))'
                    % (x0,x1,x1,x2,x1,x2,x2,x3,y0,y1,y1,y2,y1,y2,y2,y3))
            start = seg.end_point
    return latex

def get_points(filename):
    edged, _ = get_contours(filename)
    segs_out = []
    for curve in get_trace(edged).curves:
        start = curve.start_point
        for seg in curve.segments:
            if seg.is_corner:
                segs_out.append(('line', start, seg.c))
                segs_out.append(('line', seg.c, seg.end_point))
            else:
                segs_out.append(('cubic', start, seg.c1, seg.c2, seg.end_point))
            start = seg.end_point
    return segs_out

def process_frame(fname):
    filename = os.path.join(FRAME_DIR, fname)
    exprs = [{'id': 'expr-%d' % (j+1), 'latex': e, 'color': COLOUR, 'secret': False}
             for j, e in enumerate(get_latex(filename))]
    pts  = get_points(filename)
    img  = cv2.imread(filename)
    h, w = img.shape[:2] if img is not None else (0, 0)
    return exprs, pts, h, w

start = time()
print(f'Processing {len(frames)} frames on {cpu_count()} cores...')

with Pool(processes=cpu_count()) as pool:
    results = pool.map(process_frame, frames)

frame_latex  = []
frame_points = []
max_h = max_w = 0

for i, (exprs, pts, h, w) in enumerate(results):
    frame_latex.append(exprs)
    frame_points.append(pts)
    max_h = max(max_h, h)
    max_w = max(max_w, w)
    print(f'\r--> {i+1}/{len(frames)} ({len(exprs)} segments)', end='')

print(f'\n```python\n# Cell 4: process frames → bezier latex expressions
import cv2, numpy as np, potrace, os
from multiprocessing import Pool, cpu_count
from time import time
# note: if a frame is missing, the processing will continue on the next available frame
# e.g. if in frames 1-4, frame 2 is missing from the google frames folder, it will process ['frame1', 'frame3', 'frame4']

def get_contours(filename, nudge=0.33):
    image = cv2.imread(filename)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    src   = cv2.bilateralFilter(gray, 5, 50, 50) if BILATERAL_FILTER else gray
    if ADAPTIVE_CANNY:
        median = max(10, min(245, np.median(src)))
        lo = int(max(0,   (1 - nudge) * median))
        hi = int(min(255, (1 + nudge) * median))
        edged = cv2.Canny(src, lo, hi, L2gradient=USE_L2_GRADIENT)
    else:
        edged = cv2.Canny(src, 30, 200)
    return edged[::-1], image.shape

def get_trace(data):
    for i in range(len(data)):
        data[i][data[i] > 1] = 1
    pixels   = data.shape[0] * data.shape[1]
    turdsize = TURDSIZE if TURDSIZE is not None else max(2, min(8, round(pixels / 40000)))
    bmp  = potrace.Bitmap(data)
    path = bmp.trace(turdsize, potrace.TURNPOLICY_MINORITY, ALPHAMAX, 1, OPTTOLERANCE)
    return path

def get_latex(filename):
    latex = []
    edged, _ = get_contours(filename)
    for curve in get_trace(edged).curves:
        start = curve.start_point
        for seg in curve.segments:
            x0, y0 = start
            if seg.is_corner:
                x1, y1 = seg.c
                x2, y2 = seg.end_point
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x0, x1, y0, y1))
                latex.append('((1-t)%f+t%f,(1-t)%f+t%f)' % (x1, x2, y1, y2))
            else:
                x1, y1 = seg.c1
                x2, y2 = seg.c2
                x3, y3 = seg.end_point
                latex.append(
                    '((1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)),'
                    '(1-t)((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f))+t((1-t)((1-t)%f+t%f)+t((1-t)%f+t%f)))'
                    % (x0,x1,x1,x2,x1,x2,x2,x3,y0,y1,y1,y2,y1,y2,y2,y3))
            start = seg.end_point
    return latex

def get_points(filename):
    edged, _ = get_contours(filename)
    segs_out = []
    for curve in get_trace(edged).curves:
        start = curve.start_point
        for seg in curve.segments:
            if seg.is_corner:
                segs_out.append(('line', start, seg.c))
                segs_out.append(('line', seg.c, seg.end_point))
            else:
                segs_out.append(('cubic', start, seg.c1, seg.c2, seg.end_point))
            start = seg.end_point
    return segs_out

def process_frame(fname):
    filename = os.path.join(FRAME_DIR, fname)
    exprs = [{'id': 'expr-%d' % (j+1), 'latex': e, 'color': COLOUR, 'secret': False}
             for j, e in enumerate(get_latex(filename))]
    pts  = get_points(filename)
    img  = cv2.imread(filename)
    h, w = img.shape[:2] if img is not None else (0, 0)
    return exprs, pts, h, w

start = time()
print(f'Processing {len(frames)} frames on {cpu_count()} cores...')

with Pool(processes=cpu_count()) as pool:
    results = pool.map(process_frame, frames)

frame_latex  = []
frame_points = []
max_h = max_w = 0

for i, (exprs, pts, h, w) in enumerate(results):
    frame_latex.append(exprs)
    frame_points.append(pts)
    max_h = max(max_h, h)
    max_w = max(max_w, w)
    print(f'\r--> {i+1}/{len(frames)} ({len(exprs)} segments)', end='')

print(f'\n\nDone in {time()-start:.1f}s')
print(f'Total expressions: {sum(len(f) for f in frame_latex)}')
```
# Cell 5: draws each frame sequentially
```sh
import json, threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.colab.output import eval_js

CALCULATOR_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Desmos Bezier Renderer</title>
  <script src="https://www.desmos.com/api/v1.8/calculator.js?apiKey=dcb31709b452b1cf9dc26972add0fda6"></script>
  <style>html,body,#calc{width:100%;height:100%;margin:0;padding:0;overflow:hidden}</style>
</head>
<body>
<div id="calc"></div>
<script>
var calc = Desmos.GraphingCalculator(document.getElementById('calc'), {
  showGrid: """ + str(SHOW_GRID).lower() + """,
  expressionsCollapsed: true
});

var allExprs = [];
var drawIndex = 0;
var drawTimer = null;
var DRAW_SPEED = 5;
var TICK_MS = 16;
var currentFrame = 0;
var totalFrames = """ + str(len(frame_latex)) + """;
var autoPlaying = false;

calc.setMathBounds({left:0, right:""" + str(max_w) + """, bottom:0, top:""" + str(max_h) + """});

function clearCanvas() {
  var existing = calc.getExpressions();
  if (existing.length > 0) calc.removeExpressions(existing);
}

function loadFrame(frameIndex, onComplete) {
  fetch('/?frame=' + frameIndex)
    .then(r => r.json())
    .then(data => {
      if (!data.result) { if (onComplete) onComplete(); return; }
      allExprs = data.result;
      startDraw(onComplete);
    })
    .catch(e => console.error(e));
}

function startDraw(onComplete) {
  clearCanvas();
  drawIndex = 0;
  if (drawTimer) clearInterval(drawTimer);

  drawTimer = setInterval(function() {
    if (drawIndex >= allExprs.length) {
      clearInterval(drawTimer);
      drawTimer = null;
      if (onComplete) onComplete();
      return;
    }
    var batch = allExprs.slice(drawIndex, drawIndex + DRAW_SPEED);
    batch.forEach(function(expr) { calc.setExpression(expr); });
    drawIndex += DRAW_SPEED;
  }, TICK_MS);
}

function playNext() {
  if (!autoPlaying) return;
  currentFrame = (currentFrame + 1) % totalFrames;
  updateFrameLabel();
  loadFrame(currentFrame, playNext); // draw fully, then call playNext again
}

function togglePlay() {
  autoPlaying = !autoPlaying;
  document.getElementById('playBtn').innerText = autoPlaying ? 'Pause' : 'Play';
  if (autoPlaying) playNext();
  else {
    if (drawTimer) { clearInterval(drawTimer); drawTimer = null; }
  }
}

function goNext() {
  autoPlaying = false;
  document.getElementById('playBtn').innerText = 'Play';
  if (drawTimer) clearInterval(drawTimer);
  currentFrame = (currentFrame + 1) % totalFrames;
  updateFrameLabel();
  loadFrame(currentFrame);
}

function goPrev() {
  autoPlaying = false;
  document.getElementById('playBtn').innerText = 'Play';
  if (drawTimer) clearInterval(drawTimer);
  currentFrame = (currentFrame - 1 + totalFrames) % totalFrames;
  updateFrameLabel();
  loadFrame(currentFrame);
}

function redraw() {
  autoPlaying = false;
  document.getElementById('playBtn').innerText = 'Play';
  if (drawTimer) clearInterval(drawTimer);
  loadFrame(currentFrame);
}

function setSpeed(v) {
  DRAW_SPEED = parseInt(v);
  document.getElementById('speedLabel').innerText = v + ' curves/tick';
}

function updateFrameLabel() {
  document.getElementById('frameLabel').innerText = 'Frame ' + (currentFrame+1) + '/' + totalFrames;
}

// Load first frame on start
loadFrame(0);
updateFrameLabel();
</script>

<div style="position:fixed;bottom:10px;left:10px;z-index:999;background:#fff;padding:8px 12px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.2);font-family:sans-serif;font-size:13px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
  <button onclick="goPrev()">&#9664; Prev</button>
  <button id="playBtn" onclick="togglePlay()">&#9654; Play</button>
  <button onclick="goNext()">Next &#9654;</button>
  <button onclick="redraw()">&#8635; Redraw</button>
  <label>Speed:
    <input type="range" min="1" max="50" value="5" oninput="setSpeed(this.value)">
    <span id="speedLabel">5 curves/tick</span>
  </label>
  <span id="frameLabel">Frame 1/""" + str(len(frame_latex)) + """</span>
</div>
</body>
</html>
"""

server_app = Flask(__name__)
CORS(server_app)

@server_app.route('/')
def index():
    f = request.args.get('frame')
    if f is None:
        return CALCULATOR_HTML, 200, {'Content-Type': 'text/html'}
    try:
        f = int(f)
    except (ValueError, TypeError):
        return jsonify({'result': None})
    if f < 0 or f >= len(frame_latex):
        return jsonify({'result': None})
    return jsonify({'result': frame_latex[f]})

t = threading.Thread(target=lambda: server_app.run(port=5000, use_reloader=False))
t.daemon = True
t.start()

print('Open this link:')
print(eval_js("google.colab.kernel.proxyPort(5000)"))\n```python\n# Cell 5: draws each frame sequentially
```
# Cell 6: renders mp4
```sh
import cv2
import numpy as np
from time import time

MP4_PATH   = '/content/drive/MyDrive/output.mp4'
MP4_FPS    = 30
BG_COLOR   = (255, 255, 255)
LINE_WIDTH = 1
T_STEPS    = 40

def hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

LINE_COLOR = hex_to_bgr(COLOUR)

def cubic_bezier_points(p0, p1, p2, p3, steps=T_STEPS):
    pts = []
    for i in range(steps+1):
        t = i / steps
        mt = 1 - t
        x = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
        y = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
        pts.append((int(round(x)), int(round(y))))
    return pts

def render_frame(segments, width, height):
    img = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    for seg in segments:
        if seg[0] == 'line':
            p0 = (int(round(seg[1][0])), int(round(seg[1][1])))
            p1 = (int(round(seg[2][0])), int(round(seg[2][1])))
            cv2.line(img, p0, p1, LINE_COLOR, LINE_WIDTH, cv2.LINE_AA)
        elif seg[0] == 'cubic':
            pts = cubic_bezier_points(seg[1], seg[2], seg[3], seg[4])
            for i in range(len(pts)-1):
                cv2.line(img, pts[i], pts[i+1], LINE_COLOR, LINE_WIDTH, cv2.LINE_AA)

    img = cv2.flip(img, 0)

    return img

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(MP4_PATH, fourcc, MP4_FPS, (max_w, max_h))

total = len(frame_points)
start = time()

for i, segs in enumerate(frame_points):
    img = render_frame(segs, max_w, max_h)
    out.write(img)
    print(f'\r--> Rendered frame {i+1}/{total}', end='')

out.release()
print(f'\n\nDone in {time()-start:.1f}s')
print(f'Saved to {MP4_PATH}')\n```\n\n
[Uploading ProceduralParametricTracing.md…]()
``` 
