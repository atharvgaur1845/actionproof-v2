"""Render the ActionProof video: camera moves over 4K stills, annotations, captions, narration.
Low-memory: 3 workers, each caching one source image; frames stream straight into ffmpeg."""
import json, re, subprocess, sys
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.')
from script import SEGMENTS

W, H, FPS = 1920, 1080, 30
LEAD, TAIL, END_HOLD, MOVE = 0.35, 0.75, 2.6, 0.9
VER, CHR, DK = (240, 82, 54), (213, 243, 106), (23, 26, 24)
def dm(size, wt):
    f = ImageFont.truetype('fonts/DMSans.ttf', size)
    try: f.set_variation_by_name(wt)
    except Exception: pass
    return f
F_LABEL, F_CAP = dm(34, 'Bold'), dm(40, 'Medium')
F_CHIP = ImageFont.truetype('/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-SemiBold.ttf', 22)
CARD_BOXES = json.load(open('shots/card_boxes.json'))
WORDS = json.load(open('audio/words.json'))

def word_chars(text, words):
    pos, out, low = 0, [], text.lower()
    for s, e, w in words:
        i = low.find(w.lower(), pos)
        if i < 0: i = pos
        out.append((i, s, e)); pos = i + len(w)
    return out
def t_at_char(wc, idx):
    for i, s, e in wc:
        if i >= idx: return s
    return wc[-1][1]
def img_size(name):
    with Image.open(f'shots/{name}.png') as im: return im.size
def fit(rect, iw, ih, pad=0.14, minw=1500):
    x0, y0, x1, y1 = rect
    w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    w = max(w, h * 16 / 9, minw); h = w * 9 / 16
    if w > iw: w, h = iw, iw * 9 / 16
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x = min(max(cx - w / 2, 0), iw - w); y = min(max(cy - h / 2, 0), ih - h)
    return (x, y, x + w, y + h)

timeline, t = [], 0.0
for seg in SEGMENTS:
    words = WORDS[seg['id']]; wc = word_chars(seg['text'], words)
    dur = LEAD + words[-1][1] + TAIL + (END_HOLD if seg['id'] == 'end' else 0)
    iw, ih = img_size(seg['img']); cb = CARD_BOXES.get(seg['img'].replace('card_', ''), {})
    keys, cpos = [], 0
    for cue, rect, ann in seg['shots']:
        if cue is None: ct = 0.0
        else:
            idx = seg['text'].find(cue, cpos); assert idx >= 0, (seg['id'], cue); cpos = idx
            ct = LEAD + t_at_char(wc, idx)
        if rect == 'full': r = (0, 0, iw, min(ih, iw * 9 / 16))
        elif isinstance(rect, str): r = fit(cb[rect], iw, ih, pad=0.06)
        else: r = fit(rect, iw, ih)
        keys.append(dict(t=ct, cam=r, ann=[(cb[a], '') if isinstance(a, str) else a for a in ann]))
    caps, cur, start = [], '', 0
    for m in re.finditer(r'\S+', seg['text']):
        if cur and len(cur) + len(m.group()) > 46: caps.append((start, cur)); cur = ''
        if not cur: start = m.start()
        cur = (cur + ' ' + m.group()).strip()
        if cur.endswith(('.', '?', ':')) and len(cur) > 18: caps.append((start, cur)); cur = ''
    if cur: caps.append((start, cur))
    capt = [(LEAD + t_at_char(wc, s), c) for s, c in caps]
    capt = [(a, capt[i + 1][0] if i + 1 < len(capt) else LEAD + words[-1][1] + 0.4, c) for i, (a, c) in enumerate(capt)]
    timeline.append(dict(seg=seg, t0=t, dur=dur, keys=keys, caps=capt)); t += dur
TOTAL = t

def ease(x): x = min(max(x, 0), 1); return 4 * x**3 if x < .5 else 1 - (-2 * x + 2)**3 / 2
def lerp(a, b, k): return tuple(p + (q - p) * k for p, q in zip(a, b))
_cache = {}
def load(name):
    if name not in _cache: _cache.clear(); _cache[name] = Image.open(f'shots/{name}.png').convert('RGB')
    return _cache[name]

def render(fi):
    T = fi / FPS
    item = next((x for x in timeline if x['t0'] <= T < x['t0'] + x['dur']), timeline[-1])
    lt = T - item['t0']; keys = item['keys']
    k = max(i for i, kk in enumerate(keys) if kk['t'] <= lt or i == 0)
    cam = keys[k]['cam']
    if k > 0: cam = lerp(keys[k - 1]['cam'], keys[k]['cam'], ease((lt - keys[k]['t'] + 0.15) / MOVE))
    settle = keys[k]['t'] + MOVE if k > 0 else 0
    drift = 0.018 * min(max((lt - settle) / 8, 0), 1)
    cx, cy = (cam[0] + cam[2]) / 2, (cam[1] + cam[3]) / 2
    hw, hh = (cam[2] - cam[0]) / 2 * (1 - drift), (cam[3] - cam[1]) / 2 * (1 - drift)
    cam = (cx - hw, cy - hh, cx + hw, cy + hh)
    fr = load(item['seg']['img']).resize((W, H), Image.BICUBIC, box=cam, reducing_gap=2.0).convert('RGBA')
    sx = W / (cam[2] - cam[0])
    def m(b): return (int((b[0] - cam[0]) * sx) - 12, int((b[1] - cam[1]) * sx) - 10, int((b[2] - cam[0]) * sx) + 12, int((b[3] - cam[1]) * sx) + 10)
    ann = keys[k]['ann']; a = ease((lt - (keys[k]['t'] + (MOVE * 0.8 if k > 0 else 0.3))) / 0.35)
    if k + 1 < len(keys): a *= 1 - ease((lt - (keys[k + 1]['t'] - 0.2)) / 0.2)
    if ann and a > 0:
        ov = Image.new('RGBA', (W, H), (0, 0, 0, int(150 * a))); od = ImageDraw.Draw(ov)
        for b, _ in ann: od.rounded_rectangle(m(b), 16, fill=(0, 0, 0, 0))
        fr = Image.alpha_composite(fr, ov); d = ImageDraw.Draw(fr)
        for b, lab in ann:
            r = m(b); d.rounded_rectangle(r, 16, outline=VER + (int(255 * a),), width=6)
            if lab:
                below = lab.startswith('v '); lab = lab[2:] if below else lab
                tw = d.textlength(lab, font=F_LABEL); ph = 58
                x = min(max(r[0], 30), W - tw - 60)
                y = r[1] - ph - 14 if (r[1] - ph - 14 > 95 and not below) else r[3] + 14
                y = min(y, H - 200)
                d.rounded_rectangle((x, y, x + tw + 40, y + ph), 12, fill=VER + (int(255 * a),))
                d.text((x + 20, y + 9), lab, font=F_LABEL, fill=(255, 255, 255, int(255 * a)))
    d = ImageDraw.Draw(fr); chip = item['seg']['chip']
    if chip:
        tw = d.textlength(chip, font=F_CHIP); d.rounded_rectangle((36, 30, 36 + tw + 36, 78), 10, fill=DK + (225,))
        d.ellipse((50, 48, 62, 60), fill=CHR + (255,)); d.text((72, 41), chip, font=F_CHIP, fill=CHR + (255,))
    for a0, a1, c in item['caps']:
        if a0 <= lt < a1:
            tw = d.textlength(c, font=F_CAP); x = (W - tw) / 2
            d.rounded_rectangle((x - 26, H - 132, x + tw + 26, H - 62), 14, fill=(10, 10, 10, 200))
            d.text((x, H - 124), c, font=F_CAP, fill=(255, 255, 255, 255))
    out = fr.convert('RGB'); sid = item['seg']['id']
    fade = min(1, lt / (0.6 if sid == 'hook' else 0.25), (item['dur'] - lt) / (1.2 if sid == 'end' else 0.2))
    if fade < 1: out = Image.eval(out, lambda v: int(v * max(fade, 0)))
    return out.tobytes()

if __name__ == '__main__':
    inputs, filt = [], []
    for i, it in enumerate(timeline):
        inputs += ['-i', f"audio/{it['seg']['id']}.mp3"]; ms = int((it['t0'] + LEAD) * 1000)
        filt.append(f'[{i}:a]adelay={ms}|{ms}[a{i}]')
    filt.append(''.join(f'[a{i}]' for i in range(len(timeline))) + f'amix=inputs={len(timeline)}:normalize=0,apad,atrim=0:{TOTAL:.2f}[out]')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *inputs, '-filter_complex', ';'.join(filt), '-map', '[out]', '-ar', '48000', 'audio/narration.wav'], check=True)
    def ts(x): return f'{int(x//3600):02}:{int(x%3600//60):02}:{int(x%60):02},{int(x*1000)%1000:03}'
    n, srt = 0, []
    for it in timeline:
        for a0, a1, c in it['caps']: n += 1; srt.append(f"{n}\n{ts(it['t0']+a0)} --> {ts(it['t0']+a1)}\n{c}\n")
    open('captions/actionproof_video.srt', 'w').write('\n'.join(srt))
    if len(sys.argv) > 1:
        for f in map(int, sys.argv[1].split(',')): Image.frombytes('RGB', (W, H), render(f)).save(f'prev_frame_{f}.png')
        sys.exit()
    frames = range(int(TOTAL * FPS))
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-i', 'audio/narration.wav', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '4',
                            '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart', 'actionproof_video.mp4'], stdin=subprocess.PIPE)
    with Pool(3, maxtasksperchild=400) as pool:
        for i, b in enumerate(pool.imap(render, frames, chunksize=10)):
            enc.stdin.write(b)
            if i % 300 == 0: print(f'{i}/{len(frames)}', flush=True)
    enc.stdin.close(); enc.wait(); print('done', round(TOTAL, 1), flush=True)
