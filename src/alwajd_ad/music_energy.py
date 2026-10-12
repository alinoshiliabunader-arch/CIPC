import numpy as np, wave, sys
from scipy.signal import butter, sosfilt

SR = 44100; DUR = 80.0; N = int(SR * DUR)
rng = np.random.default_rng(11)
BPM = 125; BEAT = 60 / BPM; BAR = 4 * BEAT; STEP = BEAT / 4

def bus(): return np.zeros((N, 2))
DR, BA, CH, LE, FX = bus(), bus(), bus(), bus(), bus()
SC = np.ones(N)  # sidechain gain

def add(b, sig, t, pan=0.0, g=1.0):
    i = int(round(t * SR))
    if i >= N: return
    if i < 0: sig = sig[-i:]; i = 0
    n = min(len(sig), N - i)
    b[i:i+n, 0] += sig[:n] * g * np.sqrt((1 - pan) / 2) * 1.414
    b[i:i+n, 1] += sig[:n] * g * np.sqrt((1 + pan) / 2) * 1.414

def filt(x, f, kind='low', order=2):
    f = np.clip(f, 20, SR / 2 - 100)
    return sosfilt(butter(order, f, kind, fs=SR, output='sos'), x)
def bp(x, lo, hi): return sosfilt(butter(2, [lo, hi], 'band', fs=SR, output='sos'), x)
def hz(m): return 440 * 2 ** ((m - 69) / 12)
def tt(d): return np.arange(int(d * SR)) / SR
def saw(f, t, ph=0.0): return 2 * ((f * t + ph) % 1.0) - 1
def sweep_lp(x, f0, f1, curve=2.0, blk=1024):
    out = np.zeros_like(x); n = len(x); zi = None
    for k in range(0, n, blk):
        u = (k / max(1, n - 1)) ** curve
        sos = butter(2, np.clip(f0 + (f1 - f0) * u, 30, 20000), 'low', fs=SR, output='sos')
        if zi is None: zi = np.zeros((sos.shape[0], 2))
        out[k:k+blk], zi = sosfilt(sos, x[k:k+blk], zi=zi)
    return out

# ---------- instruments ----------
def kick(g=1.0):
    t = tt(0.42); f = 48 + 170 * np.exp(-t * 38)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6.5)
    click = filt(rng.uniform(-1, 1, len(t)), 3000, 'high') * np.exp(-t * 300) * 0.4
    return np.tanh((body + click) * 2.2) * 0.9 * g
def clap():
    t = tt(0.35); n = rng.uniform(-1, 1, len(t)); e = np.zeros(len(t))
    for d in (0, 0.011, 0.022): e += (t >= d) * np.exp(-np.clip(t - d, 0, None) * (90 if d < 0.02 else 14))
    return bp(n, 900, 5000) * e * 0.8
def hat(open_=False):
    t = tt(0.32 if open_ else 0.06); n = rng.uniform(-1, 1, len(t))
    return filt(n, 7500, 'high') * np.exp(-t * (11 if open_ else 70))
def snare():
    t = tt(0.22); n = rng.uniform(-1, 1, len(t))
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)
    return (bp(n, 1200, 8000) * np.exp(-t * 18) + tone * 0.5) * 0.7
def crash(d=3.0):
    t = tt(d); n = rng.uniform(-1, 1, len(t))
    return filt(n, 5000, 'high') * np.exp(-t * 1.6) * 0.5
def impact():
    t = tt(3.0); f = 32 + 80 * np.exp(-t * 6)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.5)
    nz = filt(rng.uniform(-1, 1, len(t)), 1200) * np.exp(-t * 5) * 0.7
    return np.tanh((boom + nz) * 1.8)
def riser(d, f0=400, f1=9000):
    t = tt(d); n = rng.uniform(-1, 1, len(t))
    x = sweep_lp(n, f0, f1, 2.2) * (t / d) ** 1.6
    tone = saw(1, t) * 0  # placeholder for clarity
    pitch = np.sin(2 * np.pi * np.cumsum(220 + 880 * (t / d) ** 2) / SR) * (t / d) ** 2 * 0.18
    return x + pitch
def downlifter(d=1.5):
    t = tt(d); n = rng.uniform(-1, 1, len(t))
    return sweep_lp(n, 8000, 300, 0.6) * np.exp(-t * 2.5)
def whoosh(d=0.9):
    t = tt(d); n = rng.uniform(-1, 1, len(t))
    return sweep_lp(n, 500, 6000, 1.0) * np.sin(np.pi * t / d) ** 2
def bass_note(m, d):
    t = tt(d); f = hz(m)
    x = saw(f, t) + 0.6 * np.sin(2 * np.pi * f * t) + 0.5 * saw(f * 1.005, t, .3)
    env = np.minimum(1, t / 0.004) * np.exp(-t * 4)
    return sweep_lp(x, 1200, 300, 0.3, 256) * env * 0.55
def supersaw(notes, d, cutoff=4200):
    t = tt(d); x = np.zeros(len(t))
    for m in notes:
        f = hz(m)
        for k, det in enumerate((-0.18, -0.11, -0.05, 0, 0.05, 0.11, 0.18)):
            x += saw(f * 2 ** (det / 12), t, rng.random()) * (0.9 if det == 0 else 0.6)
    env = np.minimum(1, t / 0.01) * np.minimum(1, (d - t) / 0.04).clip(0)
    return filt(x, cutoff, 'low') * env / (len(notes) * 5)
def pluck(m, d=0.22, bright=6000):
    t = tt(d); f = hz(m)
    x = saw(f, t) + saw(f * 1.007, t, .4) * .7 + np.sin(2 * np.pi * 2 * f * t) * .3
    fenv = 400 + bright * np.exp(-t * 28)
    return sweep_lp(x, fenv[0], fenv[-1], 0.4, 256) * np.exp(-t * 12) * 0.5
def snare_roll(t0, bars):
    d = bars * BAR; t = 0.0
    while t < d:
        u = t / d; step = STEP * (4 if u < .25 else 2 if u < .5 else 1 if u < .8 else .5)
        add(DR, snare(), t0 + t, g=0.25 + 0.6 * u, pan=0.0); t += step

CHORDS = [[50, 53, 57, 62], [46, 50, 53, 58], [48, 53, 57, 60], [48, 52, 55, 60]]  # Dm Bb F C
ROOTS = [38, 34, 41, 36]
ARP = [0, 1, 2, 3, 2, 1, 2, 3, 0, 2, 3, 2, 1, 2, 3, 2]

def section(t0, nbars, lead=True, lead_oct=24, chord_cut=4200, crash_every=8, breaks=()):
    kicks = []
    for b in range(nbars):
        tb = t0 + b * BAR; c = CHORDS[b % 4]; r = ROOTS[b % 4]
        brk = b in breaks
        if b % crash_every == 0 or (b - 1 in breaks and not brk): add(DR, crash(), tb, g=.5)
        if brk and b + 1 not in breaks: snare_roll(tb, 1)
        for s in range(16):
            ts = tb + s * STEP
            if brk:
                if s % 4 == 0 and b + 1 not in breaks: add(DR, hat(), ts, g=.08)
                continue
            if s % 4 == 0: add(DR, kick(), ts); kicks.append(ts)
            if s in (4, 12): add(DR, clap(), ts, g=.55, pan=.05)
            if s in (2, 6, 10, 14): add(DR, hat(True), ts, g=.13, pan=.25)
            if s % 2 == 1: add(DR, hat(), ts, g=.08 + .03 * (s % 4 == 3), pan=-.3)
            if s in (2, 6, 10, 14): add(BA, bass_note(r, STEP * 1.8), ts, g=1.0)
            if s in (3, 7, 11, 15): add(BA, bass_note(r + 12, STEP * .9), ts, g=.45)
            if lead:
                m = c[ARP[s] % len(c)] + lead_oct - (12 if c[ARP[s]] > 60 else 0)
                add(LE, pluck(m), ts, g=.42, pan=(-.35 if s % 2 else .35))
        add(CH, supersaw(c, BAR), tb, g=.9)
    return kicks

def duck(kts, depth=.7, rel=.16):
    for k in kts:
        i = int(k * SR); n = min(int(BEAT * SR), N - i)
        if n <= 0: continue
        x = np.arange(n) / SR
        SC[i:i+n] = np.minimum(SC[i:i+n], 1 - depth * np.exp(-x / rel * 2.2) * (x < BEAT))

# ---------- arrangement ----------
# INTRO 0 - 6.4: filtered chords opening, heartbeat kick, impact on logo, snare roll + riser into the drop
intro_bars = 6.4 / BAR
pad = np.zeros(int(6.4 * SR))
for b in range(4):
    s = supersaw(CHORDS[b % 4], BAR); i = int((6.4 - (4 - b) * BAR) * SR)
    seg_ = s[:len(pad) - i] if i >= 0 else s[-i:]
    i = max(i, 0); pad[i:i+len(seg_)] += seg_
pad[:int(.0 * SR)] = 0
pad = sweep_lp(pad, 250, 3500, 2.0) * np.minimum(1, np.arange(len(pad)) / SR / 1.2)
add(CH, pad, 0, g=1.0)
t = 6.4 - 8 * BEAT
while t < 6.4 - BEAT * .5:
    if t >= 0: add(DR, filt(kick(.5), 500), t)
    t += BEAT
add(FX, impact(), 3.3, g=.9); add(FX, crash(3), 3.3, g=.5)
add(FX, riser(3.0), 3.4, g=.35)
snare_roll(6.4 - 2 * BAR, 2)

# DROP A 6.4 -> 61.6 (lead arp enters after 4 bars)
nA = int(np.floor((61.6 - 6.4) / BAR))  # 28 bars
k1 = section(6.4, 4, lead=False)
k2 = section(6.4 + 4 * BAR, nA - 4, lead=True, breaks=(12, 13))
duck(k1 + k2)
endA = 6.4 + nA * BAR
# BUILD (features) endA -> 67.6: kick 4otf, chords opening, snare roll, riser
nb = (67.6 - endA)
t = endA; kb = []
while t < 67.6 - BEAT * .9:
    add(DR, kick(.9), t); kb.append(t); t += BEAT
duck(kb, .5)
bp_ = np.zeros(int(nb * SR) + 1); j = 0
while j * BAR < nb:
    s = supersaw(CHORDS[j % 4], BAR); i = int(j * BAR * SR); n = min(len(s), len(bp_) - i); bp_[i:i+n] += s[:n]; j += 1
add(CH, sweep_lp(bp_, 600, 9000, 1.5), endA, g=1.0)
snare_roll(67.6 - 2 * BAR, 2)
add(FX, riser(nb, 300, 12000), endA, g=.4)
# DROP B 67.6 -> ~77.2, brighter lead an octave up
nB = 5
kB = section(67.6, nB, lead=True, lead_oct=36, crash_every=4)
duck(kB)
tEnd = 67.6 + nB * BAR
add(FX, impact(), tEnd, g=1.0); add(FX, crash(4), tEnd, g=.6)
add(CH, supersaw([38, 50, 57, 62, 65, 69], 2.8) * np.exp(-tt(2.8) * 1.1), tEnd, g=1.3)
add(LE, pluck(74, 1.5, 4000), tEnd, g=.5)

# whooshes on the scene transitions (between the beats)
for x in [9.8, 15.4, 18.1, 23.1, 26.5, 31.3, 34.5, 36.4, 39.6, 43.6, 48.2, 54.2, 57.8]:
    add(FX, whoosh(), x - 0.45, g=.18, pan=rng.uniform(-.5, .5))
add(FX, downlifter(), 61.6, g=.25)

# ---------- mix ----------
def conv(x, h):
    n = len(x) + len(h); F = 1 << int(np.ceil(np.log2(n)))
    return np.fft.irfft(np.fft.rfft(x, F) * np.fft.rfft(h, F), F)[:len(x)]
ti = tt(1.8)
def ir(seed): return filt(np.random.default_rng(seed).uniform(-1, 1, len(ti)) * np.exp(-ti * 3.5), 6000) * .05
CHd = CH * SC[:, None]; BAd = BA * SC[:, None]; LEd = LE * (0.6 + 0.4 * SC)[:, None]
wet_in = CHd * .5 + LEd * .8 + FX * .4
wet = np.stack([conv(wet_in[:, 0], ir(1)), conv(wet_in[:, 1], ir(2))], 1)
mix = DR * 1.0 + BAd * 1.1 + CHd * 0.85 + LEd * 0.75 + FX * 0.9 + wet * 0.8
for c in range(2): mix[:, c] = filt(mix[:, c], 28, 'high')
m = np.ones(N); fo = int(1.2 * SR); m[-fo:] = np.linspace(1, 0, fo) ** 1.5
mix *= m[:, None]
mix /= np.percentile(np.abs(mix), 99.9) + 1e-9
mix = np.tanh(mix * 1.2) / np.tanh(1.2) * 0.95
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(mix, -1, 1) * 32767).astype('<i2').tobytes())
print('ok', nA, round(endA, 2), round(tEnd, 2))
