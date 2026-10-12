import numpy as np, wave, sys
SR=44100; DUR=80.0; N=int(SR*DUR); rng=np.random.default_rng(3)
L=np.zeros(N); R=np.zeros(N)
def add(sig,t,pan=0.0,g=1.0):
    i=int(t*SR); n=min(len(sig),N-i)
    if n<=0: return
    L[i:i+n]+=sig[:n]*g*np.sqrt((1-pan)/2); R[i:i+n]+=sig[:n]*g*np.sqrt((1+pan)/2)
def hz(m): return 440*2**((m-69)/12)
def env(n,a,r):
    e=np.ones(n); A=int(a*SR); Rr=int(r*SR)
    if A: e[:A]=np.linspace(0,1,A)
    if Rr: e[-Rr:]*=np.linspace(1,0,Rr)
    return e
def lp(x,c):  # one-pole lowpass
    a=np.exp(-2*np.pi*c/SR); y=np.empty_like(x); s=0.0
    from scipy.signal import lfilter
    return lfilter([1-a],[1,-a],x)
BPM=90; BEAT=60/BPM; BAR=4*BEAT
# chords (midi) Dm, Bb, Gm, A
CH=[[50,57,62,65,69],[46,58,62,65,70],[43,55,62,67,70],[45,57,61,64,69]]
BASS=[38,34,31,33]
def pad(notes,d):
    n=int(d*SR); t=np.arange(n)/SR; s=np.zeros(n)
    for m in notes:
        for det in (-0.08,0.0,0.08):
            f=hz(m)*2**(det/12); ph=rng.random()*6.28
            s+=np.sin(2*np.pi*f*t+ph)+0.35*np.sin(4*np.pi*f*t+ph)+0.12*np.sin(6*np.pi*f*t)
    s*=0.5+0.5*np.sin(2*np.pi*0.25*t)**2*0.4+0.3
    return lp(s,1800)*env(n,1.2,1.4)/len(notes)/3
def pluck(m,d=1.6,b=0.996):
    f=hz(m); P=int(SR/f); n=int(d*SR); buf=rng.uniform(-1,1,P); out=np.zeros(n)
    for k in range(0,n,P):
        m=min(P,n-k); out[k:k+m]=buf[:m]; buf=b*0.5*(buf+np.roll(buf,-1))
    out=lp(out,3500)
    return out*env(n,0.002,0.3)
def kick():
    n=int(.5*SR); t=np.arange(n)/SR; f=45+90*np.exp(-t*30)
    return np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*7)
def hat():
    n=int(.08*SR); x=rng.uniform(-1,1,n); x=x-lp(x,6000); return x*np.exp(-np.arange(n)/SR*60)
def boom():
    n=int(3.5*SR); t=np.arange(n)/SR; f=30+70*np.exp(-t*8)
    s=np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*1.4); nz=lp(rng.uniform(-1,1,n),900)*np.exp(-t*4)*.6
    return s+nz
def whoosh(d=1.2, up=True):
    n=int(d*SR); x=rng.uniform(-1,1,n); t=np.linspace(0,1,n)
    e=np.sin(np.pi*t)**2 if up else np.exp(-t*4)
    out=np.zeros(n); seg=2048
    for k in range(0,n,seg):
        c=300+5000*(np.sin(np.pi*k/n)); out[k:k+seg]=lp(x[k:k+seg],c)
    return out*e
def riser(d):
    n=int(d*SR); t=np.arange(n)/SR; x=rng.uniform(-1,1,n); out=np.zeros(n); seg=2048
    for k in range(0,n,seg): out[k:k+seg]=lp(x[k:k+seg],200+6000*(k/n)**2)
    tone=np.sin(2*np.pi*np.cumsum(110+330*(t/d)**2)/SR)*0.25
    return (out+tone)*(t/d)**2
def shimmer(d=3):
    n=int(d*SR); t=np.arange(n)/SR; s=np.zeros(n)
    for m in (86,90,93,98): s+=np.sin(2*np.pi*hz(m)*t+rng.random()*6)*np.exp(-t*1.5)*(0.5+0.5*np.sin(2*np.pi*6*t))
    return s*env(n,0.01,0.5)/4

# ---- intro 0-6.4
add(riser(3.3),0.0,g=.22)
add(pad([38,50,57],6.8),0,g=.9)
add(boom(),3.3,g=.9); add(shimmer(3.5),3.35,g=.35,pan=.2)
# ---- groove 6.4 -> 62
start=6.4; t=start; bar=0
ARP=[0,2,3,4,3,2,1,2]
while t<67.6:
    c=CH[bar%4]; full=t>8.0; feat=t>61.6
    add(pad(c,BAR+1.4),t,g=.75 if not feat else .9)
    nb=int(BAR*SR); tt=np.arange(nb)/SR
    b=np.sin(2*np.pi*hz(BASS[bar%4])*tt)*env(nb,.02,.3)*.55
    add(lp(b,300),t,g=1)
    for k in range(8):
        m=c[ARP[k]]+12; add(pluck(m,1.4),t+k*BEAT/2,pan=(-.4 if k%2 else .4),g=.32)
    if full:
        for k in range(4):
            if k in (0,2) or feat: add(kick(),t+k*BEAT,g=.55)
            add(hat(),t+k*BEAT+BEAT/2,g=.10,pan=.3)
            if feat: add(hat(),t+k*BEAT+BEAT*0.75,g=.06,pan=-.3)
    t+=BAR; bar+=1
# transitions whooshes
for tt in [5.6,9.8,15.4,18.1,23.1,26.5,31.3,34.5,36.4,39.6,43.6,48.2,54.2,57.8,61.6]:
    add(whoosh(1.1),tt-0.35,g=.13,pan=rng.uniform(-.5,.5))
add(riser(2.0),65.6,g=.2)
# ---- CTA 67.6 -> 80
add(boom(),67.6,g=.85); add(shimmer(4),67.65,g=.35)
add(pad([38,50,57,62,65,69],12.5),67.6,g=1.0)
t=67.6; k=0
while t<77:
    seq=[62,65,69,74,72,69,65,69]; add(pluck(seq[k%8],1.8),t,pan=(-.3 if k%2 else .3),g=.3); t+=BEAT/2; k+=1
add(pluck(74,4),77.0,g=.35); add(pluck(62,4),77.0,g=.25); add(shimmer(3),77.0,g=.25)
# ---- reverb
ir_n=int(2.8*SR); ti=np.arange(ir_n)/SR
def ir(seed):
    r=np.random.default_rng(seed).uniform(-1,1,ir_n)*np.exp(-ti*2.6); return lp(r,5000)*0.06
def conv(x,h):
    n=len(x)+len(h); F=1<<int(np.ceil(np.log2(n)))
    return np.fft.irfft(np.fft.rfft(x,F)*np.fft.rfft(h,F),F)[:len(x)]
Lw=conv(L,ir(1)); Rw=conv(R,ir(2))
L2=L+Lw*0.9; R2=R+Rw*0.9
# master: fade in/out, soft clip, normalize
m=np.ones(N); m[:int(.3*SR)]=np.linspace(0,1,int(.3*SR)); fo=int(2.5*SR); m[-fo:]=np.linspace(1,0,fo)**1.5
st=np.stack([L2*m,R2*m],1); st/=np.max(np.abs(st))+1e-9; st=np.tanh(st*1.4)/np.tanh(1.4)*0.89
with wave.open(sys.argv[1],'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st*32767).astype('<i2').tobytes())
print('ok')
