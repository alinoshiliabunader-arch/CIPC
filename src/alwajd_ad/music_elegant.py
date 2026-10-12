import numpy as np, wave, sys
from scipy.signal import butter, sosfilt
SR=44100; DUR=104.5; N=int(SR*DUR); rng=np.random.default_rng(21)
BPM=108; BEAT=60/BPM; BAR=4*BEAT; E=BEAT/2
def bus(): return np.zeros((N,2))
PI,ST,BA,DR,FX=bus(),bus(),bus(),bus(),bus()
def add(b,sig,t,pan=0.0,g=1.0):
    i=int(round(t*SR))
    if i>=N: return
    if i<0: sig=sig[-i:]; i=0
    n=min(len(sig),N-i); b[i:i+n,0]+=sig[:n]*g*np.sqrt((1-pan)/2)*1.414; b[i:i+n,1]+=sig[:n]*g*np.sqrt((1+pan)/2)*1.414
def filt(x,f,kind='low',o=2): return sosfilt(butter(o,np.clip(f,20,SR/2-100),kind,fs=SR,output='sos'),x)
def bp(x,lo,hi): return sosfilt(butter(2,[lo,hi],'band',fs=SR,output='sos'),x)
def hz(m): return 440*2**((m-69)/12)
def tt(d): return np.arange(int(d*SR))/SR
def saw(f,t,ph=0.0): return 2*((f*t+ph)%1.0)-1
def sweep_lp(x,f0,f1,curve=2.0,blk=1024):
    out=np.zeros_like(x); n=len(x); zi=None
    for k in range(0,n,blk):
        u=(k/max(1,n-1))**curve; sos=butter(2,np.clip(f0+(f1-f0)*u,30,20000),'low',fs=SR,output='sos')
        if zi is None: zi=np.zeros((sos.shape[0],2))
        out[k:k+blk],zi=sosfilt(sos,x[k:k+blk],zi=zi)
    return out
# ---- instruments
def piano(m,d=2.2,vel=1.0):
    t=tt(d); f=hz(m); x=np.zeros(len(t))
    for h,a in enumerate([1,.55,.32,.2,.12,.07,.05],1):
        fh=f*h*np.sqrt(1+0.0004*h*h); x+=a*np.sin(2*np.pi*fh*t)*np.exp(-t*(1.4+h*.9))
    ham=filt(rng.uniform(-1,1,len(t)),3000)*np.exp(-t*90)*.15
    return (x+ham)*np.minimum(1,t/.003)*np.minimum(1,(d-t)/.08).clip(0)*.32*vel
def strings(notes,d,bright=2600,att=.5):
    t=tt(d); x=np.zeros(len(t))
    for m in notes:
        f=hz(m)
        for det in (-.07,0,.07):
            vib=1+.003*np.sin(2*np.pi*5.2*t+rng.random()*6)
            x+=saw(f*2**(det/12),np.cumsum(vib)/SR*1.0 if False else t*vib,rng.random())
    env=np.minimum(1,t/att)*np.minimum(1,(d-t)/.6).clip(0)
    return filt(filt(x,bright),120,'high')*env/(len(notes)*3)*.6
def bass(m,d):
    t=tt(d); f=hz(m); x=np.sin(2*np.pi*f*t)+.25*np.sin(4*np.pi*f*t)+.15*saw(f,t)
    return filt(x,500)*np.minimum(1,t/.006)*np.exp(-t*1.2)*np.minimum(1,(d-t)/.05).clip(0)*.5
def kick(g=1.0):
    t=tt(.4); f=46+110*np.exp(-t*34); b=np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*7)
    return np.tanh(b*1.6)*.75*g
def clap():
    t=tt(.35); n=rng.uniform(-1,1,len(t)); e=np.zeros(len(t))
    for d in (0,.012,.024): e+=(t>=d)*np.exp(-np.clip(t-d,0,None)*(80 if d<.02 else 13))
    return bp(n,1000,6000)*e*.5
def shaker():
    t=tt(.09); n=rng.uniform(-1,1,len(t)); return filt(n,6500,'high')*np.sin(np.pi*t/.09)**2*.5
def snap():
    t=tt(.15); n=rng.uniform(-1,1,len(t)); return bp(n,2000,7000)*np.exp(-t*45)*.6
def cymbal(d=3.5):
    t=tt(d); n=rng.uniform(-1,1,len(t)); return filt(n,6000,'high')*np.exp(-t*1.3)*.3*np.minimum(1,t/.01)
def swell(d):
    t=tt(d); n=rng.uniform(-1,1,len(t)); return sweep_lp(n,300,7000,2.2)*(t/d)**2.2*.5
def impact():
    t=tt(3.2); f=34+70*np.exp(-t*6); b=np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*1.4)
    return np.tanh((b+filt(rng.uniform(-1,1,len(t)),900)*np.exp(-t*5)*.5)*1.4)*.8
def bell(m,d=2.5):
    t=tt(d); f=hz(m); x=np.sin(2*np.pi*f*t)*np.exp(-t*2.2)+.4*np.sin(2*np.pi*f*2.76*t)*np.exp(-t*5)+.2*np.sin(2*np.pi*f*5.4*t)*np.exp(-t*8)
    return x*np.minimum(1,t/.002)*.25
def whoosh(d=.9):
    t=tt(d); return sweep_lp(rng.uniform(-1,1,len(t)),500,5000,1.0)*np.sin(np.pi*t/d)**2
def rain_amb(d):
    t=tt(d); n=rng.uniform(-1,1,len(t)); x=bp(n,600,9000)*.5+filt(n,500)*.6
    dr=np.zeros(len(t)); idx=rng.integers(0,len(t),int(d*90)); dr[idx]=rng.uniform(.3,1,len(idx)); dr=filt(dr,2500,'high')*2.5
    return (x+dr)*np.minimum(1,t/1.0)*np.minimum(1,(d-t)/1.2).clip(0)
def thunder():
    t=tt(4.5); n=rng.uniform(-1,1,len(t))
    return np.tanh((filt(n,2500)*np.exp(-t*9)+filt(n,160)*np.exp(-t*.9)*(1+.6*np.sin(2*np.pi*1.7*t))*3)*1.5)*np.minimum(1,t/.02)
# ---- harmony: Bm - G - D - A (uplifting)
CH=[[59,62,66],[55,59,62],[57,62,66],[57,61,64]]
ROOT=[35,31,38,33]
ARP=[0,1,2,1,0+12,2,1,2]   # index into chord (12 -> octave of first)
MEL=[[78,76,74,76],[74,71,74,76],[78,81,78,76],[73,74,76,73]]
def chord_note(c,i): return c[i]+ (0 if i<12 else 0) if i<3 else c[0]+12
def bars(t0,n,lvl):
    """lvl: 0 piano only, 1 +strings+bass, 2 +drums light, 3 +full drums & bells melody"""
    for b in range(n):
        tb=t0+b*BAR; k=b%4; c=CH[k]
        if tb>=DUR: break
        for s in range(8):
            i=ARP[s]; m=(c[i] if i<3 else c[0]+12)
            add(PI,piano(m+(0 if s%4 else 0),2.0,.85+.15*(s%2==0)),tb+s*E,pan=-.25+.5*(s%2))
        add(PI,piano(c[0]-12,3.0,.7),tb,pan=0)
        if lvl>=1:
            add(ST,strings([c[0]-12]+c+[c[1]+12],BAR+.5),tb,g=1.0)
            add(BA,bass(ROOT[k],BEAT*1.5),tb); add(BA,bass(ROOT[k],BEAT*2.4),tb+BEAT*1.5,g=.8)
        if lvl>=2:
            for q in range(4):
                if q in (0,2) or (lvl>=3): add(DR,kick(.8 if lvl<3 else 1.0),tb+q*BEAT)
                if q in (1,3): add(DR,clap() if lvl>=3 else snap(),tb+q*BEAT,g=.7)
            for s16 in range(16): add(DR,shaker(),tb+s16*BEAT/4,g=(.10 if s16%2 else .05)*(1 if lvl>=3 else .7),pan=.3)
        if lvl>=3:
            for j,m in enumerate(MEL[k]): add(PI,bell(m,1.8),tb+j*BEAT,pan=.15,g=.75)
# ---- arrangement (scene times from the video)
FEAT=77.7; NR=84.1; CTA=92.1
add(ST,strings([47,59,62,66],6.6,1800,att=3.0),0,g=1.1)
for j,m in enumerate([71,74,78,83,78,74]): add(PI,piano(m,3.0,.7),0.6+j*E*1.5,pan=-.3+.12*j)
add(FX,swell(2.9),0.4,g=.4); add(FX,impact(),3.3,g=.8); add(FX,cymbal(),3.3,g=.6)
add(PI,piano(47,4,.9),3.3); add(PI,piano(59,4,.8),3.3); add(PI,piano(66,4,.7),3.3)
add(FX,swell(2.6),3.8,g=.35)
t=6.4
bars(t,4,1); t+=4*BAR                     # 6.4 - 14.4  piano+strings
nb=int((15.35-t)//BAR); bars(t,max(nb,0),2); t+=max(nb,0)*BAR
# rain: soft breakdown (piano + strings) over rain ambience until the pool
add(FX,thunder(),15.35,g=.6); add(FX,rain_amb(9.6),15.0,g=.30,pan=-.3); add(FX,rain_amb(9.6),15.0,g=.30,pan=.3)
nr=int((23.6-t)//BAR); bars(t,nr,1); t+=nr*BAR
add(FX,swell(23.6-t+.3),t-.3,g=.3); t=23.6
add(FX,cymbal(),t,g=.7)
nm=int((FEAT-t)//BAR); bars(t,nm,3); t+=nm*BAR
add(FX,cymbal(),t,g=.5)
bars(t,int((NR-t)//BAR)+1,2); 
# rain note: gentle again
add(FX,rain_amb(5.0),NR,g=.22,pan=-.3); add(FX,rain_amb(5.0),NR,g=.22,pan=.3)
t2=NR; bars(t2,int((CTA-1.2-t2)//BAR),1)
add(FX,swell(2.4),CTA-2.4,g=.45)
add(FX,impact(),CTA,g=.7); add(FX,cymbal(4),CTA,g=.7)
nf=int((DUR-3.5-CTA)//BAR); bars(CTA,nf,3); tE=CTA+nf*BAR
add(ST,strings([47,59,62,66,71],4.0,2200,att=.05),tE,g=1.2); add(PI,piano(47,4),tE); add(PI,piano(59,4),tE); add(PI,piano(66,4),tE); add(PI,bell(83,3.5),tE)
add(FX,cymbal(4),tE,g=.5)
for x in [9.8,19.9,26.3,31.3,34.7,39.5,42.7,45.0,49.6,52.8,56.8,61.4,64.3,70.3,73.9,88.1]:
    add(FX,whoosh(),x-.45,g=.10,pan=rng.uniform(-.5,.5))
# ---- mix
def conv(x,h):
    n=len(x)+len(h); F=1<<int(np.ceil(np.log2(n))); return np.fft.irfft(np.fft.rfft(x,F)*np.fft.rfft(h,F),F)[:len(x)]
ti=tt(2.6)
def ir(seed): return filt(np.random.default_rng(seed).uniform(-1,1,len(ti))*np.exp(-ti*2.4),6500)*.05
wet_in=PI*.6+ST*.5+DR*.12+FX*.3
wet=np.stack([conv(wet_in[:,0],ir(1)),conv(wet_in[:,1],ir(2))],1)
mix=PI*1.0+ST*.8+BA*.9+DR*.75+FX*.8+wet*.9
for c in range(2): mix[:,c]=filt(mix[:,c],30,'high')
m=np.ones(N); fo=int(1.5*SR); m[-fo:]=np.linspace(1,0,fo)**1.5; mix*=m[:,None]
mix/=np.percentile(np.abs(mix),99.95)+1e-9; mix=np.tanh(mix*1.05)/np.tanh(1.05)*.9
with wave.open(sys.argv[1],'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(mix,-1,1)*32767).astype('<i2').tobytes())
print('ok',round(tE,2))
