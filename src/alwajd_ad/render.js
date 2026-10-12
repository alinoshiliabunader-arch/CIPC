const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const path=require('path'), fs=require('fs');
const [,, mode, a, b, workersArg] = process.argv; // mode: preview "t1,t2,.." | frames start end
const FPS=30, OUT=path.join(__dirname,'out'); fs.mkdirSync(OUT,{recursive:true});
(async()=>{
  const browser=await chromium.launch({args:['--allow-file-access-from-files','--disable-web-security','--force-color-profile=srgb']});
  async function page(){const p=await browser.newPage({viewport:{width:1080,height:1920},deviceScaleFactor:1});
    p.on('console',m=>console.log('PAGE',m.text())); p.on('pageerror',e=>console.log('ERR',e.message));
    await p.goto('file://'+path.join(__dirname,'index.html')); await p.evaluate(()=>window.ready); return p;}
  if(mode==='preview'){const p=await page();
    for(const t of a.split(',').map(Number)){await p.evaluate(t=>window.renderAt(t),t);
      await p.screenshot({path:path.join(OUT,`prev_${t}.jpg`),type:'jpeg',quality:80});}
  } else {
    const s=+a,e=+b,W=+(workersArg||4); const fd=path.join(OUT,'frames'); fs.mkdirSync(fd,{recursive:true});
    let next=s; const t0=Date.now();
    await Promise.all([...Array(W)].map(async()=>{const p=await page();
      while(true){const f=next++; if(f>=e)break;
        await p.evaluate(t=>window.renderAt(t),f/FPS);
        await p.screenshot({path:path.join(fd,String(f).padStart(5,'0')+'.jpg'),type:'jpeg',quality:94});
        if(f%150===0)console.log('frame',f,((Date.now()-t0)/1000).toFixed(0)+'s');}}));
  }
  await browser.close();
})();
