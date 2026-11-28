"""Screen-record one real run of the live Space (L4) via the CDP screencast, following the trained agent's column.
Logs event times (seconds from start) and on-screen boxes of key elements, for the edit."""
import asyncio, base64, json, time, pathlib, sys
from playwright.async_api import async_playwright
H = pathlib.Path(__file__).parent
NAME = sys.argv[1] if len(sys.argv) > 1 else "live"
OUT = H / "rec"; FR = OUT / f"{NAME}_frames"; FR.mkdir(parents=True, exist_ok=True)
URL = "https://atharvgaur18-actionproof-live.hf.space/"
# boxes of visible elements whose text contains a needle (viewport px)
BOX = """(needles)=>{const out={};for(const n of needles){let best=null;for(const e of document.querySelectorAll('body *')){
 const t=(e.innerText||'');if(!t.includes(n)||t.length>600)continue;const r=e.getBoundingClientRect();
 if(r.width<20||r.height<10||r.bottom<0||r.top>innerHeight)continue;if(!best||r.width*r.height<best[4])best=[r.x,r.y,r.right,r.bottom,r.width*r.height];}
 if(best)out[n]=best.slice(0,4);}return out}"""
LAST_TURN = """()=>{const c=[...document.querySelectorAll('body *')].filter(e=>{const t=(e.innerText||'');const r=e.getBoundingClientRect();
 return r.left>900&&r.width>300&&r.width<700&&/^TURN \\d+$/m.test(t.split('\\n')[0]||'')&&t.length<700});
 if(!c.length)return null;const e=c[c.length-1];const r=e.getBoundingClientRect();return r.bottom+scrollY}"""
SCROLL_TO_TEXT = """(n)=>{const els=[...document.querySelectorAll('body *')].filter(e=>(e.innerText||'').trim().startsWith(n));
 if(!els.length)return false;const e=els[els.length-1];const r=e.getBoundingClientRect();
 window.scrollTo({top:r.top+scrollY-140,behavior:'smooth'});return true}"""
async def main():
    ev = []; frames = []; t0 = None
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        ctx = await b.new_context(viewport={"width": 1920, "height": 1080}, color_scheme="light")
        pg = await ctx.new_page()
        cdp = await ctx.new_cdp_session(pg)
        async def on_frame(prm):
            try: await cdp.send("Page.screencastFrameAck", {"sessionId": prm["sessionId"]})
            except Exception: pass
            f = FR / f"{len(frames):06d}.jpg"; frames.append((f.name, prm["metadata"]["timestamp"]))
            f.write_bytes(base64.b64decode(prm["data"]))
        cdp.on("Page.screencastFrame", lambda prm: asyncio.ensure_future(on_frame(prm)))
        t0 = time.time()
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92, "maxWidth": 1920, "maxHeight": 1080, "everyNthFrame": 2})
        async def mark(k, needles=()):
            bx = await pg.evaluate(BOX, list(needles)) if needles else {}
            ev.append({"k": k, "t": round(time.time() - t0, 2), "boxes": bx}); print(k, ev[-1]["t"], list(bx), flush=True)
        await pg.goto(URL, timeout=180000); await pg.wait_for_timeout(7000)
        await mark("ready", ["RUN IT.", "Pick a task"])
        await pg.evaluate("window.scrollTo({top:420,behavior:'smooth'})"); await pg.wait_for_timeout(2500)
        await mark("controls", ["TASK (CURATED", "REALITY (WHAT", "Run"])
        btn = pg.get_by_role("button", name="Run"); await btn.hover(); await pg.wait_for_timeout(700)
        await btn.click(); await mark("run_clicked")
        seen = set(); flags = {"empty": '"payments": [], "as_of": 0', "repay": "TURN 5", "confirmed": '"payments": [{"payment_id"',
                                "journal": "get_journal_entries", "report": "The agent submitted its report.\n"}
        for i in range(300):
            await pg.wait_for_timeout(700)
            body = await pg.inner_text("body")
            y = await pg.evaluate(LAST_TURN)
            if y: await pg.evaluate(f"window.scrollTo({{top:Math.max(0,{y}-880),behavior:'smooth'}})")
            for k, s in flags.items():
                if k not in seen and s in body:
                    seen.add(k); await pg.wait_for_timeout(600)
                    await mark(k, ["list_payments", "pay_supplier", "PAY-BEFAD4", "submit_report", "TURN 4", "TURN 5", "TURN 6"])
            if "Run finished" in body or "Run failed" in body:
                await mark("finished" if "Run finished" in body else "failed"); break
        await pg.wait_for_timeout(1500)
        for txt, hold in (("SCRIPTED: ACKNOWLEDGEMENT-TRUSTING", 5000), ("VERIFIER VERDICT PER STEP", 4500),
                          ("ACKNOWLEDGEMENT AGAINST EFFECT", 4500)):
            await pg.evaluate(SCROLL_TO_TEXT, txt); await pg.wait_for_timeout(1300)
            await mark("at:" + txt, ["VSC 0", "VSC 1", "False success", "Goal met", "PAYMENTS (PO)", "VERIFIED_SUCCESS", "pay_supplier", "none", "Reality C was"])
            await pg.wait_for_timeout(hold - 1300)
        (OUT / f"{NAME}_body.txt").write_text(await pg.inner_text("body"))
        await pg.screenshot(path=str(OUT / f"{NAME}_full.png"), full_page=True)
        json.dump(await pg.evaluate(BOX.replace("r.bottom<0||r.top>innerHeight","false").replace("r.x,r.y,r.right,r.bottom","r.x+scrollX,r.y+scrollY,r.right+scrollX,r.bottom+scrollY"),
                  ["SCRIPTED: ACKNOWLEDGEMENT-TRUSTING","TRAINED AGENT (LIVE)","VSC 0","VSC 1","PAYMENTS (PO)","VERIFIER VERDICT PER STEP","ACKNOWLEDGEMENT AGAINST EFFECT, PER WRITE","Reality C was"]),
                  open(OUT / f"{NAME}_fullboxes.json", "w"))
        await cdp.send("Page.stopScreencast"); await pg.wait_for_timeout(400)
        await ctx.close(); await b.close()
    fr = [(n, round(ts - t0, 3)) for n, ts in frames]
    json.dump({"events": ev, "frames": fr}, open(OUT / f"{NAME}_events.json", "w"), indent=1)
    print(len(fr), "frames")
asyncio.run(main())
