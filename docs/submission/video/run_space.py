import asyncio, sys, json
from playwright.async_api import async_playwright
mode=sys.argv[1]
JS="""()=>{const out=[];for(const e of document.querySelectorAll('body *')){const t=(e.innerText||'').trim();if(!t||t.length>300)continue;const r=e.getBoundingClientRect();if(r.width<5||r.height<5)continue;out.push({tag:e.tagName,t:t.slice(0,100).split(String.fromCharCode(10)).join(' | '),b:[(r.x+scrollX)*2,(r.y+scrollY)*2,(r.x+scrollX+r.width)*2,(r.y+scrollY+r.height)*2]})}return out}"""
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(channel="chrome")
        pg=await b.new_page(viewport={"width":1920,"height":1080},device_scale_factor=2)
        await pg.goto("https://atharvgaur18-actionproof-live.hf.space/",timeout=120000)
        await pg.wait_for_timeout(6000)
        if mode=="gate":
            await pg.locator("input[role=listbox], input[aria-label='Agent']").last.click(); await pg.wait_for_timeout(800)
            await pg.locator("[role=option]", has_text="Gate demo").first.click(); await pg.wait_for_timeout(800)
        await pg.get_by_role("button",name="Run").click()
        await pg.wait_for_timeout(3000)
        for i in range(80):
            await pg.wait_for_timeout(5000)
            if "Run finished" in await pg.inner_text("body"): break
        await pg.wait_for_timeout(3000)
        await pg.screenshot(path=f"shots/{mode}_full.png",full_page=True)
        json.dump(await pg.evaluate(JS),open(f"shots/{mode}_boxes.json","w"))
        await b.close()
asyncio.run(main())
