import asyncio, pathlib, json
from playwright.async_api import async_playwright
D=pathlib.Path("/home/atharv/Desktop/projects/Lexsi_Hackathon/actionproof/docs/submission/deck/project")
H=pathlib.Path(__file__).parent
FONT='<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">'
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(channel="chrome")
        pg=await b.new_page(viewport={"width":1920,"height":1080},device_scale_factor=2)
        boxes={}
        css=(D/"deck.css").read_text()+(H/"cards/extra.css").read_text()
        files=[(f,"slide_") for f in (D/"slides").glob("*.html") if f.stem in ("hook","naive","training","fails")]+[(f,"card_") for f in (H/"cards").glob("*.html")]
        for f,pre in files:
            await pg.set_content(f"<!DOCTYPE html><html><head>{FONT}<style>{css} body{{margin:0}} section{{width:1920px;height:1080px;position:relative;overflow:hidden}} aside{{display:none}}</style></head><body>{f.read_text()}</body></html>")
            await pg.wait_for_timeout(1500)
            await pg.screenshot(path=str(H/"shots"/f"{pre}{f.stem}.png"))
            if pre=="card_": boxes[f.stem]=await pg.evaluate("()=>Object.fromEntries([...document.querySelectorAll('[id]')].map(e=>{const r=e.getBoundingClientRect();return [e.id,[r.x*2,r.y*2,(r.x+r.width)*2,(r.y+r.height)*2]]}))")
        json.dump(boxes,open(H/"shots/card_boxes.json","w"))
        await b.close()
asyncio.run(main())
