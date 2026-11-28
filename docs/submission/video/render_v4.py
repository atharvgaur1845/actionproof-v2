"""Render the v4 video cards (with inlined figures) and the theory slide; merge their boxes into card_boxes.json."""
import asyncio, base64, json, pathlib
from playwright.async_api import async_playwright
D = pathlib.Path("/home/atharv/Desktop/projects/Lexsi_Hackathon/actionproof/docs/submission/deck/project")
H = pathlib.Path(__file__).parent
FONT = '<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">'
FIG = {k: "data:image/png;base64," + base64.b64encode((H / "figs" / f"{k}.png").read_bytes()).decode() for k in ("fig_c1_seeds", "fig_th1_laws")}
JS = "()=>Object.fromEntries([...document.querySelectorAll('[id]')].map(e=>{const r=e.getBoundingClientRect();return [e.id,[r.x*2,r.y*2,(r.x+r.width)*2,(r.y+r.height)*2]]}))"
async def main():
    boxes = json.load(open(H / "shots/card_boxes.json"))
    css = (D / "deck.css").read_text() + (H / "cards/extra.css").read_text()
    jobs = [(D / "slides/theory.html", "slide_theory")] + [(H / f"cards/{c}.html", f"card_{c}") for c in ("vote", "th1", "c1", "end", "proof", "c2") if (H / f"cards/{c}.html").exists()]
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        pg = await b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=2)
        for f, name in jobs:
            html = f.read_text()
            for k, v in FIG.items(): html = html.replace("{{" + k + "}}", v)
            await pg.set_content(f"<!DOCTYPE html><html><head><meta charset='utf-8'>{FONT}<style>{css} body{{margin:0}} section{{width:1920px;height:1080px;position:relative;overflow:hidden}} aside{{display:none}}</style></head><body>{html}</body></html>")
            await pg.wait_for_timeout(1500)
            await pg.screenshot(path=str(H / "shots" / f"{name}.png"))
            boxes[name.split("_", 1)[1]] = await pg.evaluate(JS)
            print(name)
        await b.close()
    json.dump(boxes, open(H / "shots/card_boxes.json", "w"))
asyncio.run(main())
