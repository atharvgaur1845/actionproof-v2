"""Render the committed deck (deck.json order) to one 16:9 PDF with Chrome."""
import asyncio, json, pathlib
from playwright.async_api import async_playwright
D = pathlib.Path("/home/atharv/Desktop/projects/Lexsi_Hackathon/actionproof/docs/submission/deck/project")
OUT = D.parent.parent / "ActionProof_deck.pdf"
FONT = '<link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">'
async def main():
    order = json.loads((D / "deck.json").read_text())["order"]
    css = (D / "deck.css").read_text()
    body = "".join((D / "slides" / f"{s}.html").read_text() for s in order)
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>{FONT}<style>{css}"
            "@page{size:1920px 1080px;margin:0} body{margin:0} "
            "section{width:1920px;height:1080px;position:relative;overflow:hidden;page-break-after:always;break-after:page} "
            "aside{display:none}</style></head><body>" + body + "</body></html>")
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        await pg.set_content(html)
        await pg.wait_for_timeout(2500)
        await pg.pdf(path=str(OUT), width="1920px", height="1080px", print_background=True)
        await b.close()
    print(OUT, len(order))
asyncio.run(main())
