import asyncio, json, edge_tts, sys
sys.path.insert(0,'.'); from script import SEGMENTS
async def one(seg):
    c=edge_tts.Communicate(seg["text"],"en-US-AndrewNeural",rate="+6%",boundary="WordBoundary")
    words=[]; audio=bytearray()
    async for ch in c.stream():
        if ch["type"]=="audio": audio+=ch["data"]
        elif ch["type"]=="WordBoundary": words.append((ch["offset"]/1e7,(ch["offset"]+ch["duration"])/1e7,ch["text"]))
    open(f"audio/{seg['id']}.mp3","wb").write(audio); return seg["id"],words
async def main():
    out=dict(await asyncio.gather(*[one(s) for s in SEGMENTS]))
    json.dump(out,open("audio/words.json","w")); print({k:round(v[-1][1],1) for k,v in out.items()})
asyncio.run(main())
