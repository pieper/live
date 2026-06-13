#!/usr/bin/env python3
"""SlicerLive one-command publish.

Exports the scene currently loaded in your running Slicer (the MCP server on :2027), uploads its bulk data
to the JS2 Ceph bucket, writes the small scene.json + a thumbnail + a gallery entry into this repo, and
commits + pushes — so it shows up at https://pieper.github.io/live .

  python3 publish.py <SceneName> ["short description"]

Requires: your Slicer running the MCP server on localhost:2027 with the scene loaded; `openstack` configured
with the CIS230102_IU cloud (clouds.yaml); `gh`/`git` auth for pushing this repo.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

LIVE = os.path.dirname(os.path.abspath(__file__))
SCENES = os.path.join(LIVE, "scenes")
BUCKET_BASE = "https://js2.jetstream-cloud.org:8001/swift/v1/slicerlive"
CLOUD = "CIS230102_IU"
CONTAINER = "slicerlive"
MCP = "http://localhost:2027/mcp"
MRML_SYNC_DIR = "/Users/pieper/slicer/desktopia/offload/spike"   # where mrml_sync.py lives (the serializer)


def mcp(code):
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": "execute_python", "arguments": {"code": code}}}
    r = urllib.request.urlopen(urllib.request.Request(
        MCP, data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=300)
    return json.load(r)["result"]["content"][0]["text"]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: publish.py <SceneName> [\"short description\"]")
    name = sys.argv[1]
    desc = sys.argv[2] if len(sys.argv) > 2 else ""
    os.makedirs(SCENES, exist_ok=True)
    tmp = tempfile.mkdtemp()
    blobdir = os.path.join(tmp, "blobs")
    os.makedirs(blobdir)

    # 1) export the live scene (node-state JSON + content-addressed blobs) + a thumbnail, written to the Mac fs
    print("exporting the current Slicer scene (:2027) …")
    code = f'''
import slicer, sys, json, os, importlib
sys.path.insert(0, {MRML_SYNC_DIR!r})
import mrml_sync; importlib.reload(mrml_sync)
state = mrml_sync.mrml_state(0)
hashes = set()
def collect(o):
    if isinstance(o, dict):
        h = o.get("hash")
        if isinstance(h, str): hashes.add(h)
        for x in o.values(): collect(x)
    elif isinstance(o, list):
        for x in o: collect(x)
collect(state)
open({os.path.join(tmp, "scene_nodes.json")!r}, "w").write(json.dumps(state))
nb = 0
for h in hashes:
    b = mrml_sync.get_blob(h)
    if b:
        open(os.path.join({blobdir!r}, h), "wb").write(b); nb += 1
try:
    tdv = slicer.app.layoutManager().threeDWidget(0).threeDView(); tdv.forceRender()
    tdv.grab().save({os.path.join(tmp, "thumb.png")!r})
except Exception:
    slicer.util.mainWindow().grab().save({os.path.join(tmp, "thumb.png")!r})
__result = json.dumps({{"nodes": len(state), "blobs": nb}})
'''
    print("  ", mcp(code))

    # 2) upload the blobs to the CORS+public bucket (container ACL makes new objects public automatically)
    blobs = glob.glob(os.path.join(blobdir, "*"))
    print(f"uploading {len(blobs)} blobs -> {CLOUD}:{CONTAINER}/{name}/blobs/ …")
    for f in blobs:
        subprocess.run(["openstack", "--os-cloud", CLOUD, "object", "create", CONTAINER, f,
                        "--name", f"{name}/blobs/{os.path.basename(f)}"], check=True, stdout=subprocess.DEVNULL)

    # 3) write the small scene wrapper (blobBase -> bucket) + thumbnail into the repo
    wrapper = {"blobBase": f"{BUCKET_BASE}/{name}/blobs/",
               "nodes": json.load(open(os.path.join(tmp, "scene_nodes.json")))}
    json.dump(wrapper, open(os.path.join(SCENES, f"{name}.json"), "w"))
    shutil.copy(os.path.join(tmp, "thumb.png"), os.path.join(SCENES, f"{name}.png"))

    # 4) update the gallery index
    idxpath = os.path.join(SCENES, "index.json")
    idx = json.load(open(idxpath)) if os.path.exists(idxpath) else []
    idx = [s for s in idx if s.get("name") != name]
    idx.append({"name": name, "scene": f"{name}.json", "thumb": f"{name}.png", "desc": desc})
    json.dump(idx, open(idxpath, "w"), indent=1)

    # 5) commit + push
    subprocess.run(["git", "-C", LIVE, "add", "-A"], check=True)
    subprocess.run(["git", "-C", LIVE, "commit", "-q", "-m", f"publish scene: {name}"], check=True)
    subprocess.run(["git", "-C", LIVE, "push", "-q", "origin", "main"], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n published: https://pieper.github.io/live/viewer.html?scene=scenes/{name}.json")


if __name__ == "__main__":
    main()
