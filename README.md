# SlicerLive demo gallery → https://pieper.github.io/live

A GitHub Pages site that hosts live, in-browser [SlicerLive](https://github.com/pieper/SlicerLive) scenes.
The gallery (`index.html`) shows a thumbnail per scene; clicking one opens it in the viewer in a new tab and
renders it on your GPU — no Slicer install, no server.

**Split hosting:** the small scene description (`scenes/<name>.json`, a `{blobBase, nodes}` wrapper) lives in
this repo / Pages; the bulk data (content-addressed gzip blobs) lives in a public, CORS-enabled Ceph bucket on
Jetstream2 (`…/swift/v1/slicerlive/<name>/blobs/`). The viewer reads the scene from Pages and the blobs from
the bucket.

## Publish a scene from your Slicer (one command)
With your scene loaded in Slicer (running the MCP server on `localhost:2027`) and `openstack`/`gh` configured:
```bash
python3 publish.py "MyScene" "a short description"
```
That exports the scene (node-state JSON + blobs) via the offload serializer, uploads the blobs to the JS2
bucket, writes `scenes/MyScene.json` + a thumbnail + a gallery entry, and commits + pushes.

## Layout
- `index.html` — the thumbnail gallery.
- `viewer.html` + `slicerlive-bundle.js` — the SlicerLive viewer (committed so Pages can serve it).
- `scenes/` — `<name>.json` (scene wrapper) + `<name>.png` (thumbnail) + `index.json` (gallery manifest).
- `publish.py` — the one-command publisher.
