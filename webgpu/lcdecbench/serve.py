"""Static server for /Users/pieper/slicer/live/webgpu on :8099 with a POST /report
endpoint that appends JSON lines to reports.jsonl in the scratchpad."""
import http.server
import json
import os
import sys

ROOT = "/Users/pieper/slicer/live/webgpu"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports.jsonl")


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_POST(self):
        if self.path.startswith("/report"):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode("utf-8", "replace")
            with open(OUT, "a") as f:
                f.write(body.strip() + "\n")
            print("REPORT:", body[:400], flush=True)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
