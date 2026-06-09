from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "CONCLUSION_REVIEW_STATE.json"
HTML_PATH = ROOT / "CONCLUSION_REVIEW.html"


class Handler(BaseHTTPRequestHandler):
    def _send(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._send()

    def do_GET(self):
        if self.path in ("/", "/CONCLUSION_REVIEW.html"):
            self._send(200, "text/html; charset=utf-8")
            self.wfile.write(HTML_PATH.read_bytes())
            return

        if self.path == "/state":
            data = STATE_PATH.read_text(encoding="utf-8") if STATE_PATH.exists() else "{}"
            self._send(200, "application/json; charset=utf-8")
            self.wfile.write(data.encode("utf-8"))
            return

        target = ROOT / self.path.lstrip("/")
        if target.exists() and target.is_file():
            self._send(200, mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.wfile.write(target.read_bytes())
            return

        self._send(404)
        self.wfile.write(b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/save":
            self._send(404)
            self.wfile.write(b'{"error":"not found"}')
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send(400)
            self.wfile.write(b'{"error":"invalid json"}')
            return

        STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._send(200)
        self.wfile.write(b'{"ok":true}')


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8771), Handler)
    print("Conclusion review server running at http://127.0.0.1:8771/CONCLUSION_REVIEW.html")
    server.serve_forever()
