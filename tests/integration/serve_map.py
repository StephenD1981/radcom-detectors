"""
Simple HTTP server to view the overshooting map.

Run this script and open http://localhost:8000 in your browser.
The lazy-loaded grid files will work correctly via HTTP.
"""
import http.server
import socketserver
from pathlib import Path
import webbrowser
import time
import threading

PORT = 8888
DIRECTORY = Path("data/output-data/vf-ie/recommendations/maps").resolve()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def log_message(self, format, *args):
        # Suppress verbose logging, only show important messages
        if "GET" in format and (".json" in args[0] or ".html" in args[0]):
            print(f"[Server] {args[0]}")


def open_browser():
    """Open browser after short delay to allow server to start."""
    time.sleep(1)
    url = f"http://localhost:{PORT}/overshooting_cells_map.html"
    print(f"\n🌐 Opening browser to: {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    print("="*80)
    print("OVERSHOOTING MAP SERVER")
    print("="*80)
    print(f"\n📁 Serving directory: {DIRECTORY}")
    print(f"🌐 Server URL: http://localhost:{PORT}")
    print(f"📄 Map URL: http://localhost:{PORT}/overshooting_cells_map.html")
    print(f"\n✅ Server starting...")
    print(f"   Press Ctrl+C to stop\n")

    # Check if map file exists
    map_file = DIRECTORY / "overshooting_cells_map.html"
    if not map_file.exists():
        print(f"❌ Error: Map file not found at {map_file}")
        print(f"   Please run test_visualize_overshooters.py first")
        exit(1)

    # Start browser in background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    # Start server
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"✅ Server running at http://localhost:{PORT}")
            print(f"   Browser should open automatically...")
            print(f"\n💡 Usage:")
            print(f"   • Click cells to load grids on-demand")
            print(f"   • Grid data loads via AJAX (lazy loading)")
            print(f"   • Press Ctrl+C to stop server\n")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n\n✅ Server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ Error: Port {PORT} is already in use")
            print(f"   Try: lsof -ti:{PORT} | xargs kill")
        else:
            raise
