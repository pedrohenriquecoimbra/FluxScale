"""The FluxScale page: load EddyPro raw datasets, decompose, look at covariances.

``fluxscale gui`` serves the page from this computer and runs fluxscale itself
behind it. The same page is published online, where it runs fluxscale inside
the browser instead (see scripts/build_site.py). Both call `tool`.
"""
# built-in modules
import argparse
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
EXPORTS = {'csv': 'text/csv', 'netcdf': 'application/x-netcdf'}

__all__ = ['serve', 'cli']


class _Session:
    """The files, the work folder and the job of one `fluxscale gui`."""

    def __init__(self, input_path=None):
        self.root = tempfile.mkdtemp(prefix='fluxscale_gui_')
        self.uploads = os.path.join(self.root, 'input')
        self.work = os.path.join(self.root, 'work')
        os.makedirs(self.uploads)
        self.preloaded = os.path.abspath(input_path) if input_path else None
        self.folder = self.preloaded or self.uploads
        self.lock = threading.Lock()
        self.job = dict(state='idle', text='', done=0, total=0)

    def clear(self):
        shutil.rmtree(self.uploads, ignore_errors=True)
        os.makedirs(self.uploads)
        self.folder = self.uploads

    def notify(self, text, done, total):
        self.job.update(text=text, done=done, total=total)

    def start(self, options):
        from . import tool
        with self.lock:
            if self.job['state'] == 'running':
                raise RuntimeError('A run is already going on.')
            self.job = dict(state='running', text='Starting', done=0, total=0)

        def work():
            tool.set_notify(self.notify)
            try:
                result = tool.run(self.folder, self.work, options)
                self.job = dict(state='done', result=result)
            except Exception as error:      # the page shows it
                self.job = dict(state='error', text=str(error)[-600:])

        threading.Thread(target=work, daemon=True).start()

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def _handler(session, hosts):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, kind='application/json', status=200, name=None):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            if name:
                self.send_header('Content-Disposition', f'attachment; filename="{name}"')
            self.end_headers()
            self.wfile.write(body)

        def _local(self):
            # a page from elsewhere must not reach this server through the browser
            origin = self.headers.get('Origin')
            if self.headers.get('Host') in hosts and (origin is None or urlparse(origin).netloc in hosts):
                return True
            self._send(dict(error='not from this computer'), status=403)
            return False

        def _body(self):
            return self.rfile.read(int(self.headers.get('Content-Length') or 0))

        def do_GET(self):
            if not self._local():
                return
            url = urlparse(self.path)
            try:
                if url.path == '/api/info':
                    from . import tool
                    self._send(dict(version=tool.__version__, folder=session.preloaded))
                elif url.path == '/api/status':
                    self._send(session.job)
                elif url.path == '/api/export':
                    from . import tool
                    kind = parse_qs(url.query).get('kind', [''])[0]
                    path = tool.write_export(session.work, kind)
                    with open(path, 'rb') as file:
                        self._send(file.read(), EXPORTS[kind], name=os.path.basename(path))
                else:
                    self._static(url.path)
            except Exception as error:
                self._send(dict(error=str(error)[-600:]), status=500)

        def _static(self, path):
            name = os.path.normpath(path.lstrip('/') or 'index.html')
            full = os.path.join(WEB, name)
            if name.startswith('..') or os.path.isabs(name) or not os.path.isfile(full):
                self._send(dict(error='not found'), status=404)
                return
            with open(full, 'rb') as file:
                body = file.read()
            if name == 'index.html':    # tells the page that fluxscale is behind it (c.f. web/app.js)
                body = body.replace(b'<html lang="en">', b'<html lang="en" data-engine="local">', 1)
            self._send(body, mimetypes.guess_type(full)[0] or 'application/octet-stream')

        def do_POST(self):
            if not self._local():
                return
            url = urlparse(self.path)
            try:
                if url.path == '/api/clear':
                    self._body()
                    session.clear()
                    self._send({})
                elif url.path == '/api/file':
                    name = os.path.basename(parse_qs(url.query).get('name', [''])[0])
                    if not name or name.startswith('.'):
                        raise ValueError('a file needs a name')
                    remaining = int(self.headers.get('Content-Length') or 0)
                    with open(os.path.join(session.uploads, name), 'wb') as file:
                        while remaining:
                            chunk = self.rfile.read(min(remaining, 1 << 20))
                            if not chunk:
                                break
                            file.write(chunk)
                            remaining -= len(chunk)
                    self._send({})
                elif url.path == '/api/scan':
                    from . import tool
                    self._body()
                    self._send(tool.scan(session.folder))
                elif url.path == '/api/run':
                    session.start(json.loads(self._body() or b'{}'))
                    self._send({})
                else:
                    self._send(dict(error='not found'), status=404)
            except Exception as error:
                self._send(dict(error=str(error)[-600:]), status=500)

    return Handler


def serve(input_path=None, port=0, open_browser=True):
    """
    function: serve the FluxScale page from this computer until interrupted.
        It only answers this computer
    call: serve()
    Input:
        input_path: folder of EddyPro raw datasets to open the page with;
            without it, files are chosen in the page
        port: port to listen on, 0 for any free one
        open_browser: open the page in the default browser
    """
    if input_path and not os.path.isdir(input_path):
        raise NotADirectoryError(f"{input_path!r} is not a folder.")
    session = _Session(input_path)
    hosts = set()
    server = ThreadingHTTPServer(('127.0.0.1', port), _handler(session, hosts))
    port = server.server_address[1]
    hosts.update({f'127.0.0.1:{port}', f'localhost:{port}'})
    url = f'http://127.0.0.1:{port}/'
    print(f'FluxScale is at {url}  (Ctrl+C to stop)', flush=True)
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()


def cli(argv=None):
    """`fluxscale gui [-i FOLDER] [--port N] [--no-browser]`"""
    parser = argparse.ArgumentParser(
        prog='fluxscale gui',
        description='Open the FluxScale page on this computer: load EddyPro raw datasets '
                    '(level 6), decompose them and look at covariances and partitions.')
    parser.add_argument('-i', '--input_path', type=str, default=None,
                        help='Folder of raw datasets to start with (default: choose files in the page)')
    parser.add_argument('--port', type=int, default=0, help='Port to listen on (default: any free one)')
    parser.add_argument('--no-browser', dest='open_browser', action='store_false',
                        help='Do not open the browser; the address is printed')
    args = parser.parse_args(argv)
    serve(args.input_path, args.port, args.open_browser)
