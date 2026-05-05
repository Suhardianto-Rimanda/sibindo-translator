import os
import sys

from app import create_app

app = create_app()

if __name__ == "__main__":
    debug = app.config["DEBUG"]
    host = os.getenv("FLASK_HOST", "127.0.0.1" if debug else "0.0.0.0")
    if debug and host not in ("127.0.0.1", "localhost", "::1"):
        sys.exit(
            "Refusing to start: FLASK_DEBUG=1 with non-loopback host "
            f"({host!r}) exposes the Werkzeug debugger and is an RCE risk. "
            "Either set FLASK_DEBUG=0 or FLASK_HOST=127.0.0.1."
        )
    app.run(host=host, port=int(os.getenv("PORT", "5000")), debug=debug)
