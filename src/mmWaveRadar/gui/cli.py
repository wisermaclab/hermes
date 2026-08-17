# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Command-line launcher and environment check for the local HERMES GUI."""

from __future__ import annotations

import argparse
from importlib import metadata
import hmac
import ipaddress
import os
from pathlib import Path
import re
import secrets
import sys


_MAX_HTTP_HEADER_BYTES = 16 * 1024
_MAX_HTTP_BODY_BYTES = 64 * 1024
_WEBSOCKET_PROTOCOL_OVERHEAD_BYTES = 1024 * 1024


def _websocket_message_limit(max_upload_bytes: int) -> int:
    """Size a WebSocket message for one base64-encoded FileInput upload."""

    encoded_bytes = 4 * ((int(max_upload_bytes) + 2) // 3)
    return encoded_bytes + _WEBSOCKET_PROTOCOL_OVERHEAD_BYTES


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _default_smpl_model_dir() -> Path:
    """Resolve the GUI's licensed-model convention for a source checkout."""

    configured = os.environ.get("MMWAVE_SMPL_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    public_root = Path(__file__).resolve().parents[3]
    return public_root / "models" / "smpl_models"


def _is_loopback_address(address: str) -> bool:
    """Return whether *address* is an explicit local-only bind target."""

    candidate = str(address).strip()
    if candidate.casefold() == "localhost":
        return True
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # Hostnames other than the explicit ``localhost`` spelling can change
        # resolution and are therefore not treated as a safe local bind.
        return False


def _is_unspecified_address(address: str) -> bool:
    """Return whether *address* is an IPv4 or IPv6 wildcard bind."""

    candidate = str(address).strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return ipaddress.ip_address(candidate).is_unspecified
    except ValueError:
        return False


def _origin_host(host: str) -> str:
    """Validate and normalize the host portion of a websocket origin."""

    if not host or host == "*" or "%" in host:
        raise ValueError("websocket origin must name one concrete host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if len(host) > 253 or not all(
            re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
            for label in host.rstrip(".").split(".")
        ):
            raise ValueError("websocket origin contains an invalid hostname")
        return host.casefold()
    if address.is_unspecified:
        raise ValueError("websocket origin must not be a wildcard address")
    return f"[{address.compressed}]" if address.version == 6 else address.compressed


def _normalize_websocket_origin(value: str, *, default_port: int) -> str:
    """Return a validated Panel ``HOST:PORT`` websocket allowlist entry."""

    candidate = str(value).strip()
    if (
        not candidate
        or any(character.isspace() for character in candidate)
        or "://" in candidate
        or any(character in candidate for character in "/?#@")
    ):
        raise ValueError(
            "websocket origin must use HOST[:PORT] without a scheme or path"
        )

    port = int(default_port)
    if candidate.startswith("["):
        closing = candidate.find("]")
        if closing < 0:
            raise ValueError("IPv6 websocket origins must close their brackets")
        host = candidate[1:closing]
        remainder = candidate[closing + 1 :]
        if remainder:
            if not remainder.startswith(":") or not remainder[1:].isdigit():
                raise ValueError("websocket origin port must be an integer")
            port = int(remainder[1:])
        normalized_host = _origin_host(host)
        if not normalized_host.startswith("["):
            raise ValueError("brackets are only valid around an IPv6 address")
    elif candidate.count(":") >= 2:
        normalized_host = _origin_host(candidate)
    elif ":" in candidate:
        host, port_text = candidate.rsplit(":", maxsplit=1)
        if not port_text.isdigit():
            raise ValueError("websocket origin port must be an integer")
        port = int(port_text)
        normalized_host = _origin_host(host)
    else:
        normalized_host = _origin_host(candidate)
    if not 1 <= port <= 65_535:
        raise ValueError("websocket origin port must be between 1 and 65535")
    return f"{normalized_host}:{port}"


def _set_browser_security_headers(handler) -> None:
    """Apply anti-framing and token-leakage headers to GUI HTML responses."""

    # Tornado replaces response-mutating methods such as ``set_header`` after
    # a WebSocket connection has detached from HTTP. Bokeh still evaluates
    # the authentication provider for ``/ws``, but these document-only headers
    # are neither supported nor useful on the established WebSocket handler.
    from tornado.websocket import WebSocketHandler

    if isinstance(handler, WebSocketHandler):
        return
    handler.set_header("Content-Security-Policy", "frame-ancestors 'none'")
    handler.set_header("X-Frame-Options", "DENY")
    handler.set_header("Referrer-Policy", "no-referrer")
    handler.set_header("Cache-Control", "no-store")


def _build_launch_auth_provider(password: str):
    """Build per-launch authentication checked before Bokeh creates a session."""

    from bokeh.server.auth_provider import AuthProvider
    from tornado.web import RequestHandler

    cookie_name = "hermes_gui_user"
    expected_cookie = b"hermes"

    class HermesLoginHandler(RequestHandler):
        def set_default_headers(self) -> None:
            _set_browser_security_headers(self)

        def get(self) -> None:
            self.set_header("Content-Type", "text/html; charset=UTF-8")
            self.write(
                "<!doctype html><html><head><title>HERMES access</title></head>"
                "<body><main><h1>HERMES</h1><p>Enter the per-launch password "
                "printed by the server.</p><form method='post' "
                "action='/__hermes_login'><label>Password "
                "<input name='password' type='password' required "
                "autocomplete='current-password'></label> "
                "<button type='submit'>Open HERMES</button></form></main>"
                "</body></html>"
            )

        def post(self) -> None:
            # Fetch Metadata is checked before accepting a credential so a
            # hostile origin cannot drive the login endpoint. Current GUI
            # browsers send this header; missing metadata is fail-closed.
            if self.request.headers.get("Sec-Fetch-Site", "").casefold() != (
                "same-origin"
            ):
                self.set_status(403)
                self.write("Cross-site login requests are not permitted.")
                return
            supplied = self.get_body_argument("password", default="")
            if not hmac.compare_digest(supplied, password):
                self.set_status(403)
                self.write("Invalid HERMES launch password.")
                return
            self.set_secure_cookie(
                cookie_name,
                expected_cookie,
                httponly=True,
                samesite="Strict",
                path="/",
            )
            self.redirect("/")

    class HermesLaunchAuthProvider(AuthProvider):
        @property
        def get_user(self):
            def authenticated_user(request_handler):
                _set_browser_security_headers(request_handler)
                fetch_site = request_handler.request.headers.get(
                    "Sec-Fetch-Site",
                    "",
                ).casefold()
                if fetch_site not in {"none", "same-origin"}:
                    return None
                user = request_handler.get_secure_cookie(
                    cookie_name,
                    max_age_days=1,
                )
                if user is None or not hmac.compare_digest(
                    user,
                    expected_cookie,
                ):
                    return None
                return "hermes"

            return authenticated_user

        @property
        def login_url(self) -> str:
            return "/__hermes_login"

        @property
        def login_handler(self):
            return HermesLoginHandler

    return HermesLaunchAuthProvider()


def self_check() -> int:
    """Print optional dependency and public-fixture availability."""

    required = {"panel": _version("panel"), "plotly": _version("plotly")}
    dynamic_optional = {
        "torch": _version("torch"),
        "smplx": _version("smplx"),
    }
    print("HERMES GUI self-check")
    for name, version in required.items():
        print(f"  {name}: {version or 'MISSING'}")
    for name, version in dynamic_optional.items():
        print(f"  {name} (Dynamic Scenes): {version or 'MISSING'}")
    public_root = Path(__file__).resolve().parents[3]
    fixture_root = public_root / "data" / "validation_bundles"
    print(f"  source fixtures: {'available' if fixture_root.is_dir() else 'not installed'}")
    amass_fixture = (
        public_root / "data" / "AMASS" / "walking_poses_cmu_105_02.npz"
    )
    model_dir = _default_smpl_model_dir()
    print(
        "  dynamic-scene AMASS fixture: "
        f"{'available' if amass_fixture.is_file() else 'not installed'}"
    )
    print(
        "  licensed SMPL model directory: "
        f"{model_dir if model_dir.is_dir() else 'not configured'}"
    )
    if any(version is None for version in required.values()):
        print(
            "Install the GUI with: "
            'python -m pip install "hermes-radar-sim[gui]"'
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the local HERMES simulator and measurement GUI.",
        epilog=(
            "The launcher generates a random per-launch password but does "
            "not provide TLS or user management. Non-loopback binds require "
            "--unsafe-allow-remote and should be used only on a trusted, "
            "access-controlled network. Wildcard binds also require an "
            "explicit --allow-websocket-origin allowlist."
        ),
    )
    parser.add_argument(
        "--address",
        default="127.0.0.1",
        help=(
            "bind address (default: 127.0.0.1; non-loopback addresses are "
            "rejected unless --unsafe-allow-remote is also supplied)"
        ),
    )
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--unsafe-allow-remote",
        action="store_true",
        help=(
            "UNSAFE: permit a non-loopback bind despite the lack of TLS and "
            "user management; browser-controlled server paths are disabled "
            "in this mode"
        ),
    )
    parser.add_argument(
        "--allow-websocket-origin",
        action="append",
        default=[],
        metavar="HOST[:PORT]",
        help=(
            "permit a concrete browser websocket origin; repeat for multiple "
            "origins. Required for 0.0.0.0/:: binds. If PORT is omitted, "
            "--port is used. Wildcards, schemes, and paths are rejected"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check()
    remote_access = not _is_loopback_address(args.address)
    if remote_access and not args.unsafe_allow_remote:
        parser.error(
            "refusing non-loopback --address without "
            "--unsafe-allow-remote; the per-launch password does not provide "
            "TLS or multi-user access control"
        )
    if _is_unspecified_address(args.address) and not args.allow_websocket_origin:
        parser.error(
            "wildcard --address requires at least one explicit "
            "--allow-websocket-origin HOST[:PORT]"
        )
    try:
        websocket_origins = [
            _normalize_websocket_origin(value, default_port=args.port)
            for value in args.allow_websocket_origin
        ]
        if not websocket_origins:
            websocket_origins = [
                _normalize_websocket_origin(
                    args.address,
                    default_port=args.port,
                )
            ]
    except ValueError as exc:
        parser.error(str(exc))
    if remote_access:
        print(
            "WARNING: UNSAFE REMOTE GUI MODE. The per-launch password does "
            "not provide TLS or user management. Use only on a trusted, "
            "access-controlled network. Browser-entered server paths and "
            "custom scene XML are disabled; configure SMPL models on the "
            "server before launch.",
            file=sys.stderr,
        )
    try:
        import panel as pn
    except ImportError:
        print(
            'HERMES GUI dependencies are missing. Install with '
            '`python -m pip install "hermes-radar-sim[gui]"`.',
            file=sys.stderr,
        )
        return 2
    from tornado.web import RequestHandler

    from .app import (
        _HERMES_ICON_PNG,
        _MAX_MOTION_UPLOAD_BYTES,
        _MAX_SCENE_XML_UPLOAD_BYTES,
        _MAX_STATIC_MESH_UPLOAD_BYTES,
        build_app,
    )

    class HermesIconHandler(RequestHandler):
        """Serve browser-probed application icons without filesystem assets."""

        def get(self) -> None:
            self.set_header("Content-Type", "image/png")
            self.set_header("Cache-Control", "public, max-age=86400")
            self.write(_HERMES_ICON_PNG)

    launch_password = secrets.token_urlsafe(24)
    cookie_secret = secrets.token_hex(32)
    auth_provider = _build_launch_auth_provider(launch_password)
    websocket_message_limit = _websocket_message_limit(
        max(
            _MAX_MOTION_UPLOAD_BYTES,
            _MAX_SCENE_XML_UPLOAD_BYTES,
            _MAX_STATIC_MESH_UPLOAD_BYTES,
        )
    )
    print(
        "HERMES per-launch password: " + launch_password,
        file=sys.stderr,
    )

    pn.serve(
        {
            "/": lambda: build_app(
                initial_bundle=None if args.bundle is None else str(args.bundle),
                remote_access=remote_access,
            )
        },
        address=args.address,
        port=args.port,
        show=not args.no_browser,
        title="HERMES",
        websocket_origin=websocket_origins,
        websocket_max_message_size=websocket_message_limit,
        auth_provider=auth_provider,
        cookie_secret=cookie_secret,
        # FileInput sends one base64-encoded upload over Bokeh's WebSocket.
        # Tornado's connection buffer must therefore match Bokeh's bounded
        # message limit. Keep ordinary HTTP request bodies independently tiny
        # so an unauthenticated cross-site form post is still rejected early.
        http_server_kwargs={
            "max_header_size": _MAX_HTTP_HEADER_BYTES,
            "max_body_size": _MAX_HTTP_BODY_BYTES,
            "max_buffer_size": websocket_message_limit,
        },
        extra_patterns=[
            (
                r"/(?:apple-touch-icon(?:-precomposed)?\.png|favicon\.ico)",
                HermesIconHandler,
            )
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
