# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Self-contained launch screen; no external assets or browser scripts."""

from __future__ import annotations

from html import escape


_LOGIN_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>HERMES · Open your workspace</title>
  <style>
    * { box-sizing: border-box; }
    :root { color-scheme: dark; }
    body {
      margin: 0; min-height: 100vh; min-height: 100svh;
      display: flex; flex-direction: column;
      background: #080f1b; color: #eef4fc;
      font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    body::before {
      content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
      background:
        radial-gradient(ellipse at 20% 62%, #0d465a40, transparent 48%),
        radial-gradient(ellipse at 100% 0%, #223e6b38, transparent 48%),
        linear-gradient(#8ab8e805 1px, transparent 1px),
        linear-gradient(90deg, #8ab8e805 1px, transparent 1px);
      background-size: auto, auto, 64px 64px, 64px 64px;
    }
    .header, .layout { width: min(1180px, calc(100% - 96px)); margin: 0 auto; }
    .header { display: flex; align-items: center; justify-content: space-between; padding: 34px 0; }
    .brand { display: flex; align-items: center; gap: 12px; font-size: 20px; font-weight: 700; letter-spacing: .13em; }
    .brand svg { width: 32px; height: 32px; color: #6fe2eb; }
    .layout { flex: 1; display: grid; grid-template-columns: minmax(0, 1fr) 410px;
      gap: clamp(40px, 7vw, 100px); align-items: center; padding: 24px 0 44px; }
    h1 { font-size: clamp(64px, 7.3vw, 98px); line-height: .98; font-weight: 650;
      letter-spacing: -.065em; margin: 0 0 25px; }
    h1 span { display: block; color: #6fe2eb; }
    .scene { max-width: 590px; margin: 24px -12px 0; }
    .scene svg { display: block; width: 100%; height: auto; overflow: visible; }
    .signal { stroke-dasharray: 10 8; animation: travel 7s linear infinite; }
    .echo { animation: echo 5s ease-in-out infinite; }
    .echo-late { animation-delay: -2.5s; }
    @keyframes travel { to { stroke-dashoffset: -108; } }
    @keyframes echo { 0%, 100% { opacity: .3; } 50% { opacity: 1; } }
    .panel {
      position: relative; padding: 38px; border: 1px solid #2a3c52; border-radius: 20px;
      background: linear-gradient(150deg, #162437f2, #101b2bf5);
      box-shadow: 0 28px 90px #0005, inset 0 1px #ffffff05;
    }
    .panel::before { content: ""; position: absolute; height: 1px; top: -1px; left: 28px; right: 28px;
      background: linear-gradient(90deg, transparent, #6fe2ebaa, transparent); }
    .panel-icon { display: grid; place-items: center; width: 44px; height: 44px;
      color: #6fe2eb; background: #6fe2eb0c; border: 1px solid #6fe2eb24; border-radius: 12px; margin-bottom: 26px; }
    .panel-icon svg { width: 22px; height: 22px; }
    h2 { font-size: 27px; line-height: 1.25; letter-spacing: -.035em; font-weight: 650; margin: 0 0 26px; }
    input, button { width: 100%; font: inherit; border-radius: 9px; }
    input { height: 52px; padding: 0 15px; border: 1px solid #465c76; background: #090f1b88;
      color: #f3f8ff; font-size: 16px; transition: border-color .2s, box-shadow .2s; }
    input::placeholder { color: #778ca6; }
    input:hover { border-color: #6b839e; }
    input:focus { border-color: #6fe2eb; outline: 0; box-shadow: 0 0 0 3px #6fe2eb1f; }
    button { height: 50px; display: flex; align-items: center; justify-content: center; gap: 14px;
      margin-top: 16px; border: 1px solid #9ceff4; background: #6fe2eb; color: #082330;
      font-size: 14px; font-weight: 700; cursor: pointer; transition: background .2s, transform .2s; }
    button svg { width: 18px; height: 18px; }
    button:hover { background: #99edf3; transform: translateY(-1px); }
    button:active { transform: translateY(0); }
    button:focus-visible { outline: 3px solid #b1f4ff; outline-offset: 4px; }
    .error { color: #ffd5cc; background: #d16b4e16; border: 1px solid #eb9c794d;
      border-radius: 8px; padding: 11px 13px; font-size: 13px; line-height: 1.6; margin: 0 0 21px; }
    @media (max-width: 1000px) {
      .header, .layout { width: calc(100% - 48px); }
      .layout { grid-template-columns: minmax(0, 1fr) 360px; gap: 32px; }
      .panel { padding: 30px; }
      h1 { font-size: 72px; }
    }
    @media (max-width: 720px) {
      .header, .layout { width: min(460px, calc(100% - 40px)); }
      .header { padding: 24px 0; }
      .layout { grid-template-columns: 1fr; gap: 28px; padding: 20px 0 36px; }
      .intro { text-align: center; }
      h1 { font-size: 58px; margin-bottom: 18px; }
      h1 span { display: inline; }
      .scene { max-width: 330px; margin: 12px auto 0; }
      .panel { padding: 28px; }
      .panel-icon { display: none; }
    }
    @media (max-width: 420px) {
      .header { padding: 20px 0; }
      .layout { gap: 24px; padding: 12px 0 28px; }
      h1 { font-size: 52px; margin-bottom: 16px; }
      .scene { display: none; }
      .panel { padding: 24px; }
      h2 { font-size: 25px; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation: none !important; transition: none !important; }
    }
  </style>
</head>
<body>
  <header class="header">
    <div class="brand">
      <svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <path d="M5 24V8m22 16V8M5 16h22" stroke="currentColor" stroke-width="2"/>
        <path d="m11 8 5-3 5 3m-10 16 5 3 5-3" stroke="currentColor" stroke-width="1.5" opacity=".55"/>
        <circle cx="16" cy="16" r="3" fill="currentColor"/>
      </svg>
      HERMES
    </div>
  </header>
  <main class="layout">
    <section class="intro" aria-labelledby="intro-title">
      <h1 id="intro-title">Scene to <span>signal.</span></h1>
      <figure class="scene" aria-hidden="true">
        <svg viewBox="0 0 620 280" fill="none">
          <defs>
            <linearGradient id="floor" x1="300" y1="70" x2="300" y2="280" gradientUnits="userSpaceOnUse">
              <stop stop-color="#6fe2eb" stop-opacity=".05"/><stop offset="1" stop-color="#6fe2eb" stop-opacity=".01"/>
            </linearGradient>
            <radialGradient id="glow"><stop stop-color="#6fe2eb" stop-opacity=".22"/><stop offset="1" stop-color="#6fe2eb" stop-opacity="0"/></radialGradient>
          </defs>
          <ellipse cx="310" cy="180" rx="270" ry="100" fill="url(#glow)"/>
          <path d="m55 185 288-102 230 78-287 105Z" fill="url(#floor)" stroke="#6fe2eb" stroke-opacity=".24"/>
          <g stroke="#6fe2eb" stroke-opacity=".1">
            <path d="m113 205 287-102M170 225l287-102M228 246l287-102M112 165l232 80M170 144l231 81M227 124l232 80M285 103l231 81"/>
          </g>
          <path d="M55 185v-65L343 18l230 78v65M343 18v65" stroke="#6fe2eb" stroke-opacity=".28"/>
          <path d="M55 120 286 201l287-105" stroke="#6fe2eb" stroke-opacity=".06"/>
          <path d="m459 138 43-15 35 12v35l-43 16-35-13Z" fill="#779acb0c" stroke="#a9c3e5" stroke-opacity=".25"/>
          <path d="m459 138 35 13 43-16m-43 16v35" stroke="#a9c3e5" stroke-opacity=".18"/>
          <path class="signal" d="m126 183 238-53 151 15-151 27-238 11" stroke="#f5bb74" stroke-opacity=".7"/>
          <path d="m126 183 238-53" stroke="#6fe2eb" stroke-opacity=".45"/>
          <path d="M160 151q21 16 19 42M184 135q39 23 35 64M210 120q52 32 48 87" stroke="#6fe2eb" stroke-opacity=".3"/>
          <ellipse class="echo" cx="126" cy="183" rx="19" ry="7" stroke="#6fe2eb" stroke-opacity=".6"/>
          <ellipse cx="126" cy="183" rx="8" ry="3" fill="#6fe2eb"/>
          <path d="M126 183v-14m-6 8 6-8 6 8" stroke="#b1f4ff" stroke-width="1.5"/>
          <g stroke="#c5eef4" stroke-width="1.4" stroke-linejoin="round">
            <circle cx="365" cy="84" r="8" fill="#6fe2eb18"/>
            <path d="m352 105 25-4 8 29-29 5Zm0 0 33 25m-8-29-21 34m7-31 4 29" fill="#6fe2eb12"/>
            <path d="m352 105-12 20 5 19m32-43 18 15 4 18M356 135l-6 28-11 18m46-51-9 28 11 20"/>
          </g>
          <ellipse class="echo echo-late" cx="365" cy="178" rx="28" ry="10" stroke="#6fe2eb" stroke-opacity=".35"/>
          <g fill="#91a7c1" font-family="ui-monospace,monospace" font-size="9" letter-spacing="1">
            <text x="83" y="217">RADAR</text><text x="396" y="86">MOTION</text><text x="501" y="110">SCENE</text>
          </g>
          <g stroke="#91a7c1" stroke-opacity=".45"><path d="m99 208 14-16m282-101-16 4m128 19 0 9"/></g>
        </svg>
      </figure>
    </section>
    <section class="panel" aria-labelledby="login-title">
      <div class="panel-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="5" y="10" width="14" height="11" rx="3"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3"/>
        </svg>
      </div>
      <h2 id="login-title">Sign in</h2>
      <!--LOGIN_ERROR-->
      <form method="post" action="/__hermes_login">
        <input id="password" name="password" type="password" aria-label="Password"
          required autocomplete="current-password" placeholder="Password" spellcheck="false" autofocus>
        <button type="submit">Open HERMES
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true">
            <path d="M4 12h15m-6-6 6 6-6 6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </form>
    </section>
  </main>
</body>
</html>
"""


def render_login_page(error: str | None = None) -> str:
    message = f'<div class="error" role="alert">{escape(error)}</div>' if error else ""
    return _LOGIN_PAGE.replace("<!--LOGIN_ERROR-->", message)
