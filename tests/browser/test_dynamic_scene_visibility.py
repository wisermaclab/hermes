# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Optional browser regression: HERMES_BROWSER_TESTS=1 pytest tests/browser.

Requires Playwright with Chromium. Uses Panel's local Plotly bundle and the
actual scene interaction script without launching the complete simulator.
"""

import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

pn = pytest.importorskip("panel")
go = pytest.importorskip("plotly.graph_objects")
playwright = pytest.importorskip("playwright.sync_api")

from bokeh.resources import INLINE
from panel.io.save import save

from mmWaveRadar.gui.app import _dynamic_scene_interaction_component, _fit_active_tab_height


@pytest.mark.skipif(
    os.environ.get("HERMES_BROWSER_TESTS") != "1",
    reason="Set HERMES_BROWSER_TESTS=1 to run Chromium UI checks",
)
def test_client_tab_restore_reveals_the_selected_content_without_server_roundtrip():
    inner = pn.Tabs(
        ("Scene and motion", pn.Column("DYNAMIC SCENE")),
        ("Simulation results", pn.Column("DYNAMIC RESULTS")),
        dynamic=False,
    )
    outer = pn.Tabs(
        ("Static Target", pn.Column("STATIC CONTENT")),
        ("Dynamic Scenes", pn.Column(inner)),
        dynamic=False,
    )
    _fit_active_tab_height(inner)
    _fit_active_tab_height(outer)
    html = BytesIO()
    # A standalone document deliberately has no Python roundtrip to repair
    # child.visible after the same client-side active change used at launch.
    save(outer, html, resources=INLINE)
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).is_file():
            pytest.skip("Playwright Chromium is not installed")
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_content(html.getvalue().decode(), wait_until="domcontentloaded")
            page.get_by_text("STATIC CONTENT", exact=True).wait_for(state="visible")
            page.evaluate("""() => {
                const models = Array.from(Bokeh.documents[0]._all_models.values())
                window.outerTabs = models.find(m => m.tabs?.[1]?.title === 'Dynamic Scenes')
                window.innerTabs = models.find(m => m.tabs?.[0]?.title === 'Scene and motion')
                outerTabs.active = 1
            }""")
            # CustomJS compilation is asynchronous, but must reveal the pane
            # locally; there is no server in this standalone browser page.
            page.get_by_text("DYNAMIC SCENE", exact=True).wait_for(state="visible", timeout=5000)
            assert page.evaluate("outerTabs.tabs[1].child.visible")
            page.get_by_text("STATIC CONTENT", exact=True).wait_for(state="hidden")
            page.evaluate("innerTabs.active = 1")
            page.get_by_text("DYNAMIC RESULTS", exact=True).wait_for(state="visible")
            page.get_by_text("DYNAMIC SCENE", exact=True).wait_for(state="hidden")
        finally:
            browser.close()


@pytest.mark.skipif(
    os.environ.get("HERMES_BROWSER_TESTS") != "1",
    reason="Set HERMES_BROWSER_TESTS=1 to run Chromium UI checks",
)
def test_hidden_scene_resizes_when_revealed_without_a_size_change():
    interaction = _dynamic_scene_interaction_component(pn, pn.pane.Plotly(go.Figure()))
    scripts = interaction._scripts
    bundle_dir = Path(pn.__file__).parent / "dist/bundled/plotlyplot"
    bundle = next(bundle_dir.glob("plotly-*.min.js"), None)
    if bundle is None:
        pytest.skip("Panel's local Plotly bundle is unavailable")
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).is_file():
            pytest.skip("Playwright Chromium is not installed")
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1200, "height": 800})
            page.set_default_timeout(10000)
            page.set_content("""<div id="tab" style="visibility:hidden;width:720px;height:570px">
                <div id="host" style="height:100%"></div></div>""")
            page.add_script_tag(path=str(bundle))
            page.evaluate("""async () => {
                const root = document.querySelector('#host').attachShadow({mode:'open'})
                root.innerHTML = '<div id="scene" style="width:100%;height:100%">'
                    + '<div id="plot" style="width:100%;height:100%"></div></div>'
                window.scene = root.querySelector('#scene')
                window.plot = root.querySelector('#plot')
                window.state = {}
                window.resizeCount = 0
                await Plotly.newPlot(plot, [{type:'mesh3d',x:[0,1,0],y:[0,0,1],z:[0,0,1],
                    i:[0],j:[1],k:[2]}], {width:320,height:300,autosize:true,
                    scene:{camera:{eye:{x:1.7,y:1.2,z:.8}}}})
                window.initialCamera = plot._fullLayout.scene._scene.getCamera()
                const relayout = Plotly.relayout.bind(Plotly)
                Plotly.relayout = (target, changes) => {
                    if ('width' in changes && 'height' in changes) window.resizeCount++
                    return relayout(target, changes)
                }
            }""")
            page.evaluate(
                "script => new Function('state', 'scene', script)(state, scene)", scripts["render"]
            )
            page.wait_for_function("() => state.record != null")
            assert page.evaluate("resizeCount") == 0

            # Panel tab activation changes visibility, retaining the same size.
            # The stale first layout must be repaired without a window resize.
            page.evaluate("document.querySelector('#tab').style.visibility = 'visible'")
            page.wait_for_function("() => plot._fullLayout.width === 720")
            assert page.evaluate("resizeCount") == 1
            page.evaluate(
                "script => new Function('state', script)(state)", scripts["after_layout"]
            )
            page.wait_for_timeout(100)
            assert page.evaluate("resizeCount") == 1

            page.evaluate("document.querySelector('#tab').style.width = '900px'")
            page.wait_for_function("() => resizeCount === 2 && plot._fullLayout.width === 900")
            page.evaluate("document.querySelector('#tab').style.visibility = 'hidden'")
            page.wait_for_timeout(100)
            page.evaluate("document.querySelector('#tab').style.visibility = 'visible'")
            page.wait_for_function("() => resizeCount === 3")
            camera_before, camera_after = page.evaluate(
                "[initialCamera, plot._fullLayout.scene._scene.getCamera()]"
            )
            for vector in ("eye", "center", "up"):
                np.testing.assert_allclose(
                    list(camera_after[vector].values()), list(camera_before[vector].values())
                )

            page.evaluate("script => new Function('state', script)(state)", scripts["remove"])
            page.evaluate("document.querySelector('#tab').style.width = '950px'")
            page.wait_for_timeout(150)
            assert page.evaluate("resizeCount") == 3
        finally:
            browser.close()
