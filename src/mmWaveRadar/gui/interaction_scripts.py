# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Browser scripts shared by the target and motion scene components."""

FIND_PLOT_SCRIPT = """
state.findPlot = root => {
  if (root == null) return null
  const direct = root.querySelector?.(".js-plotly-plot")
  if (direct != null) return direct
  for (const element of root.querySelectorAll?.("*") ?? []) {
    if (element.shadowRoot != null) {
      const nested = state.findPlot(element.shadowRoot)
      if (nested != null) return nested
    }
  }
  return null
}
"""

TARGET_INTERACTION_SCRIPTS = {
    "render": FIND_PLOT_SCRIPT
    + """
const multiply = (left, right) => left.map(
  row => right[0].map(
    (_, column) => row.reduce(
      (sum, value, index) => sum + value * right[index][column], 0
    )
  )
)
const matrixVector = (matrix, vector) => matrix.map(
  row => row.reduce(
    (sum, value, index) => sum + value * vector[index], 0
  )
)
const transpose = matrix => matrix[0].map(
  (_, column) => matrix.map(row => row[column])
)
const rotation = (yawDeg, pitchDeg, rollDeg) => {
  const yaw = yawDeg * Math.PI / 180
  const pitch = pitchDeg * Math.PI / 180
  const roll = rollDeg * Math.PI / 180
  const rz = [
    [Math.cos(yaw), -Math.sin(yaw), 0],
    [Math.sin(yaw), Math.cos(yaw), 0],
    [0, 0, 1],
  ]
  const ry = [
    [Math.cos(pitch), 0, Math.sin(pitch)],
    [0, 1, 0],
    [-Math.sin(pitch), 0, Math.cos(pitch)],
  ]
  const rx = [
    [1, 0, 0],
    [0, Math.cos(roll), -Math.sin(roll)],
    [0, Math.sin(roll), Math.cos(roll)],
  ]
  return multiply(multiply(rz, ry), rx)
}
const clamp = (value, lower, upper) => Math.min(
  upper, Math.max(lower, value)
)
const install = () => {
  if (state.removed) return
  const plot = state.findPlot(scene)
  if (plot == null || typeof plot.on !== "function") {
    state.installTimer = setTimeout(install, 50)
    return
  }
  state.plot = plot
  const emitGesture = () => {
    data.target_gesture = {
      sequence: (data.target_gesture.sequence ?? 0) + 1,
      yaw_deg: state.nextYaw,
      pitch_deg: state.nextPitch,
    }
  }
  const applyPreview = (yawDeg, pitchDeg) => {
    const current = rotation(
      state.startYaw, state.startPitch, state.startRoll
    )
    const next = rotation(yawDeg, pitchDeg, state.startRoll)
    const undoCurrent = transpose(current)
    const xs = []
    const ys = []
    const zs = []
    for (let index = 0; index < state.startX.length; index++) {
      const relative = [
        state.startX[index] - state.pivot[0],
        state.startY[index] - state.pivot[1],
        state.startZ[index] - state.pivot[2],
      ]
      const local = matrixVector(undoCurrent, relative)
      const rotated = matrixVector(next, local)
      xs.push(rotated[0] + state.pivot[0])
      ys.push(rotated[1] + state.pivot[1])
      zs.push(rotated[2] + state.pivot[2])
    }
    return window.Plotly.restyle(
      state.plot, {x: [xs], y: [ys], z: [zs]}, [0]
    )
  }
  const applySettledPreview = preview => {
    state.previewInFlight = true
    Promise.resolve(applyPreview(preview.yaw, preview.pitch))
      .catch(error => console.error("Target preview failed", error))
      .finally(() => {
        state.previewInFlight = false
        if (state.finalPreview != null) {
          const finalPreview = state.finalPreview
          state.finalPreview = null
          applySettledPreview(finalPreview)
          return
        }
        if (state.shiftDragging && state.previewPending) {
          requestPreview()
        }
      })
  }
  const requestPreview = () => {
    state.previewPending = true
    if (state.previewInFlight || state.previewFrame != null) return
    state.previewFrame = requestAnimationFrame(() => {
      state.previewFrame = null
      if (!state.shiftDragging) {
        state.previewPending = false
        return
      }
      const preview = {yaw: state.nextYaw, pitch: state.nextPitch}
      state.previewPending = false
      applySettledPreview(preview)
    })
  }
  const finish = event => {
    if (!state.shiftDragging) return
    event?.preventDefault?.()
    if (event?.type?.startsWith("pointer")) {
      event.stopImmediatePropagation()
    }
    state.shiftDragging = false
    window.removeEventListener("pointermove", state.pointerMove, true)
    window.removeEventListener("pointerup", state.pointerUp, true)
    window.removeEventListener("pointercancel", state.pointerCancel, true)
    if (
      state.pointerId != null
      && target_overlay.hasPointerCapture?.(state.pointerId)
    ) {
      target_overlay.releasePointerCapture(state.pointerId)
    }
    if (state.previewFrame != null) {
      cancelAnimationFrame(state.previewFrame)
      state.previewFrame = null
    }
    state.previewPending = false
    const finalPreview = {yaw: state.nextYaw, pitch: state.nextPitch}
    if (state.previewInFlight) {
      // Replace every intermediate request with exactly one final draw.
      state.finalPreview = finalPreview
    } else {
      applySettledPreview(finalPreview)
    }
    emitGesture()
    target_overlay.style.cursor = "grab"
    target_overlay.style.pointerEvents = state.shiftHeld ? "auto" : "none"
  }
  state.pointerMove = event => {
    if (!state.shiftDragging) return
    if ((event.buttons & 1) === 0) {
      finish(event)
      return
    }
    event.preventDefault()
    event.stopImmediatePropagation()
    const sensitivity = 0.35
    state.nextYaw = Math.round(10 * clamp(
      state.startYaw + sensitivity * (event.clientX - state.startClientX),
      -75, 75
    )) / 10
    state.nextPitch = Math.round(10 * clamp(
      state.startPitch - sensitivity * (event.clientY - state.startClientY),
      -75, 75
    )) / 10
    requestPreview()
  }
  state.pointerUp = event => finish(event)
  state.pointerCancel = event => {
    state.nextYaw = state.startYaw
    state.nextPitch = state.startPitch
    finish(event)
  }
  state.pointerDown = event => {
    if (!state.shiftHeld || event.button !== 0) return
    if (
      state.shiftDragging
      || state.previewInFlight
      || state.finalPreview != null
    ) {
      event.preventDefault()
      event.stopImmediatePropagation()
      return
    }
    const activePlot = state.findPlot(scene)
    if (activePlot == null) return
    state.plot = activePlot
    const trace = state.plot.data?.[0]
    if (trace == null) return
    event.preventDefault()
    event.stopImmediatePropagation()
    state.shiftDragging = true
    state.pointerId = event.pointerId
    target_overlay.setPointerCapture?.(event.pointerId)
    target_overlay.style.cursor = "grabbing"
    state.startClientX = event.clientX
    state.startClientY = event.clientY
    state.startX = Array.from(trace.x)
    state.startY = Array.from(trace.y)
    state.startZ = Array.from(trace.z)
    const meta = trace.meta ?? {}
    state.startYaw = Number(meta.yaw_deg ?? 0)
    state.startPitch = Number(meta.pitch_deg ?? 0)
    state.startRoll = Number(meta.roll_deg ?? 0)
    state.nextYaw = state.startYaw
    state.nextPitch = state.startPitch
    state.previewPending = false
    state.finalPreview = null
    state.pivot = Array.from(
      meta.orientation_pivot ?? [0, 0, 0]
    ).map(Number)
    window.addEventListener("pointermove", state.pointerMove, true)
    window.addEventListener("pointerup", state.pointerUp, true)
    window.addEventListener("pointercancel", state.pointerCancel, true)
  }
  state.keyDown = event => {
    if (event.key !== "Shift") return
    state.shiftHeld = true
    target_overlay.style.cursor = "grab"
    target_overlay.style.pointerEvents = "auto"
  }
  state.keyUp = event => {
    if (event.key !== "Shift") return
    state.shiftHeld = false
    if (!state.shiftDragging) {
      target_overlay.style.pointerEvents = "none"
    }
  }
  state.windowBlur = () => {
    state.shiftHeld = false
    if (state.shiftDragging) {
      finish()
    } else {
      target_overlay.style.pointerEvents = "none"
    }
  }
  state.lostPointerCapture = event => {
    if (state.shiftDragging) finish(event)
  }
  // The overlay, rather than Plotly's drag layer, owns every Shift gesture.
  target_overlay.addEventListener("pointerdown", state.pointerDown, true)
  target_overlay.addEventListener(
    "lostpointercapture", state.lostPointerCapture, true
  )
  window.addEventListener("keydown", state.keyDown, true)
  window.addEventListener("keyup", state.keyUp, true)
  window.addEventListener("blur", state.windowBlur, true)
}
state.removed = false
state.shiftHeld = false
state.previewInFlight = false
state.previewPending = false
state.finalPreview = null
state.installTimer = setTimeout(install, 0)
""",
    "target_result": """
const update = data.target_result
const activePlot = state.findPlot?.(scene)
if (activePlot != null) state.plot = activePlot
if (
  state.plot != null
  && update != null
  && update.sequence !== state.resultSequence
) {
  state.resultSequence = update.sequence
  const traceUpdate = {
    facecolor: [update.facecolor],
    meta: [update.meta],
  }
  if (update.x != null && update.y != null && update.z != null) {
    traceUpdate.x = [update.x]
    traceUpdate.y = [update.y]
    traceUpdate.z = [update.z]
  }
  window.Plotly.restyle(
    state.plot,
    traceUpdate,
    [0],
  )
}
""",
    "remove": """
state.removed = true
clearTimeout(state.installTimer)
if (state.previewFrame != null) cancelAnimationFrame(state.previewFrame)
if (state.pointerDown != null) {
  target_overlay.removeEventListener("pointerdown", state.pointerDown, true)
  target_overlay.removeEventListener(
    "lostpointercapture", state.lostPointerCapture, true
  )
}
if (state.pointerMove != null) {
  window.removeEventListener("pointermove", state.pointerMove, true)
  window.removeEventListener("pointerup", state.pointerUp, true)
  window.removeEventListener("pointercancel", state.pointerCancel, true)
}
if (state.keyDown != null) {
  window.removeEventListener("keydown", state.keyDown, true)
  window.removeEventListener("keyup", state.keyUp, true)
  window.removeEventListener("blur", state.windowBlur, true)
}
""",
}

DYNAMIC_INTERACTION_SCRIPTS = {
    "render": FIND_PLOT_SCRIPT
    + """
const install = () => {
  if (state.removed) return
  const plot = state.findPlot(scene)
  if (
    plot == null || window.Plotly == null || !scene.isConnected
    || plot._fullLayout == null || typeof plot.on !== "function"
  ) {
    state.installTimer = setTimeout(install, 50)
    return
  }

  // Panel applies server-side Figure.plotly_restyle calls through the global
  // Plotly.restyle entry point. Guard only this plot and retain only its most
  // recent mesh update while Plotly owns a camera gesture.
  const registryKey = "__hermesDynamicSceneRestyleGuard"
  let guard = window[registryKey]
  if (guard == null) {
    const originalRestyle = window.Plotly.restyle.bind(window.Plotly)
    const originalUpdate = window.Plotly.update.bind(window.Plotly)
    const records = new WeakMap()
    const cloneCamera = camera => {
      if (camera == null || typeof camera !== "object") return null
      return JSON.parse(JSON.stringify(camera))
    }
    const readCamera = target => {
      const liveCamera = target?._fullLayout?.scene?._scene?.getCamera?.()
      return cloneCamera(
        liveCamera
        ?? target?.layout?.scene?.camera
        ?? target?._fullLayout?.scene?.camera
      )
    }
    const runLatest = (target, record) => {
      if (
        record.interacting
        || record.inFlight
        || record.pending == null
      ) return Promise.resolve(target)
      const args = record.pending
      record.pending = null
      record.inFlight = true
      // Plotly can rebuild a WebGL scene when mesh coordinates change. Send
      // the most recently observed user camera in the same update so no
      // playback frame can restore the figure's initial perspective or zoom.
      const camera = readCamera(target) ?? record.camera
      if (camera != null) record.camera = cloneCamera(camera)
      const update = (
        camera == null
          ? originalRestyle(target, ...args)
          : originalUpdate(
              target,
              args[0],
              {"scene.camera": camera},
              args[1],
            )
      )
      return Promise.resolve(update)
        .catch(error => console.error(
          "Dynamic-scene frame refresh failed", error
        ))
        .finally(() => {
          record.inFlight = false
          if (!record.interacting && record.pending != null) {
            requestAnimationFrame(() => runLatest(target, record))
          }
        })
    }
    const guardedRestyle = (target, ...args) => {
      const record = records.get(target)
      if (record == null) return originalRestyle(target, ...args)
      record.pending = args
      return runLatest(target, record)
    }
    guard = {
      cloneCamera,
      originalRestyle,
      originalUpdate,
      readCamera,
      records,
      guardedRestyle,
      runLatest,
    }
    window[registryKey] = guard
    window.Plotly.restyle = guardedRestyle
  }

  state.plot = plot
  state.guard = guard
  state.record = {
    camera: guard.readCamera(plot),
    interacting: false,
    inFlight: false,
    pending: null,
  }
  guard.records.set(plot, state.record)

  state.captureCamera = eventData => {
    if (state.record == null) return
    const eventCamera = eventData?.["scene.camera"]
    const camera = (
      eventCamera != null
        ? state.guard.cloneCamera(eventCamera)
        : state.guard.readCamera(state.plot)
    )
    if (camera != null) state.record.camera = camera
  }

  state.flush = () => {
    if (state.record == null) return
    state.record.interacting = false
    state.guard.runLatest(state.plot, state.record)
  }
  state.beginInteraction = () => {
    if (state.record != null) state.record.interacting = true
  }
  state.pointerDown = event => {
    if (event.button == null || event.button === 0) {
      clearTimeout(state.wheelTimer)
      state.beginInteraction()
    }
  }
  state.pointerEnd = () => {
    // Let Plotly finish recording the new camera before drawing the newest
    // queued human frame.
    requestAnimationFrame(() => requestAnimationFrame(state.flush))
  }
  state.wheel = () => {
    state.beginInteraction()
    clearTimeout(state.wheelTimer)
    state.wheelTimer = setTimeout(state.flush, 180)
  }
  state.windowBlur = () => state.flush()

  plot.addEventListener("pointerdown", state.pointerDown, true)
  plot.addEventListener("wheel", state.wheel, true)
  plot.on("plotly_relayout", state.captureCamera)
  plot.on("plotly_relayouting", state.captureCamera)
  window.addEventListener("pointerup", state.pointerEnd, true)
  window.addEventListener("pointercancel", state.pointerEnd, true)
  window.addEventListener("blur", state.windowBlur, true)

  // Mounted inactive Panel tabs retain their size and use visibility:hidden.
  // A restored tab can become visible after Plotly's first layout, without
  // another Plotly layout pass. Resize once the visible container has settled.
  const plotContainer = plot.parentElement ?? plot.getRootNode()?.host ?? scene
  state.resizeVisiblePlot = () => {
    if (state.removed || !scene.isConnected) return
    const bounds = plotContainer.getBoundingClientRect()
    const visible = (
      bounds.width > 0 && bounds.height > 0
      && window.getComputedStyle(scene).visibility === "visible"
    )
    if (!visible) {
      state.lastVisibleSize = null
      return
    }
    const size = [bounds.width, bounds.height]
    if (state.lastVisibleSize?.every((value, i) => value === size[i])) return
    state.lastVisibleSize = size
    // Panel gives Plotly explicit width/height, for which Plots.resize is a
    // no-op. Match its actual pane after the tab's layout becomes visible.
    Promise.resolve(window.Plotly.relayout(plot, {
      width: bounds.width, height: bounds.height,
    })).catch(error => {
      state.lastVisibleSize = null
      console.warn("Dynamic-scene resize failed", error)
    })
  }
  state.scheduleResize = () => {
    cancelAnimationFrame(state.resizeFrame)
    state.resizeFrame = requestAnimationFrame(() => {
      state.resizeFrame = requestAnimationFrame(state.resizeVisiblePlot)
    })
  }
  state.resizeObserver = new ResizeObserver(state.scheduleResize)
  state.resizeObserver.observe(scene)
  if (plotContainer !== scene) state.resizeObserver.observe(plotContainer)
  state.visibilityObserver = new MutationObserver(state.scheduleResize)
  // Cross shadow roots to observe the containing tab panels, not the moving
  // mesh or Plotly's frequently changing child nodes.
  let ancestor = scene
  while (ancestor != null) {
    state.visibilityObserver.observe(ancestor, {
      attributes: true,
      attributeFilter: ["style", "class", "hidden", "aria-hidden"],
    })
    ancestor = ancestor.parentElement ?? ancestor.getRootNode()?.host
  }
  state.scheduleResize()
}
state.removed = false
state.installTimer = setTimeout(install, 0)
""",
    "after_layout": "state.scheduleResize?.()",
    "remove": """
state.removed = true
clearTimeout(state.installTimer)
clearTimeout(state.wheelTimer)
cancelAnimationFrame(state.resizeFrame)
state.resizeObserver?.disconnect()
state.visibilityObserver?.disconnect()
if (state.record != null) {
  state.record.interacting = false
  state.record.pending = null
}
if (state.plot != null) {
  state.guard?.records?.delete(state.plot)
  state.plot.removeEventListener("pointerdown", state.pointerDown, true)
  state.plot.removeEventListener("wheel", state.wheel, true)
  state.plot.removeListener?.("plotly_relayout", state.captureCamera)
  state.plot.removeListener?.("plotly_relayouting", state.captureCamera)
}
if (state.pointerEnd != null) {
  window.removeEventListener("pointerup", state.pointerEnd, true)
  window.removeEventListener("pointercancel", state.pointerEnd, true)
  window.removeEventListener("blur", state.windowBlur, true)
}
""",
}
