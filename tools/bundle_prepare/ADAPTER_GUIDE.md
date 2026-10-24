# Write a dataset adapter

Keep dataset-specific decoding and synchronization upstream of `BundleWriter`.
The adapter supplies numeric ADC, a `BundleSensorConfig`, frame records and
optional geometry or motion. Use the public writer demonstrated in
[the measurement tutorial](BRING_YOUR_OWN.md), then load the result through the
public API. No dataset-name dispatch belongs in the loader.

The mmRadPose adapter is a worked example: it discovers a parsed radar cube,
maps a selected trial frame to fitted motion, selects a motion window covering
the full acquisition, transforms coordinates into the radar world, and supplies
its sensor/frame/motion sidecars. Its laser fitting remains dataset-specific.
The common writer canonicalizes the sidecars, checks scene dependencies and
publishes only a valid completed bundle. The GUI exporters use the same writer.

For a new adapter, establish these facts against acquisition records:

| Item | Evidence to retain |
| --- | --- |
| Timing | Frame starts, chirp repetition, dropped frames, source clock alignment, motion coverage through the final physical slot. |
| Frames | Contiguous bundle indices plus original frame identifiers. |
| Coordinates | Source axes, handedness, units, radar pose, and the transform actually applied. |
| Channels | Exact channel-axis mapping to physical zero-based TX/RX indices. |
| TDM | Full physical slot sequence, independent of stored/selected channels. |
| IQ | Mixer sign and transformations performed by the decoder. |
| Phase signs | Whether fixed board signs were retained or removed upstream. |
| Carrier | Frequency at the first ADC sample, start frequency and ADC start time; retain source waveform fields. |
| Amplitude | Counts, volts or another declared unit; any scale/calibration applied. |
| Distribution | Rights for the capture, fitted motion and scene assets. |

Test a relocated directory and ZIP after the original input tree becomes
unavailable. Include invalid-timing and channel-order cases and test static
geometry without SMPL. Treat plausibility diagnostics as warnings and evaluate
physical conventions with multiple independently known target positions.
