# data/

Drop the PoC inputs here. These files are **not** committed (see `../.gitignore`).

**Inputs (added in later steps):**

- `sample_run.mp4` — a short recorded clip of rail/track footage used as the
  video frame source. Any rail video works for a demo; public benchmark
  datasets are introduced in a later step.
- `sim_gps_track.csv` — the simulated GPS track replayed by the localisation
  module. Column layout is defined in a later step
  (expected: `t, lat, lng` and/or `chainage_m`).
- `frames/` — optional folder of still images, an alternative to the video.

**Generated at runtime (later steps):**

- `output/` — annotated frames / saved defect crops referenced by
  `Detection.image_ref`.
- `agv.db` — SQLite database of detections and telemetry.
