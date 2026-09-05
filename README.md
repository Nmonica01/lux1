# Nightwatch — updated files

What changed, and why.

## Backend (`backend/main.py`)

- Kept your YOLOv8 + BoT-SORT pipeline (already detects all 80 COCO classes,
  which includes person, car, truck, bus, motorcycle, bicycle and animals
  like dog / cat / bird / horse / cow / sheep / bear / elephant / zebra /
  giraffe — no model change needed for "people, cars, vans, animals, etc").
- Added suspicious-behavior detection on top of the existing tracker output.
  For every tracked object it now checks:
  - **Loitering** — present for 20+ seconds while staying inside a small
    radius of its starting point.
  - **Fast / erratic movement** — covers an unusually large fraction of the
    frame in a short time between two detections.
  - **Activity during restricted hours** — a person detected while the
    server's wall-clock is inside a restricted window (default 22:00–06:00).
    Raw video files don't carry a reliable capture timestamp, so this checks
    the time of *analysis*, not the time of recording — pass real camera
    timestamps in if you have them and want it tied to actual footage time.
  All three are tunable constants at the top of the file
  (`LOITER_SECONDS_THRESHOLD`, `LOITER_DISPLACEMENT_RATIO`,
  `FAST_MOVEMENT_SPEED_RATIO`), and the restricted-hours window can also be
  overridden per request via the `restricted_start_hour` / `restricted_end_hour`
  form fields on `/analyze`.
- `/analyze` now returns `suspicious_events`, `analyzed_at`, and
  `restricted_hours_active` alongside the existing `objects` summary.

Run it the same way as before:

```
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

## Frontend (`frontend/src/*`)

- `App.tsx` now reads `suspicious_events` from the backend and renders them
  as CRITICAL (after-hours activity) or HIGH (loitering, fast movement)
  alerts, shown above the regular per-class object-count alerts.
- The browser-side quick-preview path (`ALLOWED_CLASSES`) is broadened
  beyond vehicles to also flag people and common animals, so the live
  overlay while scrubbing matches what the backend eventually reports.
- Virtual-fence rules on the Fences tab now mirror the three real detection
  rules (loitering, restricted hours, rapid movement) instead of placeholder
  copy.
- New visual direction: **deep sea / slate blue control room** —
  `theme.css` holds the palette and type tokens (Inter + IBM Plex Mono,
  near-black slate surfaces, a single sonar-cyan accent, amber/coral
  reserved for behavior alerts). `App.css`, `Pages.css`, `Video.css` and
  `Readability.css` were rewritten against those tokens; drop them in over
  your existing files with the same names.

No other component names or file layout changed, so this drops into your
existing Vite project structure directly.
