# ClassSightAI: AI-Based Classroom Attendance from Video / CCTV Footage

SmartAttend takes a classroom video file (MP4/AVI/MOV), an RTSP/CCTV stream or a webcam. It detects and
recognises every student's face, marks attendance automatically and exports a formatted Excel report, an
annotated video, and CSV and JSON copies of the data.
**A student is never marked present more than once per session.** All processing runs locally.

```
video/CCTV ─► frame skip ─► CLAHE (low light) ─► SCRFD detection (full frame + tiles) ─► ByteTrack IDs
          ─► per-track best-quality crops (sharpness·size·frontal pose) ─► ArcFace embeddings
          ─► cosine match vs encrypted gallery ─► multi-frame voting ─► session de-duplication
          ─► Excel / CSV / JSON / annotated MP4 / unknown-face clusters
```

## Project layout

```
main.py                 CLI (process / enroll / enroll-bulk / override / assign-unknown / report / ...)
evaluate.py             precision / recall / accuracy against a ground-truth CSV
config.yaml             every threshold and path
src/
  config.py             YAML loader (attribute access, path resolution)
  db.py                 SQLite + Fernet field encryption, users, audit log, history
  detector.py           InsightFace SCRFD + ArcFace (buffalo_l), tiling, upscaling, GPU→CPU fallback
  tracker.py            ByteTrack (Kalman + two-stage IoU association)
  recognizer.py         embedding gallery, cosine matching (NumPy or FAISS)
  attendance.py         voting, de-duplication, spoof/duplicate flags, unknown clustering
  quality.py            sharpness / pose / size quality score, CLAHE low-light enhancement
  liveness.py           basic anti-spoofing (ONNX model or heuristic)
  enroll.py             single + bulk enrollment with photo validation
  pipeline.py           end-to-end processor, unknown → student assignment
  report_excel.py       4-sheet formatted workbook + CSV/JSON
  video_annotator.py    boxes, labels, "Present: X / Y" overlay
  notify.py             e-mail / WhatsApp (text only)
  privacy.py            retention / auto-deletion
  evaluation.py         metrics
  behaviour/            Class Attentiveness & Behaviour Analytics (see section 9)
    pose_estimator.py   YOLO11-pose body keypoints, face↔person association
    head_pose.py        solvePnP / 5-point / body head pose, board-zone geometry
    landmarks.py        InsightFace 68-pt landmarks → EAR / MAR
    object_detector.py  YOLO phone/book/laptop(/pen) detection + hand association
    behaviour_classifier.py  BehaviourClassifier interface + rule-based classifier
    timeline.py         smoothing, out-of-frame gaps, events, per-second data
    scoring.py          per-frame / session / class scores, engagement drops
    analyzer.py         live pass, signal cache (encrypted Parquet), rescoring
    report_behaviour_excel.py  6-sheet behaviour workbook + CSV/JSON
    video_overlay.py    behaviour video (HUD, sparkline, pop-ups, 720p preview)
    zones.py            per-camera board/teacher zones
    evaluation.py       per-behaviour P/R/F1 + confusion matrix
app/streamlit_app.py    web UI (role-aware menu)
app/pages/attentiveness.py  "Attentiveness & Behaviour" tab
evaluate_behaviour.py   behaviour evaluation CLI
tests/                  pytest suite (de-duplication, Excel, DB, tracker, end-to-end pipeline)
data/                   DB, key, models (created at runtime)
outputs/                one folder per session
```

## 1. Setup

**Requirements:** Python 3.10–3.14, about 2 GB of free RAM, and a CPU. An NVIDIA GPU is optional.

```bash
# Windows (PowerShell). Keep the venv outside OneDrive if the project lives there.
python -m venv C:\venvs\smartattend
C:\venvs\smartattend\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On first use, InsightFace downloads the `buffalo_l` model pack (~280 MB) into `data/models/buffalo_l`.

**GPU (optional):** run `pip uninstall onnxruntime` and then `pip install onnxruntime-gpu` (matching your CUDA version).
Keep `models.use_gpu: true`. If CUDA is not available, SmartAttend logs this and falls back to the CPU.

**Encryption key:** on first run a Fernet key is written to `data/.secret.key`. In production, set the
environment variable `SMARTATTEND_KEY` and store the key separately from the database. **If you lose the key,
the stored embeddings and names cannot be decrypted.**

**Docker:**
```bash
docker build -t smartattend .
docker run -p 8501:8501 -v smartattend_data:/app/data -v smartattend_out:/app/outputs \
           -e SMARTATTEND_KEY=<your-fernet-key> smartattend
# Generate a key: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## 2. Enrollment

Each student needs 3–10 clear photos taken from slightly different angles, or a 5–10 second selfie video.
Every photo is checked as follows:
- a photo with no face, an unreadable image, a face smaller than `enrollment.min_face_size` or a blurry
  face (Laplacian variance below `min_blur_var`) is **rejected**;
- a photo with **more than one face** triggers a **warning**, and the largest face is used;
- a photo whose embedding disagrees with the student's other photos is rejected as an **outlier**;
- the student's embeddings are averaged, L2-normalised and **encrypted**. The photos are never stored.

Consent is mandatory (DPDP Act). Enrollment is refused without it.

```bash
# single student
python main.py enroll --roll 23 --name "Aditya Kumar" --class 10A --images p1.jpg p2.jpg p3.jpg --consent
python main.py enroll --roll 24 --name "Riya Shah" --class 10A --video riya_selfie.mp4 --consent

# bulk:  dataset/<RollNo>_<Name>/*.jpg   e.g. dataset/23_Aditya_Kumar/1.jpg
python main.py enroll-bulk --dataset dataset --class 10A --consent
python main.py list-students --class 10A
python main.py delete-student --roll 23          # right to erasure
```

## 3. Usage

### CLI
```bash
python main.py --video class.mp4 --class 10A --subject Maths
python main.py --video cctv_4k.mp4 --class 10A --subject Maths --tiled          # small back-bench faces
python main.py process --source "rtsp://user:pass@192.168.1.20:554/stream1" --class 10A --subject Physics --max-minutes 50
python main.py process --source 0 --class 10A --subject Demo                    # webcam
```
Output goes to `outputs/<date>_<class>_<subject>/`:
`Attendance_<Class>_<Subject>_<YYYY-MM-DD_HH-MM>.xlsx`, a `.csv` and `.json` with the same name,
`annotated.mp4`, `snapshots/` and `Unknown/` (cluster thumbnails).

Corrections after processing (each one is logged in the audit trail and the report is regenerated):
```bash
python main.py sessions
python main.py override --session 3 --roll 23 --status Present --reason "camera blind spot"
python main.py unknowns --session 3
python main.py assign-unknown --session 3 --cluster 2 --roll 23 --update-gallery
python main.py report --session 3 [--no-images]
python main.py notify --session 3
python main.py cleanup                                   # apply retention policy now
```

### Web app
```bash
streamlit run app/streamlit_app.py
```
The first visit asks you to create the admin account. You can also create one with `python main.py create-admin`.

| Page | What it does |
|---|---|
| **Enroll Students** | Single enrollment (photos or selfie video, consent checkbox), bulk enrollment from a folder, student list, right-to-erasure delete |
| **Upload Video / Connect CCTV** | Upload a file or enter an RTSP URL or webcam index. Set frame skip, threshold, votes, tiling and upscale |
| **Live Processing View** | Live annotated frames, progress bar with ETA, live Present X / Y counter, Stop button (partial results are saved) |
| **Results & Download** | Editable status column with audit-logged overrides; Download Excel / CSV / JSON / Annotated Video; review unknown faces and assign them to students; duplicate/spoof flags; notifications |
| **History & Analytics** | Attendance trend over time, chronically absent students (< 75 %), per-student bars, class-wise comparison, full audit log |

**Screenshots:** put your own captures in `docs/screenshots/` (for example `live.png` and `results.png`).
The pages are listed in the table above. The annotated video shows green boxes labelled `Name (Roll)` for
recognised students, red `Unknown #id` boxes, orange `[DUP?]` boxes for flagged duplicates, and a
`Present: X / Y  t=HH:MM:SS` banner.

### Excel report
| Sheet | Contents |
|---|---|
| Attendance | Roll No, Name, Class, Status, First Seen, Last Seen, Total Time Visible, Confidence Score, Remarks, Face Snapshot (embedded). Rows are green for Present, amber for Late and red for Absent. The header row is frozen, columns are auto-sized and a filter is applied |
| Summary | Date, class, subject, source, duration, enrolled / present / late / absent / excused counts, attendance %, unknown faces, duplicate flags, pie chart, periodic headcount table, flag details |
| Unknown Faces | Cluster ID, Track ID, timestamps, assigned student, thumbnail |
| History | Cumulative attendance per student across sessions. Students below 75 % are highlighted in red |

## 4. Evaluation
Create a CSV with your manual labels for a session (`roll_no,present`, where present is 1/0, yes/no or present/absent):
```bash
python evaluate.py --truth data/ground_truth_example.csv --session 3
python evaluate.py --truth gt.csv --pred outputs/<session>/Attendance_10A_Maths_....csv --out metrics.json
```
The script prints TP/FP/FN/TN, precision, recall, accuracy and F1, plus the lists of false positives and missed students.

## 5. Tests
```bash
python -m pytest -q
```
The tests cover the voting and de-duplication rules (re-entry, simultaneous duplicates, blocked tracks,
late confirmation), Excel structure and formatting, the encrypted DB, ByteTrack, the gallery, the metrics and
the CLI parser. They also include a full pipeline run on a synthetic video using a fake face engine, so no
models are needed. `tests/test_behaviour.py` covers smoothing, the scoring formula, every behaviour rule,
solvePnP sign conventions, EAR/MAR, object and person association, privacy filtering, teacher corrections,
the behaviour workbook and the metrics. It also includes a sample-video integration test that runs behaviour
mode, checks the outputs and the cache, re-scores the session and applies a correction.

## 6. How duplicate marking is prevented
1. **Track-level voting.** ByteTrack gives each face a stable ID. Each quality-filtered crop of that track
   is embedded and votes for its best gallery match, but only if cosine ≥ `similarity_threshold` and the
   match beats the runner-up by `margin`. A track becomes a student only after `min_votes` (default 5) matching
   crops, or 2 or more matches spanning `min_seconds`. A single lucky frame can never mark anyone.
2. **Session-level set.** `AttendanceSession.marked` is a dict keyed by roll number. A student is added to it
   once. A later track of the same student (they left and came back, or tracking broke) only extends First/Last
   Seen and visible time, which is the union of intervals, so overlaps are never double-counted.
3. **Simultaneous conflict.** If two tracks that are visible in the same frame both resolve to the same
   student, this is physically impossible. It is logged as a possible spoof or duplicate, for example a phone
   photo held up by a classmate. The higher-confidence track keeps the identity, and the other is blocked
   from ever claiming that student again. Tracks that are only confirmed when they end are checked against
   the time segments of the current owner as well.
4. **Persistence.** The `attendance` table has a primary key of `(session_id, roll_no)`, so the database also
   cannot hold two rows for one student in one session.

## 7. Privacy & ethics (DPDP Act, 2023)
- Only face embeddings are stored, never enrollment photos. Uploaded videos are deleted after processing
  unless `privacy.retain_raw_video: true`. In that case they are auto-deleted after `raw_video_retention_days`.
  Session outputs are deleted after `output_retention_days`.
- Names, guardian contacts and every embedding are encrypted at rest with Fernet (AES-128 plus HMAC).
- The web app requires an admin login. Passwords are hashed with PBKDF2-SHA256 at 200k iterations.
- Consent is recorded with a date at enrollment, and students can request erasure (`delete-student`).
- No face data is ever sent to a third-party API. Notifications are text-only by default, and the e-mailed
  Excel file has its thumbnails stripped unless `notify.email.attach_snapshots: true`. Streamlit telemetry is disabled.

## 8. Class Attentiveness & Behaviour Analytics

> **Responsible use.** The scores are *indicative estimates of observable behaviour*, meant to help teachers
> improve their teaching. They must **not** be used for punishment or disciplinary action. SmartAttend does
> **no emotion recognition** and makes no inference about personality or intelligence. This notice appears in
> the UI, the Excel report and the CLI output.

### Architecture
```
                         ┌──────────── attendance pipeline (unchanged) ────────────┐
video ─► frame skip ─► SCRFD faces ─► ByteTrack IDs ─► ArcFace voting ─► Roll No per track
                │                          │
                │ every Nth processed frame│ track id + face box + 5 kps
                ▼                          ▼
   YOLO11-pose (17 kps)   InsightFace 1k3d68 landmarks ──► solvePnP yaw/pitch/roll, EAR, MAR
   YOLO11 objects ──► phone/book/laptop linked to the nearest wrist / desk region
                │
                ▼  FrameSignals per track  ──► encrypted Parquet cache (re-scorable)
   rule-based BehaviourClassifier (one primary state per frame, priority order)
                ▼
   centred majority smoothing (2.5 s) ─► teacher corrections ─► gaps > 4 s = out of frame
                ▼
   events · per-frame score · session score · 1-min bins · class curve · drop points · movement metrics
                ▼
   Behaviour_<Class>_<Subject>_<date>.xlsx · CSV/JSON · encrypted DB result · Streamlit tab
   live overlay ─► behaviour_full.mp4 + behaviour_preview.mp4 (720p, H.264 when available)
```
Every behaviour is linked to a recognised student through the track ID. The student's Roll No comes from the
attendance voting. A student's tracks are merged into one timeline. Tracks that are never recognised are
analysed as `Unknown_<trackid>` and are excluded from the class score by default (the teacher is usually one of them).

### Setup
`pip install -r requirements.txt` also installs `ultralytics` and `pyarrow`. On first use the YOLO11 weights
(`yolo11n-pose.pt`, `yolo11n.pt`, ~12 MB in total) are downloaded to `data/models/`. After that, everything runs locally.
If Ultralytics is not installed, the module still works in face-only mode (head pose, eyes and mouth), but hand raises,
phones and posture changes are not detected.

### Run
```bash
python main.py --video class.mp4 --class 10A --subject Maths --mode attendance+behaviour [--camera room101] [--class-level-only]
python main.py behaviour-summary  --session 7
python main.py behaviour-rescore  --session 7                       # after editing weights/thresholds in config.yaml
python main.py behaviour-correct  --session 7 --subject 23 --start 120 --end 135 --behaviour phone --to writing --reason "calculator"
python main.py set-behaviour-consent --roll 23 --value yes          # guardian consent (students are minors)
python main.py create-admin --username ms_rao --role teacher        # roles: admin | teacher | staff
python evaluate_behaviour.py --truth behaviour_labels.csv --session 7
```
In the web app, choose **Mode: Attendance + Behaviour** on the *Upload Video / Connect CCTV* page, then open
**Attentiveness & Behaviour**. That tab offers:
- **Class view:** the attention curve with drop markers, a donut of behaviour time, a seating heatmap drawn over a
  frame from the camera, and students ranked by score.
- **Student view:** the student's timeline, a Gantt-style behaviour timeline, a radar profile and a movement trail.
  Selecting an event row jumps the video player to that time, and from there you can mark the event as wrong.
- **Compare:** 2–5 students side by side, plus trends across sessions.
- **Video & downloads:** the annotated video player and download buttons.
- **Corrections:** the list of corrections, with undo, and a re-score button.
- **Board zone setup:** admin only.

Filters let you narrow by time range, behaviour type and minimum score. A toggle switches to **class-level only**.
Screenshots: save captures of the class view, student view and behaviour video to `docs/screenshots/behaviour_*.png`.

### Drawing the board / teacher zone
Go to *Attentiveness & Behaviour → Board zone setup* (admin only). Enter the camera/room id, upload a frame or
short clip from that camera, and draw around the board with the **lasso** (or box) tool. Then press *Save zone*.
The polygon is stored in normalised coordinates in `data/camera_zones.yaml`. You can also add zones inline under
`behaviour.board.zones` in config.yaml, or type the points by hand. Choose the same camera id when processing.

If a camera has no zone, *facing the camera* counts as looking at the board (|yaw| ≤ 25° and pitch between −25° and 15°).
That fits the usual set-up, with the CCTV camera mounted above or next to the board. With a zone, the 2D
head-direction ray from the face must hit the polygon. Re-score old sessions after changing a zone.

### Behaviour logic (all thresholds in `config.yaml → behaviour.rules`)
| State | Rule |
|---|---|
| Looking at board | head direction hits the board zone (or faces the camera) |
| Looking down / writing | pitch ≥ 20° and (a wrist below shoulder level, or a book/notebook/pen/laptop is linked). With no body pose, confidence is ×0.5 |
| Looking away / side | \|yaw\| > 35° continuously for ≥ 2 s |
| Head down / sleeping | pitch ≥ 35° and EAR < 0.18 (or pitch ≥ 50° if the eyes are not measurable) for ≥ 3 s |
| Talking | MAR std ≥ 0.04 over 2 s and the head turned ≥ 15° towards a neighbour |
| Hand raised | a wrist keypoint above the nose (or well above the shoulders) |
| Using phone | a phone is detected within 1.5 face heights of the student's wrist or desk region |
| Standing / moving | shoulders rise > 0.6 shoulder widths above the seated baseline (or the face rises > 1 face height), or the face moves > 1.5 face heights/s |
| Out of frame / absent | the track is unseen for > 4 s |

When several rules fire, the first one in `priority` wins (phone › sleeping › hand › standing › talking › away ›
writing › board › other). The per-frame states are then smoothed with a **centred 2.5 s majority vote**, so flickers
are never counted as behaviour changes. Each event stores the Roll No, Name, behaviour, start, end, duration,
confidence and frame range. If the face is not visible, the student is followed by body pose for up to 30 s with
confidence ×0.5. The 5-point head-pose fallback, used for faces under 24 px, applies a factor of ×0.8.

To replace the rules with a trained model (an LSTM or Transformer on keypoint sequences, SlowFast, VideoMAE, ...),
implement `BehaviourClassifier.classify/reset` and register it in `build_classifier`. Then set `behaviour.classifier`.

### Scoring formula
* **Per-frame score:** `score = clip(50 + 50·w(state), 0, 100)`. The default weights `w` are
  board +1.0, writing +0.9, hand +1.0, away −0.5, talking −0.4, phone −1.0, sleeping −1.0, standing 0,
  other 0, and out-of-frame `null` (not scored).
* **Session score:** `S = Σ score_i·dt_i·conf_i / Σ dt_i·conf_i`. This is a time- and confidence-weighted
  average, where `dt` is the time until the next processed frame. Labels: Highly Attentive ≥ 75, Attentive 50–74,
  Partially Attentive 30–49, Inattentive < 30.
* **Class curve:** for each 60 s bin, the mean of the bin scores of all present students. An **engagement drop**
  is a bin whose score is ≥ 15 points below the mean of the previous 3 bins.
* **Movement metrics:** face-centre path length (in pixels and face heights), posture changes (state transitions),
  stand-ups, head turns (|yaw| crossing 35° with 10° hysteresis), and time spent in each behaviour.
* Teacher corrections replace the corrected samples (confidence 1.0) and are re-scored from the cache. Each one is
  logged in the audit trail.

### Behaviour report (`Behaviour_<Class>_<Subject>_<YYYY-MM-DD_HH-MM>.xlsx`)
| Sheet | Contents |
|---|---|
| Class Summary | disclaimer, session details, class score and label, attentive vs inattentive counts, label bar chart, behaviour doughnut, most common distractions, drop points, top 5 / bottom 5 |
| Student Scores | one row per student: attendance, score with a green→red colour scale, label, confidence, minutes and % per behaviour, hand raises, phone uses, talking, sleep, stand-ups, head turns, posture changes, displacement |
| Behaviour Log | every event with filters: Roll No, Name, Behaviour, Start, End, Duration, Confidence, frames, corrected flag |
| Class Timeline | per-minute class score with a line chart and drop flags |
| Student Timelines | students × minutes matrix with heat-map formatting |
| Movement Data | optional (`report.include_movement_sheet`): per-second yaw, pitch, position, state and score |

CSV files (`_students.csv`, `_events.csv`, `_class_timeline.csv`) and a JSON file with the same data are written
next to the workbook. The annotated videos are `behaviour_full.mp4` and `behaviour_preview.mp4`. They show boxes
coloured by state (green ≥ 50, yellow 30–49, red < 30), the label `RollNo | Name | State | Score`, the skeleton,
a head-direction arrow, event pop-ups (`[HAND] Hand raised`, `[PHONE] Phone detected`, ...), a HUD with the time,
Present X/Y, live class attention % and a sparkline, and a live distraction panel.

### Behaviour privacy controls (implemented)
- **Role-based access:** admins can do everything. Teachers get attendance plus analytics. Staff get attendance
  only (`behaviour.allowed_roles`). Admins manage users on the **Admin** page.
- **Separate guardian consent** is required for per-student analytics (`set-behaviour-consent`, or the enrollment
  form). Students without it are counted only in class aggregates and appear as "Student" on the video.
- **Class-level only mode** (`behaviour.privacy.class_level_only`, `--class-level-only`, or the UI toggle) stores and
  shows aggregates only.
- The signal cache and the stored results are **Fernet-encrypted**. Session folders follow `privacy.output_retention_days`.

### Evaluating behaviours
Label some clips by hand in a CSV with the columns `roll_no,start,end,behaviour`. Times can be seconds or HH:MM:SS,
and behaviours use keys such as `looking_board`, `writing` or `phone`. Then run:
`python evaluate_behaviour.py --truth labels.csv --session N` (or pass `--pred <events.csv>`).
The script reports per-behaviour precision, recall and F1, accuracy, macro-F1 and the confusion matrix, computed
on 1-second steps.

## 9. Known limitations
- **Accuracy depends on the camera.** Faces below about 20 px, heavy motion blur or students looking down for the
  whole class will be missed. Use `--tiled`, `upscale_factor` or a camera placed at the front of the room.
- **Anti-spoofing is basic.** The built-in heuristic (moiré, saturation and glare cues) catches only obvious photos
  and screens. For real deployments, plug in a trained model through `liveness.model_path`. The simultaneous-duplicate
  rule still catches the common "friend holds up my photo" case.
- **Masks and occlusion.** `occlusion.enabled` relaxes the threshold but requires more votes. ArcFace was not trained
  on masked faces, so recall drops noticeably.
- **Identical twins** and students who look very much alike may be confused. The `margin` setting reduces this,
  and the teacher can override the result.
- **Throughput.** CPU-only processing runs at about 3–8 processed frames per second at 1080p (more with `frame_skip`).
  Tiled 4K detection is several times slower, so use a GPU for live multi-camera use.
- **Video codecs.** On Windows, OpenCV may print `libopenh264` errors and then fall back to the system H.264
  encoder (or `mp4v`). This is harmless. Browsers may not play `mp4v` files inline, but downloading them still works.
- **Late and left-early times** are relative to the start of the video (`late.class_start_offset_sec`). They are not wall-clock times.
- **Losing the encryption key** makes the database unreadable, so back it up.
- **Behaviour analytics** use rule-based heuristics and have not been validated on your classrooms. Measure them
  with `evaluate_behaviour.py` before relying on them. Head pose from small or blurred faces is noisy.
  "Talking" (mouth movement plus a turned head) and "writing" (head down plus hands low) are proxies, not
  certainties. Occluded students fall back to body pose with lower confidence.
- COCO has no `pen` class. Pens and notebooks need a custom-trained detector (`behaviour.objects.model`).
  The default model detects phones, books and laptops. Small phones held under desks are often missed.
- Event pop-ups use ASCII tags such as `[HAND]`, because OpenCV's fonts cannot draw emoji.
- The live overlay uses causal (trailing) smoothing. The report and the dashboard use the centred window, so the
  video can switch states up to about 1 s later than the log.
- Seating positions come from the median face position in the frame, not from a calibrated seating plan.
# ClassSightAI
