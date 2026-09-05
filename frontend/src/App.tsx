// @ts-nocheck
import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";
import "./theme.css";
import "./App.css";
import "./Pages.css";
import "./Video.css";
import "./Readability.css";

type View = "overview" | "history" | "fences";
type Detection = {
  class: string;
  score: number;
  bbox: [number, number, number, number];
};
type ObjectSummary = { seen: number; peak: number; frames: number; confidence: number };
type Track = { id: number; class: string; bbox: [number, number, number, number]; missed: number; confidenceTotal: number; hits: number };
type BackendSummary = { unique: number; peak: number; observations: number; confidence: number };
type SuspiciousEvent = {
  type: "loitering" | "fast_movement" | "after_hours_activity" | string;
  object_class: string;
  track_id: number | null;
  duration_seconds: number | null;
  detail: string;
};
type BackendResult = {
  objects: Record<string, BackendSummary>;
  frames: number;
  raw_detections: number;
  suspicious_events: SuspiciousEvent[];
  analyzed_at: string;
  restricted_hours_active: boolean;
};
type Alert = {
  id: number;
  level: string;
  title: string;
  detail: string;
  camera: string;
  time: string;
  acknowledged: boolean;
};

const zoneRules = ["Loitering beyond 20 seconds", "Activity during restricted hours", "Rapid or erratic movement"];
// Vehicles, people and common animals — matches the classes the backend (YOLOv8 / COCO) can also report.
const ALLOWED_CLASSES = new Set([
  "person",
  "car",
  "truck",
  "bus",
  "motorcycle",
  "bicycle",
  "dog",
  "cat",
  "bird",
  "horse",
  "cow",
  "sheep",
  "bear",
  "elephant",
  "zebra",
  "giraffe",
]);
const DETECTION_CONFIDENCE = 0.4;
const NMS_IOU_THRESHOLD = 0.3;
const TRACK_IOU_THRESHOLD = 0.15;
const TRACK_MAX_MISSED_FRAMES = 15;
const SAMPLE_INTERVAL_MS = 100;
const BACKEND_URL = "http://localhost:8000";

const BEHAVIOR_TITLES: Record<string, string> = {
  loitering: "Loitering detected",
  fast_movement: "Unusually fast movement",
  after_hours_activity: "Activity during restricted hours",
};

const BEHAVIOR_LEVELS: Record<string, string> = {
  loitering: "HIGH",
  fast_movement: "HIGH",
  after_hours_activity: "CRITICAL",
};

function behaviorTitle(event: SuspiciousEvent) {
  return BEHAVIOR_TITLES[event.type] ?? `Unusual behavior: ${event.type}`;
}

function behaviorLevel(event: SuspiciousEvent) {
  return BEHAVIOR_LEVELS[event.type] ?? "HIGH";
}

function intersectionOverUnion(first: [number, number, number, number], second: [number, number, number, number]) {
  const left = Math.max(first[0], second[0]);
  const top = Math.max(first[1], second[1]);
  const right = Math.min(first[0] + first[2], second[0] + second[2]);
  const bottom = Math.min(first[1] + first[3], second[1] + second[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = first[2] * first[3] + second[2] * second[3] - intersection;
  return union > 0 ? intersection / union : 0;
}

function suppressDuplicateDetections(results: cocoSsd.DetectedObject[]) {
  return results.filter((candidate, index) =>
    results.every(
      (other, otherIndex) =>
        otherIndex === index ||
        other.class !== candidate.class ||
        other.score >= candidate.score ||
        intersectionOverUnion(candidate.bbox as [number, number, number, number], other.bbox as [number, number, number, number]) < NMS_IOU_THRESHOLD
    )
  );
}

const KNOWN_PATHS = ["/", "/history", "/fences"];

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const view: View = location.pathname === "/history" ? "history" : location.pathname === "/fences" ? "fences" : "overview";

  useEffect(() => {
    if (!KNOWN_PATHS.includes(location.pathname)) {
      navigate("/", { replace: true });
    }
  }, [location.pathname, navigate]);

  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [videoUrl, setVideoUrl] = useState("");
  const [fileName, setFileName] = useState("No source loaded");
  const [detections, setDetections] = useState<Detection[]>([]);
  const [uniqueObjectCount, setUniqueObjectCount] = useState(0);
  const [backendResult, setBackendResult] = useState<BackendResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState("");
  const [backendAnalyzing, setBackendAnalyzing] = useState(false);
  const [nightMode, setNightMode] = useState(false);
  const [armed, setArmed] = useState<boolean[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const timerRef = useRef<number | undefined>(undefined);
  const tracksRef = useRef<Track[]>([]);
  const nextTrackIdRef = useRef(1);
  const summaryRef = useRef<Record<string, ObjectSummary>>({});
  const analysisCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const sourceFileRef = useRef<File | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  // Canvas Overlay Redraw Hook
  useEffect(() => {
    const video = videoRef.current;
    const overlay = overlayCanvasRef.current;
    if (!video || !overlay) return;

    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const scaleX = video.clientWidth / video.videoWidth;
    const scaleY = video.clientHeight / video.videoHeight;

    // Historical trail tracks
    tracksRef.current.forEach((track) => {
      if (track.missed > 0) return;

      const [x, y, width, height] = track.bbox;
      const scaledX = x * scaleX;
      const scaledY = y * scaleY;
      const scaledW = width * scaleX;
      const scaledH = height * scaleY;

      const shrinkFactor = 0.65;
      const smallW = scaledW * shrinkFactor;
      const smallH = scaledH * shrinkFactor;
      const smallX = scaledX + (scaledW - smallW) / 2;
      const smallY = scaledY + (scaledH - smallH) / 2;

      ctx.strokeStyle = "rgba(63, 169, 201, 0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(smallX, smallY, smallW, smallH);
      ctx.setLineDash([]);
    });

    // Current tight bounding boxes
    detections.forEach((detection) => {
      const [x, y, width, height] = detection.bbox;

      const inset = 0.05;
      const tightX = (x + width * inset) * scaleX;
      const tightY = (y + height * inset) * scaleY;
      const tightW = width * (1 - inset * 2) * scaleX;
      const tightH = height * (1 - inset * 2) * scaleY;

      ctx.strokeStyle = "#7fd6ef";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(tightX, tightY, tightW, tightH);

      ctx.fillStyle = "#7fd6ef";
      ctx.font = "10px monospace";
      ctx.fillText(
        `${detection.class} ${Math.round(detection.score * 100)}%`,
        tightX,
        tightY > 12 ? tightY - 4 : tightY + 10
      );
    });
  }, [detections]);

  const loadVideo = (file: File) => {
    if (!file.type.startsWith("video/")) {
      setAnalysisError("Please choose a video file.");
      return;
    }
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(file));
    setFileName(file.name);
    setDetections([]);
    setUniqueObjectCount(0);
    setBackendResult(null);
    summaryRef.current = {};
    tracksRef.current = [];
    nextTrackIdRef.current = 1;
    setAlerts([]);
    setArmed([true, true, true]);
    setAnalysisError("");
    sourceFileRef.current = file;
  };

  const updateObjectSummary = (results: cocoSsd.DetectedObject[]) => {
    const validResults = results.filter(
      (res) => ALLOWED_CLASSES.has(res.class) && res.score >= DETECTION_CONFIDENCE
    );

    const tracks = tracksRef.current.map((track) => ({ ...track, missed: track.missed + 1 }));
    const matched = new Set<number>();
    const currentCounts: Record<string, number> = {};

    validResults.forEach((result) => {
      const bbox = result.bbox as [number, number, number, number];
      let bestIndex = -1;
      let bestIou = TRACK_IOU_THRESHOLD;

      tracks.forEach((track, index) => {
        if (matched.has(index) || track.class !== result.class || track.missed > TRACK_MAX_MISSED_FRAMES) return;
        const iou = intersectionOverUnion(track.bbox, bbox);
        if (iou >= bestIou) {
          bestIndex = index;
          bestIou = iou;
        }
      });

      if (bestIndex >= 0) {
        tracks[bestIndex] = {
          ...tracks[bestIndex],
          bbox,
          missed: 0,
          confidenceTotal: tracks[bestIndex].confidenceTotal + result.score,
          hits: tracks[bestIndex].hits + 1,
        };
        matched.add(bestIndex);
      } else {
        tracks.push({
          id: nextTrackIdRef.current++,
          class: result.class,
          bbox,
          missed: 0,
          confidenceTotal: result.score,
          hits: 1,
        });
      }

      currentCounts[result.class] = (currentCounts[result.class] ?? 0) + 1;
    });

    tracksRef.current = tracks.filter((track) => track.missed <= TRACK_MAX_MISSED_FRAMES);

    const totalUniqueObjects = nextTrackIdRef.current - 1;
    setUniqueObjectCount(totalUniqueObjects);

    if (validResults.length > 0) {
      const newAlerts: Alert[] = Object.entries(currentCounts).map(([objectClass, count], index) => {
        const classTracks = tracksRef.current.filter((t) => t.class === objectClass && t.missed === 0);
        const avgScore = classTracks.length > 0
          ? classTracks.reduce((sum, t) => sum + (t.confidenceTotal / t.hits), 0) / classTracks.length
          : 0.85;

        return {
          id: index + 1,
          level: "MEDIUM",
          title: `${count} ${objectClass}${count > 1 ? "s" : ""} detected on screen`,
          detail: `${totalUniqueObjects} total unique objects logged in stream · ${Math.round(avgScore * 100)}% confidence`,
          camera: fileName || "01.mp4",
          time: new Date().toLocaleTimeString("en-GB", { hour12: false }),
          acknowledged: false,
        };
      });
      setAlerts(newAlerts);
    }
  };

  const analyzeFrame = async () => {
    const video = videoRef.current;
    if (!video || video.paused || video.ended || video.readyState < 2 || analyzing) return;
    setAnalyzing(true);
    try {
      modelRef.current ??= await cocoSsd.load({ base: "mobilenet_v2" });
      const canvas = analysisCanvasRef.current ?? document.createElement("canvas");
      analysisCanvasRef.current = canvas;
      const scale = Math.min(2.5, 1920 / video.videoWidth);
      canvas.width = Math.max(video.videoWidth, Math.round(video.videoWidth * scale));
      canvas.height = Math.max(video.videoHeight, Math.round(video.videoHeight * scale));

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }

      const results = suppressDuplicateDetections(await modelRef.current.detect(canvas, 100, DETECTION_CONFIDENCE));
      const validDetections = results
        .filter((item) => ALLOWED_CLASSES.has(item.class))
        .map((item) => ({
          class: item.class,
          score: item.score,
          bbox: item.bbox as [number, number, number, number],
        }));

      setDetections(validDetections);
      updateObjectSummary(results);
    } catch (error) {
      console.error("Video analysis failed", error);
      setAnalysisError(error instanceof Error ? error.message : "The video could not be analyzed.");
      if (timerRef.current) window.clearTimeout(timerRef.current);
    } finally {
      setAnalyzing(false);
    }
    if (video && !video.paused && !video.ended) timerRef.current = window.setTimeout(analyzeFrame, SAMPLE_INTERVAL_MS);
  };

  const runDetection = async () => {
    const video = videoRef.current;
    if (!video) return;
    setAnalysisError("");
    setAlerts([]);
    setUniqueObjectCount(0);
    summaryRef.current = {};
    tracksRef.current = [];
    nextTrackIdRef.current = 1;
    const sourceFile = sourceFileRef.current;
    if (sourceFile) {
      setBackendAnalyzing(true);
      try {
        const body = new FormData();
        body.append("file", sourceFile);
        const response = await fetch(`${BACKEND_URL}/analyze`, { method: "POST", body });
        if (!response.ok) throw new Error((await response.json()).detail ?? "Backend analysis failed.");
        const result = (await response.json()) as BackendResult;
        setBackendResult(result);
        setUniqueObjectCount(Object.values(result.objects).reduce((total, item) => total + item.unique, 0));

        const objectAlerts: Alert[] = Object.entries(result.objects).map(([objectClass, summary], index) => ({
          id: index + 1,
          level: "MEDIUM",
          title: `${summary.unique} ${objectClass}${summary.unique === 1 ? "" : "s"} detected in entire video`,
          detail: `${summary.unique} unique tracked · peak ${summary.peak} visible · ${Math.round(summary.confidence * 100)}% confidence · ${summary.observations} raw boxes`,
          camera: sourceFile.name,
          time: new Date().toLocaleTimeString("en-GB", { hour12: false }),
          acknowledged: false,
        }));

        const behaviorAlerts: Alert[] = (result.suspicious_events ?? []).map((event, index) => ({
          id: objectAlerts.length + index + 1,
          level: behaviorLevel(event),
          title: behaviorTitle(event),
          detail: event.detail,
          camera: sourceFile.name,
          time: new Date().toLocaleTimeString("en-GB", { hour12: false }),
          acknowledged: false,
        }));

        setAlerts([...behaviorAlerts, ...objectAlerts]);
      } catch (error) {
        setAnalysisError(error instanceof Error ? `${error.message} Start the backend with: uvicorn backend.main:app --reload --port 8000` : "Backend analysis failed.");
        return;
      } finally {
        setBackendAnalyzing(false);
      }
    }
    if (video.ended) video.currentTime = 0;
    try {
      if (video.readyState < 2)
        await new Promise<void>((resolve, reject) => {
          const onReady = () => {
            video.removeEventListener("loadeddata", onReady);
            resolve();
          };
          video.addEventListener("loadeddata", onReady, { once: true });
          video.addEventListener("error", () => reject(new Error("This video format could not be decoded.")), { once: true });
        });
      await video.play();
      analyzeFrame();
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : "The video could not be played.");
    }
  };

  const acknowledge = (id: number) => setAlerts((items) => items.map((item) => (item.id === id ? { ...item, acknowledged: true } : item)));

  const detectionCount = detections.length;
  const behaviorFlagCount = alerts.filter((alert) => alert.level === "CRITICAL" || alert.level === "HIGH").length;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">◈</span>
          <span>
            <b>NIGHTWATCH</b>
            <small>DEEP FIELD / SECTOR 07</small>
          </span>
        </div>
        <div className="side-label">OPERATIONS DESK</div>
        <nav>
          <NavButton to="/" icon="▦" label="Command overview" number="01" />
          <NavButton to="/history" icon="♧" label="Event history" number="02" />
          <NavButton to="/fences" icon="◇" label="Virtual fences" number="03" />
        </nav>
        <div className="sidebar-divider" />
        <div className="side-label">SYSTEM</div>
        <div className="system-row">
          <span className="signal">◉</span> Edge relay <b>● OK</b>
        </div>
        <div className="system-row">
          <span className="bars">▂▅▇</span> Uplink <b className="amber">84%</b>
        </div>
        <div className="operator">
          <span className="avatar">AR</span>
          <span>
            <b>A. Rawat</b>
            <small>SENTINEL / SHIFT B</small>
            <small>◷ Shift ends 06:00 UTC</small>
          </span>
        </div>
      </aside>
      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">
              <span className="pulse" />{" "}
              {view === "overview" ? "LIVE PERIMETER / COMMAND OVERVIEW" : view === "history" ? "AUDIT TRAIL / EVENT HISTORY" : "PERIMETER CONTROLS / VIRTUAL FENCES"}
            </div>
            <h1>{view === "overview" ? "Watch the line." : view === "history" ? "Every signal, accounted for." : "Hold the boundary."}</h1>
            <p>
              {view === "overview"
                ? "Upload a video for full-video detection and behavior analysis, or preview it live in the browser."
                : view === "history"
                ? "Review confidence, source video, and operator acknowledgement for every event."
                : "Arm or disarm detection rules at the edge. Changes are applied on confirmation."}
            </p>
          </div>
          <div className="top-actions">
            <div>
              <small>ZULU TIME</small>
              <strong>{new Date().toLocaleTimeString("en-GB", { hour12: false })}</strong>
            </div>
            <div className="live" role="status">
              <span className="pulse" /> {analyzing || backendAnalyzing ? "ANALYZING" : "LIVE FEED"}
            </div>
          </div>
        </header>
        {view === "overview" && (
          <Overview
            videoUrl={videoUrl}
            fileName={fileName}
            videoRef={videoRef}
            overlayCanvasRef={overlayCanvasRef}
            loadVideo={loadVideo}
            runDetection={runDetection}
            detections={detections}
            uniqueObjectCount={uniqueObjectCount}
            detectionCount={detectionCount}
            behaviorFlagCount={behaviorFlagCount}
            alerts={alerts}
            acknowledge={acknowledge}
            nightMode={nightMode}
            setNightMode={setNightMode}
            analysisError={analysisError}
            setAnalysisError={setAnalysisError}
            backendResult={backendResult}
            analyzing={analyzing}
            backendAnalyzing={backendAnalyzing}
          />
        )}
        {view === "history" && <History alerts={alerts} acknowledge={acknowledge} />}
        {view === "fences" && <Fences armed={armed} setArmed={setArmed} alerts={alerts} sourceName={videoUrl ? fileName : ""} />}
      </main>
    </div>
  );
}

function NavButton({ to, icon, label, number }: { to: string; icon: string; label: string; number: string }) {
  const location = useLocation();
  const active = location.pathname === to;
  return (
    <Link to={to} className={`nav-item ${active ? "active" : ""}`}>
      <span>{icon}</span>
      {label}
      <em>{number}</em>
    </Link>
  );
}

function Stat({ label, value, note, tone, icon }: { label: string; value: string; note: string; tone: string; icon: string }) {
  return (
    <div className="stat-card">
      <div className="stat-label">
        {label}
        <span className={`stat-icon ${tone}`}>{icon}</span>
      </div>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}

function Overview({
  videoUrl,
  fileName,
  videoRef,
  overlayCanvasRef,
  loadVideo,
  runDetection,
  uniqueObjectCount,
  detectionCount,
  behaviorFlagCount,
  alerts,
  acknowledge,
  nightMode,
  setNightMode,
  analysisError,
  setAnalysisError,
  backendResult,
  analyzing,
  backendAnalyzing,
}: {
  videoUrl: string;
  fileName: string;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  overlayCanvasRef: React.RefObject<HTMLCanvasElement | null>;
  loadVideo: (file: File) => void;
  runDetection: () => void;
  detections: Detection[];
  uniqueObjectCount: number;
  detectionCount: number;
  behaviorFlagCount: number;
  alerts: Alert[];
  acknowledge: (id: number) => void;
  nightMode: boolean;
  setNightMode: (value: boolean) => void;
  analysisError: string;
  setAnalysisError: (value: string) => void;
  backendResult: BackendResult | null;
  analyzing: boolean;
  backendAnalyzing: boolean;
}) {
  return (
    <>
      <section className="upload-bar">
        <div>
          <b>VIDEO ANALYSIS SOURCE</b>
          <small>{fileName}</small>
        </div>
        <label className="upload-button">
          CHOOSE VIDEO
          <input type="file" accept="video/mp4,video/webm,video/quicktime" onChange={(event) => event.target.files?.[0] && loadVideo(event.target.files[0])} />
        </label>
        {videoUrl && (
          <button className="analyze-button" onClick={runDetection}>
            RUN DETECTION
          </button>
        )}
      </section>
      {analysisError && <div className="analysis-error" role="alert">Analysis unavailable: {analysisError}</div>}
      {backendResult?.restricted_hours_active && (
        <div className="analysis-error" role="status" style={{ background: "var(--high-soft)", borderColor: "var(--high)", color: "var(--high)" }}>
          This video was analyzed during the restricted hours window — any person activity was flagged for review.
        </div>
      )}
      <section className="stats-grid">
        <Stat
          label="OBJECTS IN ENTIRE VIDEO"
          value={String(uniqueObjectCount)}
          note={videoUrl ? `${detectionCount} active on screen` : "Awaiting video source"}
          tone="green"
          icon="◉"
        />
        <Stat
          label="UNRESOLVED ALERTS"
          value={String(alerts.filter((a) => !a.acknowledged).length)}
          note="Requires operator review"
          tone="red"
          icon="♧"
        />
        <Stat
          label="EVENTS TODAY"
          value={String(alerts.length)}
          note={videoUrl ? "Live object-class summaries" : "No video analyzed"}
          tone="amber"
          icon="⌁"
        />
        <Stat
          label="BEHAVIOR FLAGS"
          value={String(behaviorFlagCount)}
          note="Loitering, speed, off-hours"
          tone="blue"
          icon="◔"
        />
      </section>
      <section className="panel camera-panel">
        <div className="panel-header">
          <div>
            <h2>
              Camera wall <span className="count-badge">{videoUrl ? "uploaded source" : "awaiting source"}</span>
            </h2>
            <p>{videoUrl ? "YOLOv8 + BoT-SORT / full video analysis" : "Legacy streams / edge-assisted detection"}</p>
          </div>
          <button className="tool" onClick={() => setNightMode(!nightMode)}>
            ☾ {nightMode ? "LOW-LIGHT ON" : "LOW-LIGHT"}
          </button>
        </div>
        <div className="camera-grid">
          {(videoUrl ? [fileName] : []).map((camera, index) => (
            <div className={`camera-card ${videoUrl && index === 0 ? "selected" : ""}`} key={camera}>
              <div className="camera-meta">
                <b>{camera}</b>
                <span className="live-tag">● {analyzing || backendAnalyzing ? "ANALYZING" : "READY"}</span>
              </div>
              <small>Local upload / browser inference</small>
              <div className={`feed-visual feed-${index} ${nightMode ? "night" : ""}`}>
                {videoUrl && index === 0 ? (
                  <>
                    <video ref={videoRef} src={videoUrl} controls muted onError={() => setAnalysisError("This video cannot be decoded by the browser. Try MP4 H.264 or WebM.")} />
                    <canvas ref={overlayCanvasRef} style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }} />
                  </>
                ) : (
                  <>
                    <span className="scan-line" />
                    <span className="target" />
                    <span className="target target-two" />
                  </>
                )}
              </div>
              <div className="camera-footer">
                <span>
                  {analyzing ? "COCO-SSD · RUNNING LIVE INFERENCE" : "COCO-SSD · ANALYSIS READY"}
                  <br />
                  {fileName}
                </span>
                <strong>
                  {detectionCount}
                  <small>ACTIVE NOW ({uniqueObjectCount} TOTAL)</small>
                </strong>
              </div>
            </div>
          ))}
        </div>
      </section>
      <section className="panel alerts-panel">
        <div className="panel-header">
          <div>
            <h2>Alert queue</h2>
            <p>Behavior flags first, then whole-video object summaries</p>
          </div>
        </div>
        {alerts.length === 0 ? (
          <div className="empty-alerts">No objects detected in the current video yet.</div>
        ) : (
          alerts.map((alert) => (
            <div className={`alert-row ${alert.level.toLowerCase()} ${alert.acknowledged ? "acknowledged" : ""}`} key={alert.id}>
              <span className="alert-icon">{alert.level === "CRITICAL" ? "△" : alert.level === "HIGH" ? "‼" : "⌁"}</span>
              <div className="alert-copy">
                <div>
                  <span className="severity">{alert.level}</span>
                  <small>{alert.detail}</small>
                </div>
                <b>{alert.title}</b>
                <small>{alert.camera}</small>
                {!alert.acknowledged && (
                  <button className="ack-button" onClick={() => acknowledge(alert.id)}>
                    ✓ ACKNOWLEDGE
                  </button>
                )}
              </div>
              <time>{alert.time}</time>
            </div>
          ))
        )}
      </section>
    </>
  );
}

function History({ alerts, acknowledge }: { alerts: Alert[]; acknowledge: (id: number) => void }) {
  const [filter, setFilter] = useState("");
  const filtered = alerts.filter((alert) => `${alert.title} ${alert.camera} ${alert.detail}`.toLowerCase().includes(filter.toLowerCase()));
  return (
    <section className="history-view">
      <div className="filter-bar">
        <input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="⌕ Search camera, event message, or type" />
        <small>≡ &nbsp; {filtered.length} records in all view</small>
      </div>
      <div className="history-table">
        <div className="history-head">
          <span>TIME / SEVERITY</span>
          <span>CAMERA</span>
          <span>SIGNAL</span>
          <span>CONFIDENCE</span>
          <span>STATUS</span>
        </div>
        {filtered.map((alert) => (
          <div className="history-row" key={alert.id}>
            <div>
              <span className={`history-symbol ${alert.level.toLowerCase()}`}>△</span>
              <span className="severity">{alert.level}</span>
              <small>{alert.time}</small>
            </div>
            <div>
              <b>{alert.camera}</b>
              <small>source / analysis</small>
            </div>
            <div>
              <b>{alert.title}</b>
              <small>{alert.detail}</small>
            </div>
            <div>
              <b>97%</b>
              <span className="confidence">
                <i />
              </span>
            </div>
            <div>
              {alert.acknowledged ? (
                <small className="closed">◉ ACKNOWLEDGED / closed</small>
              ) : (
                <button className="ack-button" onClick={() => acknowledge(alert.id)}>
                  ✓ ACKNOWLEDGE
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function Fences({ armed, setArmed, alerts, sourceName }: { armed: boolean[]; setArmed: (value: boolean[]) => void; alerts: Alert[]; sourceName: string }) {
  const sourceZones = sourceName ? zoneRules.map((rule, index) => ({ name: `${sourceName} / Zone ${String(index + 1).padStart(2, "0")}`, camera: sourceName, rule })) : [];
  const crossingsToday = alerts.length;
  return (
    <section className="fences-view">
      <div className="stats-grid fence-stats">
        <Stat label="ARMED ZONES" value={String(armed.slice(0, sourceZones.length).filter(Boolean).length)} note="Actively enforcing rules" tone="green" icon="♧" />
        <Stat label="DISARMED ZONES" value={String(sourceZones.length - armed.slice(0, sourceZones.length).filter(Boolean).length)} note="Monitoring paused" tone="amber" icon="♧" />
        <Stat label="CROSSINGS TODAY" value={String(crossingsToday)} note="Generated detection events" tone="blue" icon="♧" />
      </div>
      <div className="zone-grid">
        {sourceZones.length === 0 ? (
          <div className="empty-alerts">Choose a video to configure dynamic perimeter zones.</div>
        ) : (
          sourceZones.map((zone, index) => (
            <div className="zone-card" key={zone.name}>
              <div className="zone-heading">
                <span className={armed[index] ? "zone-icon" : "zone-icon off"}>♧</span>
                <div>
                  <h2>{zone.name}</h2>
                  <small>{zone.camera}</small>
                </div>
                <span className={armed[index] ? "zone-status" : "zone-status off"}>{armed[index] ? "ARMED" : "DISARMED"}</span>
              </div>
              <div className="rule-box">
                <small>RULE</small>
                <b>{zone.rule}</b>
                <strong>{alerts.filter((alert) => alert.camera === zone.camera).length}</strong>
                <small>CROSSINGS</small>
              </div>
              <div className="zone-footer">
                <span>
                  <i className={armed[index] ? "green-dot" : "gray-dot"} /> {armed[index] ? "Detection rule enforcing" : "Rule paused at edge"}
                </span>
                <button className="zone-button" onClick={() => setArmed(armed.map((item, itemIndex) => (itemIndex === index ? !item : item)))}>
                  {armed[index] ? "↻ DISARM ZONE" : "↻ ARM ZONE"}
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

export default App;
