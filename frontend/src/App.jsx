import { useEffect, useMemo, useRef, useState } from "react";

const initialParams = {
  opening_size: 3,
  th_low: 3,
  th_high: 30,
  bg_sampling_rate: 500,
  bg_number_frame: 10,
};

const backendBaseUrl =
  import.meta.env.BACKEND_URL ?? import.meta.env.VITE_BACKEND_URL ?? "";

const buildBackendUrl = (pathname) =>
  backendBaseUrl ? new URL(pathname, backendBaseUrl).href : pathname;

function App() {
  const [imageA, setImageA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [mode, setMode] = useState("cpu");
  const [params, setParams] = useState(initialParams);
  const [resultUrl, setResultUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [cameraError, setCameraError] = useState("");
  const videoRef = useRef(null);
  const [streamActive, setStreamActive] = useState(false);
  const streamRef = useRef(null);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [dropActive, setDropActive] = useState(false);
  const [captureMode, setCaptureMode] = useState("webcam");
  const [videoFile, setVideoFile] = useState(null);
  const [videoResultUrl, setVideoResultUrl] = useState(null);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoError, setVideoError] = useState("");

  const canSubmit = useMemo(() => imageA && imageB && !loading, [imageA, imageB, loading]);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setParams((prev) => ({ ...prev, [name]: Number(value) }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }

    setLoading(true);
    setError("");
    setResultUrl(null);

    const formData = new FormData();
    formData.append("image_a", imageA);
    formData.append("image_b", imageB);
    formData.append("mode", mode);
    Object.entries(params).forEach(([key, value]) => {
      formData.append(key, value.toString());
    });

    try {
      const response = await fetch(buildBackendUrl("/compare"), {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }

      const blob = await response.blob();
      setResultUrl(URL.createObjectURL(blob));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const handleVideoSubmit = async (event) => {
    event.preventDefault();
    if (!videoFile) return;

    setVideoLoading(true);
    setVideoError("");
    setVideoResultUrl(null);

    const formData = new FormData();
    formData.append("video", videoFile);
    formData.append("mode", mode);
    Object.entries(params).forEach(([key, value]) => {
      formData.append(key, value.toString());
    });

    try {
      const response = await fetch(buildBackendUrl("/compare/video"), {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }

      const blob = await response.blob();
      setVideoResultUrl(URL.createObjectURL(blob));
    } catch (err) {
      setVideoError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setVideoLoading(false);
    }
  };

  const startCamera = async () => {
    if (streamRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      setCameraError("");
      setStreamActive(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      setCameraError("Unable to access the webcam.");
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStreamActive(false);
  };

  useEffect(() => {
    if (activeTab === "image" && captureMode === "webcam") {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [activeTab, captureMode]);
  const captureFrame = async (setter, label) => {
    if (!streamRef.current || !videoRef.current) {
      setCameraError("Camera is not ready.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const context = canvas.getContext("2d");
    context?.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) =>
      canvas.toBlob((b) => resolve(b), "image/png")
    );
    if (!blob) {
      setCameraError("Failed to capture frame.");
      return;
    }
    const file = new File([blob], `${label}.png`, { type: "image/png" });
    setter(file);
  };

  const [previewA, setPreviewA] = useState(null);
  const [previewB, setPreviewB] = useState(null);

  useEffect(() => {
    if (!imageA) {
      setPreviewA(null);
      return;
    }
    const url = URL.createObjectURL(imageA);
    setPreviewA(url);
    return () => URL.revokeObjectURL(url);
  }, [imageA]);

  useEffect(() => {
    if (!imageB) {
      setPreviewB(null);
      return;
    }
    const url = URL.createObjectURL(imageB);
    setPreviewB(url);
    return () => URL.revokeObjectURL(url);
  }, [imageB]);

  useEffect(() => {
    if (!videoResultUrl) {
      return;
    }
    return () => URL.revokeObjectURL(videoResultUrl);
  }, [videoResultUrl]);

  const handleDrop = (event) => {
    event.preventDefault();
    const [first, second] = Array.from(event.dataTransfer.files);
    if (first) {
      setImageA(first);
    }
    if (second) {
      setImageB(second);
    }
    setDropActive(false);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDropActive(true);
  };

  const handleDragLeave = () => {
    setDropActive(false);
  };

  return (
    <div className="app-shell">
      <nav className="top-nav">
        <div className="logo">
          <span className="orb"></span>
          Stream Lab
        </div>
        <div className="nav-links">
          <button type="button" className={activeTab === "dashboard" ? "active" : ""} onClick={() => setActiveTab("dashboard")}>
            Dashboard
          </button>
          <button
            type="button"
            className={activeTab === "image" ? "active" : ""}
            onClick={() => {
              setActiveTab("image");
              setCaptureMode("webcam");
            }}
          >
            Image
          </button>
          <button type="button" className={activeTab === "video" ? "active" : ""} onClick={() => setActiveTab("video")}>
            Video
          </button>
          <button type="button" className={activeTab === "benchmarks" ? "active" : ""} onClick={() => setActiveTab("benchmarks")}>
            Benchmarks
          </button>
          <button type="button" className={activeTab === "settings" ? "active" : ""} onClick={() => setActiveTab("settings")}>
            Settings
          </button>
        </div>
        <div className="nav-actions">
          <button type="button">Run Demo</button>
          <button type="button" className="ghost">
            Export
          </button>
        </div>
      </nav>

      <div className="panel wide">
        {activeTab === "dashboard" && (
          <>
            <section className="hero-card" id="home">
              <div>
            <p className="eyebrow">Deterministic Motion Detection</p>
            <h1>Fast Stream Analysis</h1>
            <p className="lead">
              Compare two frames in seconds with a deterministic pixel-diff pipeline, surface the moving pixels, and monitor GPU vs CPU thresholds with a single click.
            </p>
                <div className="hero-actions">
                  <a
                    className="cta primary"
                    href="#image"
                    onClick={() => {
                      setActiveTab("image");
                      setCaptureMode("webcam");
                    }}
                  >
                    Try image capture
                  </a>
                  <a className="cta ghost" href="#video" onClick={() => setActiveTab("video")}>
                    Go to video tools
                  </a>
                </div>
              </div>
              <ul className="hero-list">
            <li>Deterministic background blending keeps your diff focused on motion.</li>
            <li>Web-based controls with webcam capture and drag/drop uploads.</li>
            <li>Switch between CPU & GPU modes to compare latency and quality.</li>
              </ul>
            </section>
            <section className="insights" id="overview">
              <article>
                <h3>Pipeline status</h3>
                <p>Ready for live capture. Switch to the pipeline tab to snap two frames.</p>
              </article>
              <article>
                <h3>Background blending</h3>
                <p>Background update happens only on motion-free regions to minimize ghosting.</p>
              </article>
            </section>
          </>
        )}

        {activeTab === "image" && (
          <>
            <header className="panel-header">
              <div>
                <h1>Stream Diff</h1>
                <p>Highlight motion between two captures, either from the webcam or via drag-and-drop.</p>
              </div>
              <div className="tabs">
                <button type="button" className={captureMode === "webcam" ? "active" : ""} onClick={() => setCaptureMode("webcam")}>
                  Webcam
                </button>
                <button type="button" className={captureMode === "upload" ? "active" : ""} onClick={() => setCaptureMode("upload")}>
                  Upload
                </button>
              </div>
            </header>

            <form className="grid" onSubmit={handleSubmit}>
              <section className="card">
                <h2>Capture area</h2>
                {captureMode === "webcam" ? (
                  <div className="camera-panel">
                    <div className="camera-video-wrapper">
                      <video ref={videoRef} className="camera-video" autoPlay muted playsInline />
                    </div>
                    <div className="camera-actions">
                      <button type="button" disabled={!streamActive} onClick={() => captureFrame(setImageA, "baseline")}>
                        Capture baseline
                      </button>
                      <button type="button" disabled={!streamActive} onClick={() => captureFrame(setImageB, "current")}>
                        Capture current
                      </button>
                    </div>
                    {cameraError && <p className="error smaller">{cameraError}</p>}
                  </div>
                ) : (
                  <div
                    className={`drop-zone${dropActive ? " active" : ""}`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                  >
                    <p>Drop two images here (first is baseline, second is current)</p>
                  </div>
                )}
              </section>

              {captureMode === "upload" && (
                <section className="card">
                  <h2>Manual upload</h2>
                  <div className="file-group">
                    <label>
                      Baseline image
                      <input type="file" accept="image/*" onChange={(event) => setImageA(event.target.files?.[0] ?? null)} />
                    </label>
                    <label>
                      Current image
                      <input type="file" accept="image/*" onChange={(event) => setImageB(event.target.files?.[0] ?? null)} />
                    </label>
                  </div>
                </section>
              )}

            <section className="card full" id="image">
                <h2>Parameters</h2>
                <div className="row">
                  <label>
                    Mode
                    <select value={mode} onChange={(event) => setMode(event.target.value)}>
                      <option value="cpu">CPU</option>
                      <option value="gpu">GPU</option>
                    </select>
                  </label>
                  <label>
                    Opening size
                    <input
                      name="opening_size"
                      type="number"
                      min="1"
                      step="2"
                      value={params.opening_size}
                      onChange={handleChange}
                    />
                  </label>
                </div>
                <div className="row">
                  <label>
                    Threshold low
                    <input name="th_low" type="number" value={params.th_low} onChange={handleChange} />
                  </label>
                  <label>
                    Threshold high
                    <input name="th_high" type="number" value={params.th_high} onChange={handleChange} />
                  </label>
                </div>
                <div className="row">
                  <label>
                    BG sampling rate
                    <input name="bg_sampling_rate" type="number" value={params.bg_sampling_rate} onChange={handleChange} />
                  </label>
                  <label>
                    BG number frame
                    <input name="bg_number_frame" type="number" value={params.bg_number_frame} onChange={handleChange} />
                  </label>
                </div>
              </section>

              <section className="card full">
                <button type="submit" disabled={!canSubmit}>
                  {loading ? "Processing…" : "Compare"}
                </button>
                {error && <p className="error">Error: {error}</p>}
              </section>
            </form>

            <section className="grid result-grid" id="pipeline-results">
              <article className="card preview-card">
                <h3>Baseline preview</h3>
                {previewA ? <img src={previewA} alt="Baseline preview" /> : <p className="placeholder">Waiting for capture…</p>}
              </article>
              <article className="card preview-card">
                <h3>Current preview</h3>
                {previewB ? <img src={previewB} alt="Current preview" /> : <p className="placeholder">Waiting for capture…</p>}
              </article>
              {resultUrl && (
                <article className="card result-card full">
                  <h2>Result</h2>
                  <img src={resultUrl} alt="Motion overlay" className="result-image" />
                </article>
              )}
            </section>
          </>
        )}

        {activeTab === "video" && (
          <section className="insights" id="video">
            <article>
              <h3>Upload video</h3>
              <p>Drop a short clip to process multiple frames at once, or stream a camera feed live.</p>
            </article>
            <article>
              <h3>Live camera</h3>
              <p>This tab will eventually host a live preview and recording controls for benchmarking video streams.</p>
            </article>
          </section>
        )}

        {activeTab === "video" && (
          <section className="video-card" id="video">
            <h2>Video upload</h2>
            <p>Drop a clip or select a file to apply the motion-overlay pipeline on multiple frames.</p>
            <form className="video-form" onSubmit={handleVideoSubmit}>
              <label>
                Video file
                <input type="file" accept="video/*" onChange={(event) => setVideoFile(event.target.files?.[0] ?? null)} />
              </label>
              <button type="submit" disabled={!videoFile || videoLoading}>
                {videoLoading ? "Processing…" : "Upload & Compare"}
              </button>
            </form>
            {videoError && <p className="error smaller">{videoError}</p>}
            {videoResultUrl && (
              <video controls muted src={videoResultUrl} className="video-preview"></video>
            )}
          </section>
        )}

        {activeTab === "benchmarks" && (
          <section className="insights" id="benchmarks">
            <article>
              <h3>Performance</h3>
              <p>CPU and GPU thresholds will be shown here once benchmarking runs are available.</p>
            </article>
            <article>
              <h3>Next steps</h3>
              <p>Queue captures, tweak morphology, and compare results side-by-side with future tabs.</p>
            </article>
          </section>
        )}

        {activeTab === "settings" && (
          <section className="insights" id="settings">
            <article>
              <h3>Configuration</h3>
              <p>Configure your environment variables, optimization flags, and output paths here.</p>
            </article>
            <article>
              <h3>Integrations</h3>
              <p>Hook into your favorite dashboard or monitoring tool via the native API.</p>
            </article>
          </section>
        )}
      </div>
    </div>
  );
}

export default App;
