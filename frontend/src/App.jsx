import { useEffect, useMemo, useRef, useState } from "react";

const initialParams = {
  opening_size: 3,
  th_low: 45,
  th_high: 65,
  bg_sampling_rate: 500,
  bg_number_frame: 10,
  cpu_simd: true,
  gpu_diff: true,
  gpu_hysteresis: true,
  gpu_morphology: true,
  gpu_background: true,
  gpu_overlay: true,
  kernel_fusion: true,
};

const backendBaseUrl =
  import.meta.env.BACKEND_URL ?? import.meta.env.VITE_BACKEND_URL ?? "";

const buildBackendUrl = (pathname) =>
  backendBaseUrl ? new URL(pathname, backendBaseUrl).href : pathname;

const LIVE_CAPTURE_INTERVAL_MS = 250;
const LIVE_SKIP_BETWEEN_PAIRS = 1;

const buildBackendWebSocketUrl = (pathname) => {
  const base =
    backendBaseUrl || (typeof window !== "undefined" ? window.location.origin : "http://localhost");
  const url = new URL(pathname, base);
  if (url.protocol === "https:") {
    url.protocol = "wss:";
  } else if (url.protocol === "http:") {
    url.protocol = "ws:";
  }
  return url.href;
};

const dataUrlToBlob = (dataUrl) => {
  const [meta, payload] = dataUrl.split(",");
  const binary = atob(payload);
  const length = binary.length;
  const buffer = new Uint8Array(length);
  for (let i = 0; i < length; i++) {
    buffer[i] = binary.charCodeAt(i);
  }
  const mimeMatch = meta.match(/data:([^;]+);/);
  const mimeType = mimeMatch ? mimeMatch[1] : "image/png";
  return new Blob([buffer], { type: mimeType });
};

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
  const liveVideoRef = useRef(null);
  const [streamActive, setStreamActive] = useState(false);
  const streamRef = useRef(null);
  const liveWsRef = useRef(null);
  const liveCaptureTimerRef = useRef(null);
  const liveSkipCounterRef = useRef(0);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [dropActive, setDropActive] = useState(false);
  const [captureMode, setCaptureMode] = useState("webcam");
  const [videoFile, setVideoFile] = useState(null);
  const [videoResultUrl, setVideoResultUrl] = useState(null);
  const [videoLoading, setVideoLoading] = useState(false);
  const [videoError, setVideoError] = useState("");
  const [videoExecTime, setVideoExecTime] = useState("");
  const [liveStreaming, setLiveStreaming] = useState(false);
  const [liveResultUrl, setLiveResultUrl] = useState(null);
  const [liveError, setLiveError] = useState("");
  const [liveStatus, setLiveStatus] = useState("Idle");
  const modeRef = useRef(mode);
  const paramsRef = useRef(params);

  const canSubmit = useMemo(() => imageA && imageB && !loading, [imageA, imageB, loading]);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setParams((prev) => ({ ...prev, [name]: Number(value) }));
  };

  const handleCheckboxChange = (event) => {
    const { name, checked } = event.target;
    setParams((prev) => ({ ...prev, [name]: checked }));
  };
  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);
  useEffect(() => {
    paramsRef.current = params;
  }, [params]);


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
    setVideoExecTime("");

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

      const execTime = response.headers.get("x-execution-time") ?? "";
      const blob = await response.blob();
      setVideoResultUrl(URL.createObjectURL(blob));
      setVideoExecTime(execTime);
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
      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = stream;
        liveVideoRef.current.play();
      }
    } catch (err) {
      setCameraError("Unable to access the webcam.");
    }
  };

  const stopLiveCaptureLoop = () => {
    if (liveCaptureTimerRef.current !== null) {
      clearInterval(liveCaptureTimerRef.current);
      liveCaptureTimerRef.current = null;
    }
  };

  const captureLiveFrame = () => {
    const video = liveVideoRef.current;
    const ws = liveWsRef.current;
    if (!video || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (liveSkipCounterRef.current > 0) {
      liveSkipCounterRef.current -= 1;
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const context = canvas.getContext("2d");
    context?.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/png");

    liveSkipCounterRef.current = LIVE_SKIP_BETWEEN_PAIRS;
    const payload = {
      type: "frame",
      frame: dataUrl,
      mode: modeRef.current,
      params: { ...paramsRef.current },
    };
    ws.send(JSON.stringify(payload));
    setLiveStatus("Frame sent");
  };

  const startLiveCaptureLoop = () => {
    if (liveCaptureTimerRef.current !== null) return;
    liveCaptureTimerRef.current = setInterval(captureLiveFrame, LIVE_CAPTURE_INTERVAL_MS);
  };

  const stopLiveRecording = () => {
    stopLiveCaptureLoop();
    if (liveWsRef.current) {
      liveWsRef.current.close();
      liveWsRef.current = null;
    }
    liveSkipCounterRef.current = 0;
    setLiveStreaming(false);
    setLiveStatus("Idle");
  };

  const startLiveStream = () => {
    if (liveStreaming) return;
    if (!streamRef.current) {
      setLiveError("Camera must be active to start live streaming.");
      return;
    }
    if (liveWsRef.current) {
      liveWsRef.current.close();
      liveWsRef.current = null;
    }

    setLiveError("");
    setLiveStatus("Connecting...");
    const wsUrl = buildBackendWebSocketUrl("/ws/live");
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      setLiveStatus("Connected");
      setLiveStreaming(true);
      liveSkipCounterRef.current = 0;
      startLiveCaptureLoop();
      setLiveStatus("Recording...");
    };
      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.error) {
            setLiveError(payload.error);
            setLiveStatus("Server error");
            return;
          }
          if (payload.overlay) {
            const blob = dataUrlToBlob(payload.overlay);
            setLiveResultUrl((prev) => {
              if (prev) URL.revokeObjectURL(prev);
              return URL.createObjectURL(blob);
            });
            setLiveStatus("Overlay received");
            return;
          }
          if (payload.status) {
            setLiveStatus(payload.status);
          }
        } catch (err) {
          setLiveError("Invalid overlay data from server.");
        }
      };
    ws.onerror = () => {
      setLiveError("Live websocket error.");
      setLiveStatus("Disconnected");
    };
    ws.onclose = () => {
      stopLiveRecording();
    };
    liveWsRef.current = ws;
  };

  const stopLiveStream = () => {
    stopLiveRecording();
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setStreamActive(false);
    stopLiveRecording();
  };

  const shouldUseCamera =
    (activeTab === "image" && captureMode === "webcam") || activeTab === "livecam";

  useEffect(() => {
    if (shouldUseCamera) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [shouldUseCamera]);

  useEffect(() => {
    if (activeTab !== "livecam") {
      stopLiveRecording();
    }
  }, [activeTab]);
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

  useEffect(() => {
    return () => {
      if (liveResultUrl) {
        URL.revokeObjectURL(liveResultUrl);
      }
    };
  }, [liveResultUrl]);

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

  const gpuOptions = [
    { name: "gpu_diff", label: "Diff" },
    { name: "gpu_hysteresis", label: "Hysteresis" },
    { name: "gpu_morphology", label: "Morphology" },
    { name: "gpu_background", label: "Background update" },
    { name: "gpu_overlay", label: "Overlay" },
    { name: "kernel_fusion", label: "Kernel fusion" },
  ];

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
          <button type="button" className={activeTab === "livecam" ? "active" : ""} onClick={() => setActiveTab("livecam")}>
            Livecam
          </button>
          <button type="button" className={activeTab === "video" ? "active" : ""} onClick={() => setActiveTab("video")}>
            Video
          </button>
          <button type="button" className={activeTab === "settings" ? "active" : ""} onClick={() => setActiveTab("settings")}>
            Settings
          </button>
        </div>
        <div className="nav-actions"></div>
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
                  <div className="stacked-control">
                    <label>
                      Mode
                      <select value={mode} onChange={(event) => setMode(event.target.value)}>
                        <option value="cpu">CPU</option>
                        <option value="gpu">GPU</option>
                      </select>
                    </label>
                    {mode === "cpu" && (
                      <label className="checkbox">
                        <input
                          name="cpu_simd"
                          type="checkbox"
                          checked={params.cpu_simd}
                          onChange={handleCheckboxChange}
                        />
                        SIMD active
                      </label>
                    )}
                    {mode === "gpu" && (
                      <div className="checkbox-grid">
                        {gpuOptions.map(({ name, label }) => (
                          <label key={name} className="checkbox">
                            <input
                              name={name}
                              type="checkbox"
                              checked={params[name]}
                              onChange={handleCheckboxChange}
                            />
                            {label}
                          </label>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="stacked-control">
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

        {activeTab === "livecam" && (
          <section className="card live-card">
            <header className="panel-header">
              <div>
                <h1>Live camera diff</h1>
                <p>Stream the webcam into the backend pipeline and visualize the latest motion overlay.</p>
              </div>
              <label className="stacked-control" style={{ maxWidth: "200px" }}>
                Mode
                <select value={mode} onChange={(event) => setMode(event.target.value)}>
                  <option value="cpu">CPU</option>
                  <option value="gpu">GPU</option>
                </select>
              </label>
            </header>
            <div className="camera-panel">
              <div className="camera-video-wrapper">
                <video ref={liveVideoRef} className="camera-video" autoPlay muted playsInline />
              </div>
              <div className="camera-actions">
                <button type="button" onClick={startLiveStream} disabled={liveStreaming || !streamActive}>
                  {liveStreaming ? "Streaming…" : "Start live capture"}
                </button>
                <button type="button" onClick={stopLiveStream} disabled={!liveStreaming}>
                  Stop
                </button>
              </div>
              <p>Status: {liveStatus}</p>
              {liveError && <p className="error smaller">{liveError}</p>}
            </div>
              <section className="card result-card full">
                <h2>Latest overlay</h2>
                {liveResultUrl ? (
                <img src={liveResultUrl} className="result-image" alt="Live overlay" />
                ) : (
                  <p className="placeholder">Waiting for the next chunk to finish.</p>
                )}
            </section>
          </section>
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
              <div className="row">
                <div className="stacked-control">
                  <label>
                    Mode
                    <select value={mode} onChange={(event) => setMode(event.target.value)}>
                      <option value="cpu">CPU</option>
                      <option value="gpu">GPU</option>
                    </select>
                  </label>
                  {mode === "cpu" && (
                    <label className="checkbox">
                      <input
                        name="cpu_simd"
                        type="checkbox"
                        checked={params.cpu_simd}
                        onChange={handleCheckboxChange}
                      />
                      SIMD active
                    </label>
                  )}
                  {mode === "gpu" && (
                    <div className="checkbox-grid">
                      {gpuOptions.map(({ name, label }) => (
                        <label key={name} className="checkbox">
                          <input
                            name={name}
                            type="checkbox"
                            checked={params[name]}
                            onChange={handleCheckboxChange}
                          />
                          {label}
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <label>
                Video file
                <input type="file" accept="video/*" onChange={(event) => setVideoFile(event.target.files?.[0] ?? null)} />
              </label>
              <button type="submit" disabled={!videoFile || videoLoading}>
                {videoLoading ? "Processing…" : "Upload & Compare"}
              </button>
            </form>
            {videoError && <p className="error smaller">{videoError}</p>}
            {videoExecTime && (
              <p className="smaller">Execution time: {Number(videoExecTime).toFixed(3)}s</p>
            )}
            {videoResultUrl && (
              <video controls muted src={videoResultUrl} className="video-preview"></video>
            )}
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
