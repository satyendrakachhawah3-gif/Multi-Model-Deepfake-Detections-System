(function () {
  const STORAGE_KEY = "deepfake_result";
  const STORAGE_SCHEMA_VERSION = "20260517-3";
  const DEFAULT_API_BASE = "http://127.0.0.1:8000";
  const DEPLOY_CONFIG = window.__DEEPFAKE_CONFIG__ || {};
  const WEIGHTS = { text: 0.3, image: 0.35, video: 0.35 };

  function byId(id) {
    return document.getElementById(id);
  }

  function getConfiguredApiBase() {
    if (DEPLOY_CONFIG && typeof DEPLOY_CONFIG.apiBaseUrl === "string") {
      const apiBaseUrl = DEPLOY_CONFIG.apiBaseUrl.trim();
      if (apiBaseUrl) return apiBaseUrl;
    }
    return "";
  }

  function getDefaultApiBase() {
    return getConfiguredApiBase() || DEFAULT_API_BASE;
  }

  function getSelectedCount(rawResults) {
    return Object.values({
      text: !!(rawResults && rawResults.text),
      image: !!(rawResults && rawResults.image),
      video: !!(rawResults && rawResults.video)
    }).filter(Boolean).length;
  }

  function getPreferredVerdict(result) {
    if (!result) return null;

    const raw = result.raw || {};
    const selectedCount = getSelectedCount(raw);
    if (selectedCount === 1) {
      return (raw.text && raw.text.label) ||
        (raw.image && raw.image.label) ||
        (raw.video && raw.video.label) ||
        result.verdict || null;
    }

    return result.verdict || null;
  }

  function getPreferredConfidence(result) {
    if (!result) return { fake: null, real: null };

    const raw = result.raw || {};
    const selectedCount = getSelectedCount(raw);
    if (selectedCount === 1) {
      const single = raw.text || raw.image || raw.video || null;
      if (single && typeof single.final_score === "number") {
        return {
          fake: toPercent(single.final_score),
          real: 100 - toPercent(single.final_score)
        };
      }
    }

    return {
      fake: result.overallFake == null ? (result.overall == null ? null : result.overall) : result.overallFake,
      real: result.overallReal == null
        ? (result.overallFake == null ? null : 100 - Number(result.overallFake))
        : result.overallReal
    };
  }

  function normalizeStoredResult(result) {
    if (!result || typeof result !== "object") return null;
    const preferredVerdict = getPreferredVerdict(result);
    const preferredConfidence = getPreferredConfidence(result);
    return {
      ...result,
      schemaVersion: result.schemaVersion || STORAGE_SCHEMA_VERSION,
      verdict: preferredVerdict || result.verdict || "N/A",
      overallFake: preferredConfidence.fake,
      overallReal: preferredConfidence.real,
      overall: preferredConfidence.fake
    };
  }

  function toPercent(score) {
    return Math.max(0, Math.min(100, Math.round((Number(score) || 0) * 100)));
  }

  function setStatus(message, type) {
    const statusEl = byId("statusMessage");
    if (!statusEl) return;
    statusEl.textContent = message;
    statusEl.className = "status-banner " + (type || "info");
  }

  function setPreview(fileInput, previewEl, mediaType) {
    if (!fileInput || !previewEl) return;
    fileInput.addEventListener("change", function () {
      const file = fileInput.files && fileInput.files[ 0 ];
      if (!file) {
        previewEl.textContent = mediaType === "image" ? "No image selected." : "No video selected.";
        return;
      }
      const url = URL.createObjectURL(file);
      if (mediaType === "image") {
        previewEl.innerHTML = '<img alt="Selected image preview" />';
        previewEl.querySelector("img").src = url;
      } else {
        previewEl.innerHTML = '<video controls muted></video>';
        previewEl.querySelector("video").src = url;
      }
    });
  }

  async function callAnalyze(apiBase, payload) {
    const res = await fetch(apiBase.replace(/\/$/, "") + "/analyze", {
      method: "POST",
      body: payload
    });

    let data = null;
    try {
      data = await res.json();
    } catch (error) {
      data = null;
    }

    if (!res.ok) {
      const detail = data && (data.detail || data.message) ? String(data.detail || data.message) : "Request failed";
      throw new Error(detail);
    }
    return data;
  }

  async function callConfusionMatrices(apiBase) {
    const res = await fetch(apiBase.replace(/\/$/, "") + "/metrics/confusion-matrices", {
      method: "GET"
    });

    let data = null;
    try {
      data = await res.json();
    } catch (error) {
      data = null;
    }

    if (!res.ok) {
      const detail = data && (data.detail || data.message) ? String(data.detail || data.message) : "Unable to fetch confusion matrices";
      throw new Error(detail);
    }
    return data;
  }

  function toPercentText(value) {
    return (value * 100).toFixed(2) + "%";
  }

  function safeDivide(numerator, denominator) {
    if (!denominator) return 0;
    return numerator / denominator;
  }

  function computeMatrixMetrics(matrix) {
    const tn = Number(matrix.tn_real_as_real || 0);
    const fp = Number(matrix.fp_real_as_fake || 0);
    const fn = Number(matrix.fn_fake_as_real || 0);
    const tp = Number(matrix.tp_fake_as_fake || 0);

    const precisionFake = safeDivide(tp, tp + fp);
    const recallFake = safeDivide(tp, tp + fn);
    const accuracy = safeDivide(tp + tn, tp + tn + fp + fn);

    return {
      precisionFake,
      recallFake,
      accuracy
    };
  }

  function renderSingleMatrix(targetId, payload) {
    const el = byId(targetId);
    if (!el) return;

    if (!payload || !payload.matrix) {
      el.innerHTML = '<p class="cm-empty">No evaluation summary found yet.</p>';
      return;
    }

    const m = payload.matrix;
    const source = payload.summary_file ? String(payload.summary_file) : "N/A";
    const metrics = computeMatrixMetrics(m);

    el.innerHTML =
      '<div class="cm-meta">Source: ' + source + '</div>' +
      '<table class="cm-table" aria-label="Confusion matrix">' +
      '<thead><tr><th></th><th>Pred REAL</th><th>Pred FAKE</th></tr></thead>' +
      '<tbody>' +
      '<tr><th>True REAL</th><td>' + (m.tn_real_as_real ?? 0) + '</td><td>' + (m.fp_real_as_fake ?? 0) + '</td></tr>' +
      '<tr><th>True FAKE</th><td>' + (m.fn_fake_as_real ?? 0) + '</td><td>' + (m.tp_fake_as_fake ?? 0) + '</td></tr>' +
      '</tbody>' +
      '</table>' +
      '<div class="cm-stats">' +
      '<div class="cm-stat"><span>Precision (FAKE)</span><strong>' + toPercentText(metrics.precisionFake) + '</strong></div>' +
      '<div class="cm-stat"><span>Recall (FAKE)</span><strong>' + toPercentText(metrics.recallFake) + '</strong></div>' +
      '<div class="cm-stat"><span>Accuracy</span><strong>' + toPercentText(metrics.accuracy) + '</strong></div>' +
      '</div>';
  }

  async function renderConfusionMatrices(result) {
    const hasCmSection = byId("cmText") || byId("cmImage") || byId("cmVideo");
    if (!hasCmSection) return;

    const apiBase = (result && result.apiBase) ? result.apiBase : getDefaultApiBase();
    try {
      const matrices = await callConfusionMatrices(apiBase);
      renderSingleMatrix("cmText", matrices ? matrices.text : null);
      renderSingleMatrix("cmImage", matrices ? matrices.image : null);
      renderSingleMatrix("cmVideo", matrices ? matrices.video : null);
    } catch (error) {
      renderSingleMatrix("cmText", null);
      renderSingleMatrix("cmImage", null);
      renderSingleMatrix("cmVideo", null);
    }
  }

  function aggregateResults(selected, rawResults) {
    const actualSelected = {
      text: !!(rawResults.text),
      image: !!(rawResults.image),
      video: !!(rawResults.video)
    };
    const selectedCount = Object.values(actualSelected).filter(Boolean).length;
    let weightSum = 0;
    if (selected.text) weightSum += WEIGHTS.text;
    if (selected.image) weightSum += WEIGHTS.image;
    if (selected.video) weightSum += WEIGHTS.video;

    const normalized = {
      text: selected.text ? WEIGHTS.text / weightSum : 0,
      image: selected.image ? WEIGHTS.image / weightSum : 0,
      video: selected.video ? WEIGHTS.video / weightSum : 0
    };

    const score01 = {
      text: selected.text ? Number(rawResults.text.final_score || 0) : null,
      image: selected.image ? Number(rawResults.image.final_score || 0) : null,
      video: selected.video ? Number(rawResults.video.final_score || 0) : null
    };

    const fused01 =
      (score01.text || 0) * normalized.text +
      (score01.image || 0) * normalized.image +
      (score01.video || 0) * normalized.video;

    const percentScoresFake = {
      text: score01.text == null ? null : toPercent(score01.text),
      image: score01.image == null ? null : toPercent(score01.image),
      video: score01.video == null ? null : toPercent(score01.video)
    };

    const percentScoresReal = {
      text: percentScoresFake.text == null ? null : 100 - percentScoresFake.text,
      image: percentScoresFake.image == null ? null : 100 - percentScoresFake.image,
      video: percentScoresFake.video == null ? null : 100 - percentScoresFake.video
    };

    const overallFake = toPercent(fused01);
    const overallReal = 100 - overallFake;

    const apiVerdict = selectedCount === 1
      ? (rawResults.text && rawResults.text.label) ||
      (rawResults.image && rawResults.image.label) ||
      (rawResults.video && rawResults.video.label) ||
      null
      : null;

    return {
      verdict: apiVerdict || (fused01 >= 0.5 ? "FAKE" : "REAL"),
      overall: overallFake,
      overallFake: overallFake,
      overallReal: overallReal,
      scores: percentScoresFake,
      scoresFake: percentScoresFake,
      scoresReal: percentScoresReal,
      score01: score01,
      thresholds: {
        text: selected.text ? rawResults.text.threshold : null,
        image: selected.image ? rawResults.image.threshold : null,
        video: selected.video ? rawResults.video.threshold : null
      }
    };
  }

  async function runAnalysis() {
    const textInput = byId("textInput");
    const imageInput = byId("imageInput");
    const videoInput = byId("videoInput");

    const useText = !!(byId("useTextChoice") && byId("useTextChoice").checked);
    const useImage = !!(byId("useImageChoice") && byId("useImageChoice").checked);
    const useVideo = !!(byId("useVideoChoice") && byId("useVideoChoice").checked);

    if (!useText && !useImage && !useVideo) {
      setStatus("Select at least one modality before running analysis.", "error");
      return;
    }

    const text = textInput ? textInput.value.trim() : "";
    const imageFile = imageInput && imageInput.files ? imageInput.files[ 0 ] : null;
    const videoFile = videoInput && videoInput.files ? videoInput.files[ 0 ] : null;

    if (useText && !text) {
      setStatus("Text modality selected but no text was provided.", "error");
      return;
    }
    if (useImage && !imageFile) {
      setStatus("Image modality selected but no image file was uploaded.", "error");
      return;
    }
    if (useVideo && !videoFile) {
      setStatus("Video modality selected but no video file was uploaded.", "error");
      return;
    }

    const apiBase = getDefaultApiBase();
    const runBtn = byId("runAnalysisBtn");

    if (runBtn) runBtn.disabled = true;
    setStatus("Running model inference. This may take a while for video files.", "info");

    const perModality = {
      text: null,
      image: null,
      video: null
    };
    const errors = {
      text: null,
      image: null,
      video: null
    };

    try {
      if (useText) {
        const fdText = new FormData();
        fdText.append("text", text);
        perModality.text = await callAnalyze(apiBase, fdText);
      }

      if (useImage && imageFile) {
        const fdImage = new FormData();
        fdImage.append("image", imageFile, imageFile.name || "image.jpg");
        perModality.image = await callAnalyze(apiBase, fdImage);
      }

      if (useVideo && videoFile) {
        const fdVideo = new FormData();
        fdVideo.append("video", videoFile, videoFile.name || "video.mp4");
        perModality.video = await callAnalyze(apiBase, fdVideo);
      }
    } catch (error) {
      setStatus("Analysis failed: " + error.message, "error");
      if (runBtn) runBtn.disabled = false;
      return;
    }

    const selected = { text: useText, image: useImage, video: useVideo };
    const fused = aggregateResults(selected, perModality);
    const selectedCount = Object.values(perModality).filter(Boolean).length;

    if (useText && perModality.text && perModality.text.errors) {
      errors.text = perModality.text.errors.text;
    }
    if (useImage && perModality.image && perModality.image.errors) {
      errors.image = perModality.image.errors.image;
    }
    if (useVideo && perModality.video && perModality.video.errors) {
      errors.video = perModality.video.errors.video;
    }

    let result = {
      verdict: selectedCount === 1
        ? ((perModality.text && perModality.text.label) ||
          (perModality.image && perModality.image.label) ||
          (perModality.video && perModality.video.label) ||
          fused.verdict)
        : fused.verdict,
      overall: fused.overall,
      overallFake: fused.overallFake,
      overallReal: fused.overallReal,
      scores: fused.scores,
      scoresFake: fused.scoresFake,
      scoresReal: fused.scoresReal,
      inputs: {
        hasText: useText,
        hasImage: useImage,
        hasVideo: useVideo
      },
      apiBase: apiBase,
      selectedModalities: selected,
      thresholds: fused.thresholds,
      apiStates: {
        text: perModality.text ? perModality.text.states.text : "Not selected",
        image: perModality.image ? perModality.image.states.image : "Not selected",
        video: perModality.video ? perModality.video.states.video : "Not selected"
      },
      errors: errors,
      raw: {
        text: perModality.text,
        image: perModality.image,
        video: perModality.video
      },
      createdAt: new Date().toLocaleString()
    };

    // Normalize result so single-modality API labels take precedence
    result = normalizeStoredResult(result) || result;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(result));

    const quickResult = byId("analysisQuickResult");
    if (quickResult) {
      quickResult.hidden = false;
      quickResult.innerHTML =
        "<strong>Result:</strong> " + result.verdict +
        " <span class=\"result-pill\">Fake " + result.overallFake + "% | Real " + result.overallReal + "%</span>";
    }

    setStatus("Analysis completed successfully. Open dashboard for details.", "success");
    if (runBtn) runBtn.disabled = false;
  }

  function clearAnalysisForm() {
    const textInput = byId("textInput");
    const imageInput = byId("imageInput");
    const videoInput = byId("videoInput");
    const imagePreview = byId("imagePreview");
    const videoPreview = byId("videoPreview");
    const quickResult = byId("analysisQuickResult");

    localStorage.removeItem(STORAGE_KEY);

    if (textInput) textInput.value = "";
    if (imageInput) imageInput.value = "";
    if (videoInput) videoInput.value = "";
    if (imagePreview) imagePreview.textContent = "No image selected.";
    if (videoPreview) videoPreview.textContent = "No video selected.";
    if (quickResult) {
      quickResult.hidden = true;
      quickResult.innerHTML = "";
    }
    setStatus("Inputs were cleared.", "info");
  }

  async function renderDashboard() {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      await renderConfusionMatrices({ apiBase: getDefaultApiBase() });
      return;
    }

    let result;
    try {
      result = JSON.parse(raw);
      // Normalize legacy or partial stored objects so dashboard shows authoritative verdicts
      result = normalizeStoredResult(result) || result;
    } catch (error) {
      return;
    }

    const verdictEl = byId("overallVerdict");
    const confidenceEl = byId("overallConfidence");

    const displayVerdict = getPreferredVerdict(result) || result.verdict || "N/A";
    const displayConfidence = getPreferredConfidence(result);

    if (verdictEl) {
      verdictEl.textContent = displayVerdict;
      verdictEl.classList.remove("fake", "real");
      if (displayVerdict === "FAKE") verdictEl.classList.add("fake");
      if (displayVerdict === "REAL") verdictEl.classList.add("real");
    }
    if (confidenceEl) {
      const overallFake = displayConfidence.fake;
      const overallReal = displayConfidence.real;
      confidenceEl.textContent = "Confidence (Fake/Real): " +
        String(overallFake == null ? "--" : overallFake) + "% / " +
        String(overallReal == null ? "--" : overallReal) + "%";
    }

    const mappings = [
      { key: "text", metric: "metricText", label: "textScoreLabel", bar: "textBar" },
      { key: "image", metric: "metricImage", label: "imageScoreLabel", bar: "imageBar" },
      { key: "video", metric: "metricVideo", label: "videoScoreLabel", bar: "videoBar" }
    ];

    mappings.forEach(function (m) {
      const enabled = result.inputs && result.inputs[ "has" + m.key.charAt(0).toUpperCase() + m.key.slice(1) ];
      const metricEl = byId(m.metric);
      const labelEl = byId(m.label);
      const barEl = byId(m.bar);
      const scoreFake = result.scoresFake
        ? result.scoresFake[ m.key ]
        : (result.scores ? result.scores[ m.key ] : null);
      const scoreReal = result.scoresReal
        ? result.scoresReal[ m.key ]
        : (scoreFake == null ? null : 100 - Number(scoreFake));

      if (metricEl) metricEl.style.display = enabled ? "" : "none";
      if (enabled && labelEl) {
        labelEl.textContent = (scoreFake == null || scoreReal == null)
          ? "--"
          : ("Fake: " + scoreFake + "% | Real: " + scoreReal + "%");
      }
      if (enabled && barEl) {
        barEl.style.width = (scoreFake == null ? 0 : scoreFake) + "%";
        barEl.title = (scoreFake == null || scoreReal == null)
          ? ""
          : ("Fake: " + scoreFake + "% | Real: " + scoreReal + "%");
      }
    });

    const insightsList = byId("insightsList");
    if (insightsList) {
      const lines = [];

      if (result.inputs && result.inputs.hasText && result.raw && result.raw.text) {
        lines.push("Text model state: " + (result.apiStates.text || "unknown") + ".");
      }
      if (result.inputs && result.inputs.hasImage && result.raw && result.raw.image) {
        lines.push("Image model state: " + (result.apiStates.image || "unknown") + ".");
      }
      if (result.inputs && result.inputs.hasVideo && result.raw && result.raw.video) {
        lines.push("Video model state: " + (result.apiStates.video || "unknown") + ".");
      }

      if (result.errors) {
        if (result.errors.text) lines.push("Text warning: " + result.errors.text + ".");
        if (result.errors.image) lines.push("Image warning: " + result.errors.image + ".");
        if (result.errors.video) lines.push("Video warning: " + result.errors.video + ".");
      }

      lines.push("Analysis timestamp: " + result.createdAt + ".");
      insightsList.innerHTML = lines.map(function (line) {
        return "<li>" + line + "</li>";
      }).join("");
    }

    const requestMeta = byId("requestMeta");
    if (requestMeta) {
      requestMeta.innerHTML =
        "<strong>Request source:</strong> " + (result.apiBase || DEFAULT_API_BASE) +
        "<br /><strong>Fusion:</strong> Weighted multimodal blend";
    }

    await renderConfusionMatrices(result);
  }

  function downloadReport() {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      alert("No analysis result found. Run analysis first.");
      return;
    }

    let result;
    try {
      result = JSON.parse(raw);
      result = normalizeStoredResult(result) || result;
    } catch (error) {
      alert("Stored result is corrupted. Please re-run analysis.");
      return;
    }

    const lines = [
      "DeepFake Detection - Multimodal Deepfake Report",
      "=========================================",
      "Verdict: " + (result.verdict || "N/A"),
      "Overall Fake%: " + (result.overallFake == null ? (result.overall == null ? "N/A" : result.overall + "%") : result.overallFake + "%"),
      "Overall Real%: " + (result.overallReal == null
        ? ((result.overallFake == null && result.overall == null) ? "N/A" : (100 - Number(result.overallFake == null ? result.overall : result.overallFake)) + "%")
        : result.overallReal + "%"),
      "Text Fake%: " + (result.inputs && result.inputs.hasText
        ? ((result.scoresFake ? result.scoresFake.text : (result.scores ? result.scores.text : "N/A")) + "%")
        : "N/A"),
      "Image Fake%: " + (result.inputs && result.inputs.hasImage
        ? ((result.scoresFake ? result.scoresFake.image : (result.scores ? result.scores.image : "N/A")) + "%")
        : "N/A"),
      "Video Fake%: " + (result.inputs && result.inputs.hasVideo
        ? ((result.scoresFake ? result.scoresFake.video : (result.scores ? result.scores.video : "N/A")) + "%")
        : "N/A"),
      "API Base: " + (result.apiBase || DEFAULT_API_BASE),
      "Generated At: " + (result.createdAt || "N/A")
    ];

    const blob = new Blob([ lines.join("\n") ], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "deepshield-analysis-report.txt";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function applyChoiceState() {
    const useText = !!(byId("useTextChoice") && byId("useTextChoice").checked);
    const useImage = !!(byId("useImageChoice") && byId("useImageChoice").checked);
    const useVideo = !!(byId("useVideoChoice") && byId("useVideoChoice").checked);

    const textInput = byId("textInput");
    const imageInput = byId("imageInput");
    const videoInput = byId("videoInput");
    const imagePreview = byId("imagePreview");
    const videoPreview = byId("videoPreview");

    if (textInput) textInput.disabled = !useText;
    if (imageInput) imageInput.disabled = !useImage;
    if (videoInput) videoInput.disabled = !useVideo;

    if (!useImage && imagePreview) imagePreview.textContent = "Image modality disabled.";
    if (!useVideo && videoPreview) videoPreview.textContent = "Video modality disabled.";
  }

  function setupRevealAnimation() {
    const revealEls = document.querySelectorAll(".reveal");
    revealEls.forEach(function (el, index) {
      el.style.animationDelay = index * 0.08 + "s";
      el.classList.add("reveal-active");
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    setupRevealAnimation();

    setPreview(byId("imageInput"), byId("imagePreview"), "image");
    setPreview(byId("videoInput"), byId("videoPreview"), "video");

    const runBtn = byId("runAnalysisBtn");
    const clearBtn = byId("clearBtn");
    const downloadBtn = byId("downloadBtn");
    const useTextChoice = byId("useTextChoice");
    const useImageChoice = byId("useImageChoice");
    const useVideoChoice = byId("useVideoChoice");

    if (runBtn) runBtn.addEventListener("click", runAnalysis);
    if (clearBtn) clearBtn.addEventListener("click", clearAnalysisForm);
    if (downloadBtn) downloadBtn.addEventListener("click", downloadReport);
    if (useTextChoice) useTextChoice.addEventListener("change", applyChoiceState);
    if (useImageChoice) useImageChoice.addEventListener("change", applyChoiceState);
    if (useVideoChoice) useVideoChoice.addEventListener("change", applyChoiceState);

    applyChoiceState();
    renderDashboard();
  });
})();
