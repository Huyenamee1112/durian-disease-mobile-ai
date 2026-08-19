from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from image_quality import inspect_image
from storage import StorageError, save_submission


DISEASES = {
    "Không rõ / cần chuyên gia xác nhận": "Unknown",
    "Lá khỏe mạnh": "Leaf_Healthy",
    "Bệnh đốm rong": "Leaf_Algal",
    "Bệnh cháy lá": "Leaf_Blight",
    "Bệnh thán thư": "Leaf_Colletotrichum",
    "Bệnh Phomopsis": "Leaf_Phomopsis",
    "Bệnh Rhizoctonia": "Leaf_Rhizoctonia",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


PAGE = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thu thập ảnh bệnh lá sầu riêng thực tế</title>
  <style>
    :root {
      --leaf: #176b3a;
      --leaf-dark: #0d4d29;
      --mint: #eaf7ef;
      --line: #cfe7d8;
      --ink: #173c27;
      --muted: #607568;
      --danger: #b42318;
      --success: #16703d;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 95% 0%, #d8f3e2 0, transparent 30%),
        linear-gradient(180deg, #f7fcf8 0%, #edf8f1 100%);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    main {
      width: min(680px, 100%);
      margin: 0 auto;
      padding: 2rem 1rem 3rem;
    }

    .hero {
      padding: 1.6rem;
      margin-bottom: 1.25rem;
      color: white;
      border-radius: 24px;
      background: linear-gradient(135deg, #0f5b31 0%, #269457 100%);
      box-shadow: 0 16px 38px rgba(20, 105, 57, 0.20);
    }

    .hero-badge {
      display: inline-block;
      padding: .35rem .7rem;
      margin-bottom: .7rem;
      border: 1px solid rgba(255,255,255,.35);
      border-radius: 999px;
      background: rgba(255,255,255,.13);
      font-size: .82rem;
      font-weight: 650;
    }

    h1 {
      margin: 0 0 .45rem;
      color: white;
      font-size: clamp(1.65rem, 6vw, 2.25rem);
      line-height: 1.15;
    }

    .hero p {
      margin: 0;
      color: rgba(255,255,255,.9);
      line-height: 1.55;
    }

    form {
      padding: 1.3rem;
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255,255,255,.94);
      box-shadow: 0 10px 30px rgba(35, 91, 56, .09);
    }

    .step {
      display: flex;
      align-items: center;
      gap: .7rem;
      margin: .15rem 0 .7rem;
    }

    .step-number {
      display: grid;
      width: 34px;
      height: 34px;
      flex: 0 0 34px;
      place-items: center;
      border-radius: 50%;
      color: white;
      background: var(--leaf);
      font-weight: 750;
    }

    .step-title {
      color: var(--ink);
      font-size: 1.08rem;
      font-weight: 750;
    }

    .camera-box {
      position: relative;
      overflow: hidden;
      padding: .65rem;
      border-radius: 16px;
      background: var(--mint);
    }

    video, canvas {
      display: block;
      width: 100%;
      min-height: 260px;
      border-radius: 13px;
      background: #102418;
      object-fit: cover;
    }

    video { display: none; }
    canvas { display: none; }
    input[type="file"] { display: none; }

    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: .7rem;
      margin-top: .7rem;
    }

    label {
      display: block;
      margin: 1.1rem 0 .45rem;
      font-weight: 650;
    }

    select,
    input[type="text"],
    textarea {
      width: 100%;
      min-height: 3rem;
      padding: .75rem .85rem;
      border: 1px solid var(--line);
      border-radius: 13px;
      color: var(--ink);
      background: #fbfefc;
      font: inherit;
    }

    textarea {
      min-height: 6rem;
      resize: vertical;
      line-height: 1.45;
    }

    button {
      min-height: 3.15rem;
      border: 0;
      border-radius: 14px;
      color: white;
      background: linear-gradient(135deg, var(--leaf-dark), #23894f);
      font-size: 1rem;
      font-weight: 750;
      box-shadow: 0 8px 18px rgba(23, 107, 58, .20);
      cursor: pointer;
    }

    button.secondary {
      color: var(--leaf-dark);
      border: 1px solid var(--line);
      background: white;
      box-shadow: none;
    }

    .camera-capture {
      display: none;
      position: absolute;
      left: 50%;
      bottom: 1rem;
      z-index: 2;
      width: auto;
      min-width: 9.5rem;
      padding: 0 1.3rem;
      transform: translateX(-50%);
      border: 2px solid rgba(255,255,255,.92);
      border-radius: 999px;
      box-shadow: 0 10px 26px rgba(0,0,0,.25);
    }

    .camera-box.is-live .camera-capture {
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    .camera-hint {
      display: none;
      position: absolute;
      left: 1rem;
      right: 1rem;
      top: 1rem;
      z-index: 2;
      padding: .55rem .75rem;
      border-radius: 999px;
      color: white;
      background: rgba(12, 48, 29, .72);
      font-size: .82rem;
      text-align: center;
      backdrop-filter: blur(6px);
    }

    .camera-box.is-live .camera-hint {
      display: block;
    }

    .zoom-control {
      display: none;
      margin-top: .7rem;
      padding: .75rem .85rem;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #f7fcf8;
    }

    .zoom-control.is-visible {
      display: block;
    }

    .zoom-control label {
      display: flex;
      justify-content: space-between;
      gap: .75rem;
      margin: 0 0 .55rem;
      font-size: .92rem;
    }

    .zoom-control input {
      width: 100%;
      accent-color: var(--leaf);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: .62;
    }

    .caption {
      margin: .55rem 0 0;
      color: var(--muted);
      font-size: .92rem;
      line-height: 1.45;
    }

    .location-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: .75rem;
      margin-top: 1rem;
      padding: .8rem;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #f7fcf8;
    }

    .location-row p {
      margin: 0;
      color: var(--muted);
      font-size: .9rem;
      line-height: 1.35;
    }

    .location-row button {
      min-height: 2.65rem;
      padding: 0 .9rem;
      flex: 0 0 auto;
      font-size: .92rem;
    }

    .message {
      display: none;
      margin-top: 1rem;
      padding: .9rem 1rem;
      border-radius: 14px;
      line-height: 1.45;
    }

    .message.error {
      display: block;
      color: var(--danger);
      background: #fff1f0;
      border: 1px solid #ffd1cc;
    }

    .message.success {
      display: block;
      color: var(--success);
      background: #ecfdf3;
      border: 1px solid #b8e7ca;
    }

    .privacy-note {
      margin-top: .9rem;
      text-align: center;
      color: var(--muted);
      font-size: .84rem;
    }

    @media (max-width: 640px) {
      main { padding: 1rem .8rem 2rem; }
      .hero { padding: 1.35rem; border-radius: 20px; }
      form { padding: 1rem; border-radius: 18px; }
      .actions { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-badge">DỮ LIỆU CỘNG ĐỒNG</div>
      <h1>Thu thập ảnh bệnh lá sầu riêng thực tế</h1>
      <p>Chụp ảnh lá ngoài vườn, nhập nhãn nghi ngờ và thông tin bối cảnh để tạo dữ liệu phục vụ train model.</p>
    </section>

    <form id="submission-form">
      <div class="step">
        <span class="step-number">1</span>
        <span class="step-title">Chụp ảnh lá</span>
      </div>

      <div class="camera-box">
        <video id="camera" autoplay playsinline muted></video>
        <canvas id="snapshot"></canvas>
        <input id="fallback-image" type="file" accept="image/*" capture="environment">
        <div class="camera-hint" id="camera-hint">Chạm vào lá để lấy nét, chỉnh zoom rồi bấm Chụp ảnh</div>
        <button type="button" class="camera-capture" id="capture-overlay">Chụp ảnh</button>
      </div>
      <div class="zoom-control" id="zoom-control">
        <label for="zoom-range">
          <span>Phóng to / thu nhỏ</span>
          <span id="zoom-value">1x</span>
        </label>
        <input id="zoom-range" type="range" min="1" max="1" step="0.1" value="1">
      </div>
      <p class="caption">Mẹo: chạm vào lá để lấy nét, chụp cận một lá sầu riêng, để lá chiếm phần lớn khung hình và tránh rung tay.</p>

      <div class="actions">
        <button type="button" class="secondary" id="start-camera">Mở camera</button>
        <button type="button" class="secondary" id="capture">Chụp ảnh</button>
      </div>
      <div id="message" class="message" role="status"></div>

      <label for="disease">Tên bệnh trên lá</label>
      <select id="disease" name="disease">
        {% for label in diseases %}
          <option value="{{ label }}">{{ label }}</option>
        {% endfor %}
      </select>
      <p class="caption">Nếu chưa chắc bệnh, hãy chọn Chưa xác định để chuyên gia kiểm tra lại sau.</p>

      <label for="tree-stage">Tuổi cây hoặc giai đoạn sinh trưởng</label>
      <input id="tree-stage" name="tree_stage" type="text" maxlength="120" placeholder="Ví dụ: cây 3 năm, ra đọt non, sau thu hoạch">

      <label for="notes">Ghi chú thêm</label>
      <textarea id="notes" name="notes" maxlength="600" placeholder="Ví dụ: lá bị đốm nhiều ở mép, sau mưa, vườn vừa phun thuốc"></textarea>
      <p class="caption">Ghi chú giúp dữ liệu thực tế có bối cảnh tốt hơn khi chuyên gia xác nhận nhãn.</p>

      <input id="latitude" name="latitude" type="hidden">
      <input id="longitude" name="longitude" type="hidden">
      <input id="location-accuracy" name="location_accuracy" type="hidden">
      <input id="location-name" name="location_name" type="hidden">
      <input id="captured-at" name="captured_at" type="hidden">

      <div class="location-row">
        <p id="location-status">Vị trí: chưa lấy. App chỉ lưu vị trí nếu bạn đồng ý cấp quyền.</p>
        <button type="button" class="secondary" id="get-location">Lấy vị trí</button>
      </div>

      <button type="submit" id="submit">Gửi dữ liệu</button>
    </form>

    <p class="privacy-note">Ảnh chỉ được dùng để xây dựng bộ dữ liệu nghiên cứu.</p>
  </main>

  <script>
    const video = document.getElementById("camera");
    const canvas = document.getElementById("snapshot");
    const cameraBox = document.querySelector(".camera-box");
    const fallbackInput = document.getElementById("fallback-image");
    const startButton = document.getElementById("start-camera");
    const captureButton = document.getElementById("capture");
    const captureOverlayButton = document.getElementById("capture-overlay");
    const zoomControl = document.getElementById("zoom-control");
    const zoomRange = document.getElementById("zoom-range");
    const zoomValue = document.getElementById("zoom-value");
    const locationButton = document.getElementById("get-location");
    const locationStatus = document.getElementById("location-status");
    const submitButton = document.getElementById("submit");
    const form = document.getElementById("submission-form");
    const message = document.getElementById("message");
    let stream = null;
    let hasSnapshot = false;
    let snapshotAccepted = false;
    let cameraReady = false;
    let videoTrack = null;

    function showMessage(text, type) {
      message.textContent = text;
      message.className = `message ${type}`;
    }

    function isSecureCameraContext() {
      return window.isSecureContext || ["localhost", "127.0.0.1"].includes(window.location.hostname);
    }

    function isIOSDevice() {
      return /iPad|iPhone|iPod/.test(navigator.userAgent)
        || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    }

    function setLivePreview(isLive) {
      cameraBox.classList.toggle("is-live", isLive);
    }

    function resetZoomControl() {
      zoomControl.classList.remove("is-visible");
      zoomRange.min = "1";
      zoomRange.max = "1";
      zoomRange.step = "0.1";
      zoomRange.value = "1";
      zoomValue.textContent = "1x";
    }

    function setupZoomControl(track) {
      const capabilities = track.getCapabilities ? track.getCapabilities() : {};
      if (!capabilities.zoom) {
        resetZoomControl();
        return;
      }

      const min = capabilities.zoom.min || 1;
      const max = capabilities.zoom.max || min;
      const step = capabilities.zoom.step || 0.1;
      if (max <= min) {
        resetZoomControl();
        return;
      }

      zoomRange.min = String(min);
      zoomRange.max = String(max);
      zoomRange.step = String(step);
      zoomRange.value = String(min);
      zoomValue.textContent = `${Number(min).toFixed(1)}x`;
      zoomControl.classList.add("is-visible");
    }

    async function applyCameraZoom(value) {
      if (!videoTrack) {
        return;
      }
      const zoom = Number(value);
      zoomValue.textContent = `${zoom.toFixed(1)}x`;
      try {
        await videoTrack.applyConstraints({ advanced: [{ zoom }] });
      } catch (error) {
        showMessage("Trình duyệt không hỗ trợ chỉnh zoom cho camera này.", "error");
      }
    }

    async function requestFocus() {
      if (!videoTrack || !videoTrack.applyConstraints) {
        return;
      }
      const capabilities = videoTrack.getCapabilities ? videoTrack.getCapabilities() : {};
      if (!capabilities.focusMode) {
        return;
      }
      const modes = capabilities.focusMode;
      const focusMode = modes.includes("continuous") ? "continuous" : modes.includes("auto") ? "auto" : "";
      if (!focusMode) {
        return;
      }
      try {
        await videoTrack.applyConstraints({ advanced: [{ focusMode }] });
        showMessage("Đã yêu cầu camera lấy nét. Giữ chắc điện thoại rồi bấm Chụp ảnh.", "success");
      } catch (error) {
        showMessage("Hãy chạm lấy nét trên màn hình và giữ chắc điện thoại trước khi chụp.", "success");
      }
    }

    function waitForVideo() {
      return new Promise((resolve) => {
        if (video.readyState >= 2 && video.videoWidth && video.videoHeight) {
          resolve();
          return;
        }
        video.onloadedmetadata = () => resolve();
      });
    }

    function openDeviceCamera() {
      setLivePreview(false);
      resetZoomControl();
      fallbackInput.click();
      showMessage("Camera hệ thống đang mở. Chạm vào lá để lấy nét, chỉnh zoom rồi chụp ảnh.", "success");
    }

    function formatLocationName(address) {
      if (!address) {
        return "";
      }
      const ward = address.village || address.town || address.suburb || address.quarter || address.neighbourhood;
      const district = address.county || address.city_district || address.district;
      const city = address.city || address.state || address.province;
      return [ward, district, city].filter(Boolean).join(", ");
    }

    async function lookupLocationName(latitude, longitude) {
      const url = new URL("https://nominatim.openstreetmap.org/reverse");
      url.searchParams.set("format", "jsonv2");
      url.searchParams.set("lat", latitude);
      url.searchParams.set("lon", longitude);
      url.searchParams.set("zoom", "16");
      url.searchParams.set("accept-language", "vi");

      const response = await fetch(url.toString());
      if (!response.ok) {
        return "";
      }

      const result = await response.json();
      return formatLocationName(result.address) || result.display_name || "";
    }

    function blobFromCanvas() {
      return new Promise((resolve) => {
        canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.95);
      });
    }

    async function inspectSnapshot() {
      if (!hasSnapshot) {
        snapshotAccepted = false;
        return false;
      }

      const blob = await blobFromCanvas();
      if (!blob) {
        snapshotAccepted = false;
        showMessage("Không đọc được ảnh vừa chụp. Hãy thử lại.", "error");
        return false;
      }

      const data = new FormData();
      data.append("image", blob, "leaf.jpg");

      try {
        const response = await fetch("/inspect", { method: "POST", body: data });
        const result = await response.json();
        snapshotAccepted = Boolean(result.ok);
        showMessage(result.message, result.ok ? "success" : "error");
        return snapshotAccepted;
      } catch (error) {
        snapshotAccepted = false;
        showMessage("Không kiểm tra được ảnh. Hãy thử lại.", "error");
        return false;
      }
    }

    function drawFallbackImage(file) {
      const image = new Image();
      image.onload = async () => {
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        canvas.getContext("2d").drawImage(image, 0, 0);
        canvas.style.display = "block";
        video.style.display = "none";
        setLivePreview(false);
        hasSnapshot = true;
        snapshotAccepted = false;
        document.getElementById("captured-at").value = new Date().toISOString();
        showMessage("Đang kiểm tra ảnh...", "success");
        URL.revokeObjectURL(image.src);
        await inspectSnapshot();
      };
      image.onerror = () => {
        showMessage("Không đọc được ảnh vừa chụp. Hãy thử lại.", "error");
      };
      image.src = URL.createObjectURL(file);
    }

    async function startCamera() {
      if (isIOSDevice()) {
        openDeviceCamera();
        return false;
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        openDeviceCamera();
        return false;
      }

      if (!isSecureCameraContext()) {
        openDeviceCamera();
        return false;
      }

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" } },
          audio: false
        });
        video.srcObject = stream;
        videoTrack = stream.getVideoTracks()[0] || null;
        await waitForVideo();
        await video.play();
        if (videoTrack) {
          await requestFocus();
          setupZoomControl(videoTrack);
        }
        video.style.display = "block";
        canvas.style.display = "none";
        setLivePreview(true);
        hasSnapshot = false;
        snapshotAccepted = false;
        cameraReady = true;
        showMessage("", "");
        return true;
      } catch (error) {
        cameraReady = false;
        videoTrack = null;
        resetZoomControl();
        openDeviceCamera();
        return false;
      }
    }

    function captureImage() {
      if (!cameraReady || !video.videoWidth || !video.videoHeight) {
        showMessage("Camera chưa sẵn sàng. Hãy đợi hình ảnh hiện lên rồi chụp lại.", "error");
        return false;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      canvas.style.display = "block";
      video.style.display = "none";
      setLivePreview(false);
      resetZoomControl();
      hasSnapshot = true;
      snapshotAccepted = false;
      document.getElementById("captured-at").value = new Date().toISOString();
      showMessage("Đang kiểm tra ảnh...", "success");
      inspectSnapshot();
      return true;
    }

    startButton.addEventListener("click", startCamera);
    cameraBox.addEventListener("click", requestFocus);
    zoomRange.addEventListener("input", () => applyCameraZoom(zoomRange.value));
    locationButton.addEventListener("click", () => {
      if (!navigator.geolocation) {
        locationStatus.textContent = "Vị trí: trình duyệt không hỗ trợ GPS.";
        return;
      }

      locationButton.disabled = true;
      locationStatus.textContent = "Vị trí: đang lấy...";
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const latitude = position.coords.latitude;
          const longitude = position.coords.longitude;
          const accuracy = position.coords.accuracy || 0;
          document.getElementById("latitude").value = latitude;
          document.getElementById("longitude").value = longitude;
          document.getElementById("location-accuracy").value = accuracy;
          locationStatus.textContent = `Vị trí: đang tìm tên địa điểm từ ${latitude.toFixed(6)}, ${longitude.toFixed(6)}...`;

          try {
            const locationName = await lookupLocationName(latitude, longitude);
            document.getElementById("location-name").value = locationName;
            locationStatus.textContent = locationName
              ? `Vị trí: ${locationName}; sai số khoảng ${Math.round(accuracy)}m.`
              : `Vị trí: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}; sai số khoảng ${Math.round(accuracy)}m.`;
          } catch (error) {
            locationStatus.textContent = `Vị trí: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}; sai số khoảng ${Math.round(accuracy)}m.`;
          } finally {
            locationButton.disabled = false;
          }
        },
        () => {
          locationStatus.textContent = "Vị trí: chưa được cấp quyền hoặc không lấy được GPS.";
          locationButton.disabled = false;
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
      );
    });

    fallbackInput.addEventListener("change", () => {
      const file = fallbackInput.files && fallbackInput.files[0];
      if (file) {
        drawFallbackImage(file);
      }
    });

    async function handleCaptureClick() {
      if (!stream) {
        const started = await startCamera();
        if (!started) {
          return;
        }
      }
      captureImage();
    }

    captureButton.addEventListener("click", handleCaptureClick);
    captureOverlayButton.addEventListener("click", handleCaptureClick);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      if (!hasSnapshot || !snapshotAccepted) {
        const captured = captureImage();
        if (!captured) {
          return;
        }
        const accepted = await inspectSnapshot();
        if (!accepted) {
          return;
        }
      }

      submitButton.disabled = true;
      showMessage("Đang lưu ảnh...", "success");

      const blob = await blobFromCanvas();
      if (!blob) {
        submitButton.disabled = false;
        showMessage("Không đọc được ảnh vừa chụp. Hãy thử lại.", "error");
        return;
      }

      try {
        const data = new FormData();
        data.append("image", blob, "leaf.jpg");
        data.append("disease", document.getElementById("disease").value);
        data.append("tree_stage", document.getElementById("tree-stage").value);
        data.append("notes", document.getElementById("notes").value);
        data.append("latitude", document.getElementById("latitude").value);
        data.append("longitude", document.getElementById("longitude").value);
        data.append("location_accuracy", document.getElementById("location-accuracy").value);
        data.append("location_name", document.getElementById("location-name").value);
        data.append("captured_at", document.getElementById("captured-at").value || new Date().toISOString());

        const response = await fetch("/submit", { method: "POST", body: data });
        const result = await response.json();
        showMessage(result.message, result.ok ? "success" : "error");
      } catch (error) {
        showMessage("Không gửi được ảnh. Hãy thử lại.", "error");
      } finally {
        submitButton.disabled = false;
      }
    });

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showMessage("Bấm Mở camera để chụp ảnh bằng camera điện thoại.", "success");
    } else if (!isSecureCameraContext()) {
      showMessage("Điện thoại cần HTTPS để mở camera trực tiếp. Bấm Mở camera để dùng camera hệ thống, hoặc mở link https:// sau khi deploy.", "success");
    } else {
      showMessage("Bấm Mở camera để bắt đầu chụp ảnh.", "success");
    }
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE, diseases=DISEASES.keys())


@app.post("/inspect")
def inspect_submission_image():
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify(ok=False, message="Bạn chưa chụp ảnh."), 400

    result = inspect_image(image_file.read())
    status_code = 200 if result.acceptable else 400
    return jsonify(ok=result.acceptable, message=result.message), status_code


@app.post("/submit")
def submit():
    disease_label = request.form.get("disease", "")
    if disease_label not in DISEASES:
        return jsonify(ok=False, message="Tên bệnh không hợp lệ."), 400

    image_file = request.files.get("image")
    if image_file is None:
        return jsonify(ok=False, message="Bạn chưa chụp ảnh."), 400

    result = inspect_image(image_file.read())
    if not result.acceptable or result.image is None:
        return jsonify(ok=False, message=result.message), 400

    try:
        submission_id = save_submission(
            image=result.image,
            disease=DISEASES[disease_label],
            tree_stage=request.form.get("tree_stage"),
            notes=request.form.get("notes"),
            latitude=request.form.get("latitude"),
            longitude=request.form.get("longitude"),
            location_accuracy=request.form.get("location_accuracy"),
            location_name=request.form.get("location_name"),
            captured_at=request.form.get("captured_at"),
        )
    except StorageError as error:
        return jsonify(ok=False, message=f"Không lưu được lên Supabase: {error}"), 502

    return jsonify(
        ok=True,
        message=f"Lưu ảnh thành công! Mã mẫu của bạn là {submission_id[:8]}.",
        submission_id=submission_id,
    )


@app.get("/health")
def health():
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
