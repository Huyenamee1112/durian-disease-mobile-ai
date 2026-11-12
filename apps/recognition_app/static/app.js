const $ = (selector) => document.querySelector(selector);
const careCatalog = window.DURIAN_CARE || { guides: {}, sources: {}, unconfirmed: null };
const state = { file: null, imageUrl: "", diagnosis: null, resultImage: "", resultOrigin: "home" };
function readStored(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function persist(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); return true; }
  catch { notify("Không thể lưu dữ liệu trên thiết bị. Kiểm tra dung lượng hoặc quyền lưu trữ."); return false; }
}
const settings = { autoSpeak: true, rate: 0.9, voice: "", ttsProvider: "gtts", ...readStored("durianSettings", {}) };
// Migrate the retired FPT selection, while preserving an explicit device choice.
if (!["gtts", "device"].includes(settings.ttsProvider)) settings.ttsProvider = "gtts";
delete settings.fptVoice;
// Enable automatic reading once for installations created before this default.
if (settings.autoReadVersion !== 1) {
  settings.autoSpeak = true;
  settings.autoReadVersion = 1;
}
let historyItems = readStored("durianHistory", []);
if (!Array.isArray(historyItems)) historyItems = [];
historyItems = historyItems.filter((item) => item && typeof item === "object").map((item, index) => ({ ...item, id: item.id || `legacy-${index}` }));
function notify(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => toast.classList.add("hidden"), 4500);
}
function navigate(screen) {
  stopSpeech();
  document.querySelectorAll(".screen").forEach((node) => node.classList.toggle("active", node.id === `${screen}-screen`));
  document.querySelectorAll("[data-screen]").forEach((button) => {
    const active = button.dataset.screen === screen || (screen === "result" && button.dataset.screen === state.resultOrigin);
    button.classList.toggle("selected", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}
document.querySelectorAll("[data-screen]").forEach((button) => button.onclick = () => navigate(button.dataset.screen));
const cameraInput = $("#camera-input");
const galleryInput = $("#gallery-input");

$("#camera-button").onclick = () => cameraInput.click();
$("#gallery-button").onclick = () => galleryInput.click();
$("#predict-button").onclick = predict;
$("#back-button").onclick = () => navigate(state.resultOrigin);
$("#retry-button").onclick = showHome;
$("#open-agent").onclick = openAgent;
$("#ask-from-result").onclick = openAgent;
$("#close-agent").onclick = () => $("#agent-panel").classList.add("hidden");
function clearHistory() {
  if (!historyItems.length || !confirm("Xóa toàn bộ lịch sử trên thiết bị này?")) return;
  if (persist("durianHistory", [])) { historyItems = []; renderHistory(); notify("Đã xóa lịch sử."); }
}
$("#clear-history").onclick = clearHistory;
$("#clear-all-history").onclick = clearHistory;
$("#disease-info").onclick = () => navigate("diseases");
$("#settings-button").onclick = () => navigate("settings");

[cameraInput, galleryInput].forEach((input) => input.addEventListener("change", () => selectImage(input.files[0])));

function selectImage(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return alert("Vui lòng chọn tệp ảnh.");
  if (file.size > 12 * 1024 * 1024) return alert("Ảnh phải nhỏ hơn 12 MB.");
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.file = file;
  state.imageUrl = URL.createObjectURL(file);
  $("#image-preview").src = state.imageUrl;
  $("#preview-stage").classList.add("has-image");
  $("#predict-button").classList.remove("hidden");
}

async function predict() {
  if (!state.file) return;
  const button = $("#predict-button");
  button.disabled = true; button.textContent = "Đang phân tích ảnh…";
  $("#model-status").textContent = "Đang gửi ảnh tới mô hình";
  const submittedFile = state.file;
  const submittedImage = state.imageUrl;
  const body = new FormData();
  body.append("image", submittedFile);
  body.append("model", window.DURIAN_CONFIG.defaultBackend);
  try {
    const response = await fetch("/api/predict", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.message || "Nhận diện thất bại");
    state.diagnosis = payload.result;
    state.resultOrigin = "home";
    state.resultImage = submittedImage;
    renderResult(payload.result);
    if (settings.autoSpeak) speakDiagnosis();
    await saveLocalHistory(payload.result, submittedFile);
  } catch (error) {
    alert(error.message);
    $("#model-status").textContent = "Không thể nhận diện, vui lòng thử lại";
  } finally {
    button.disabled = false; button.textContent = "Nhận diện ảnh này";
  }
}

function renderResult(result) {
  const isHealthy = result.label === "Leaf_Healthy" || result.disease_name === "Lá khỏe";
  $("#result-image").src = state.resultImage || "";
  $("#result-image").classList.toggle("hidden", !state.resultImage);
  $("#disease-name").classList.toggle("healthy-result", isHealthy);
  $("#disease-name").textContent = isHealthy
    ? "Lá khỏe"
    : result.uncertain
      ? `Chưa chắc chắn: ${result.disease_name}`
      : `Nghi mắc ${result.disease_name}`;
  $("#confidence").textContent = `${Math.round(result.confidence * 100)}%`;
  const care = careGuideFor(result);
  $("#guidance").textContent = care?.summary || result.guidance;
  renderCareGuide($("#care-details"), care);
  $("#uncertain-warning").classList.toggle("hidden", !result.uncertain);
  $("#probabilities").innerHTML = result.probabilities.map((item) => {
    const percent = Math.round(item.probability * 100);
    return `<div class="prob-row"><span>${item.name}</span><div class="bar"><i style="width:${percent}%"></i></div><b>${percent}%</b></div>`;
  }).join("");
  $("#model-status").textContent = `Đã nhận diện bằng ${result.model}`;
  renderRelatedArticles(result);
  navigate("result");
}

function showHome() { navigate("home"); }

function speakDevice(text) {
  if (!("speechSynthesis" in window)) return notify("Trình duyệt này chưa hỗ trợ đọc văn bản.");
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "vi-VN";
  utterance.rate = Number(settings.rate);
  utterance.voice = speechSynthesis.getVoices().find((voice) => voice.voiceURI === settings.voice)
    || speechSynthesis.getVoices().find((voice) => voice.lang.toLowerCase().startsWith("vi")) || null;
  utterance.onerror = (event) => {
    if (!["interrupted", "canceled"].includes(event.error)) notify("Không phát được giọng đọc. Hãy thử giọng khác hoặc kiểm tra thiết bị.");
  };
  speechSynthesis.speak(utterance);
}
let speechController = null;
const cloudAudio = $("#cloud-audio");
function speechStatus(message) {
  $("#speech-status").textContent = message;
  $("#voice-preview-status").textContent = message;
}
function stopSpeech() {
  const previous = speechController;
  speechController = null;
  if (previous) previous.abort();
  cloudAudio.oncanplay = null;
  cloudAudio.onerror = null;
  cloudAudio.onplaying = null;
  cloudAudio.onended = null;
  cloudAudio.pause();
  cloudAudio.removeAttribute("src");
  cloudAudio.load();
  cloudAudio.classList.add("hidden");
  if ("speechSynthesis" in window) speechSynthesis.cancel();
  speechStatus("");
}
$("#stop-speech").onclick = stopSpeech;
$("#stop-preview").onclick = stopSpeech;
function playCloudAudio(url, controller) {
  const signal = controller.signal;
  return new Promise((resolve, reject) => {
    let timer;
    let settled = false;
    const current = () => !signal.aborted && speechController === controller;
    function finish(error) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      cloudAudio.oncanplay = null;
      cloudAudio.onerror = null;
      signal.removeEventListener("abort", aborted);
      if (error) reject(error); else resolve();
    }
    function aborted() { finish(new DOMException("Cancelled", "AbortError")); }
    signal.addEventListener("abort", aborted, { once: true });
    if (!current()) return aborted();
    cloudAudio.onerror = () => finish(new Error("Không tải được MP3 Google."));
    cloudAudio.oncanplay = () => {
      cloudAudio.oncanplay = null;
      if (!current()) return aborted();
      cloudAudio.playbackRate = Number(settings.rate);
      cloudAudio.play().then(() => {
        if (!current()) return aborted();
        speechStatus("Đang đọc bằng giọng Google.");
        finish();
      }).catch((error) => {
        if (!current()) return aborted();
        if (error.name === "NotAllowedError") {
          speechStatus("Âm thanh đã sẵn sàng. Nhấn ▶ trên thanh phát để nghe.");
          notify("Âm thanh đã sẵn sàng. Nhấn ▶ trên thanh phát để nghe.");
          finish();
        } else finish(error);
      });
    };
    timer = setTimeout(() => finish(new Error("Hết thời gian tải âm thanh Google.")), 15000);
    cloudAudio.classList.remove("hidden");
    cloudAudio.src = url;
    cloudAudio.load();
  });
}
async function speak(text) {
  stopSpeech();
  if (settings.ttsProvider === "device") return speakDevice(text);
  if (!window.DURIAN_CONFIG.tts?.available) {
    speechStatus("Máy chủ chưa cài gTTS; đang dùng giọng thiết bị.");
    notify("Máy chủ chưa cài gTTS. Đang dùng giọng thiết bị.");
    return speakDevice(text);
  }
  const controller = new AbortController();
  speechController = controller;
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 40000);
  function fallback(message) {
    if (speechController !== controller) return;
    cloudAudio.oncanplay = null;
    cloudAudio.onerror = null;
    cloudAudio.pause();
    cloudAudio.classList.add("hidden");
    speechStatus(message + " Đang dùng giọng thiết bị.");
    notify(message + " Đang dùng giọng thiết bị dự phòng.");
    speakDevice(text);
  }
  speechStatus("Đã có kết quả. Đang tạo giọng đọc Google…");
  try {
    const response = await fetch("/api/tts", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }), signal: controller.signal,
    });
    const payload = await response.json();
    if (speechController !== controller || controller.signal.aborted) return;
    if (!response.ok || !payload.ok) throw new Error(payload.message || "Không tạo được âm thanh.");
    if (!/^\/api\/tts\/audio\/[a-f0-9]{64}\.mp3$/.test(payload.audio_url)) {
      throw new Error("Địa chỉ âm thanh không hợp lệ.");
    }
    await playCloudAudio(payload.audio_url, controller);
    if (speechController !== controller || controller.signal.aborted) return;
    cloudAudio.onplaying = () => {
      if (speechController === controller) speechStatus("Đang đọc bằng giọng Google.");
    };
    cloudAudio.onended = () => {
      if (speechController === controller) speechStatus("Đã đọc xong. Có thể nhấn Nghe để nghe lại.");
    };
    cloudAudio.onerror = () => fallback("Phát âm thanh Google bị gián đoạn.");
  } catch (error) {
    if (speechController !== controller) return;
    if (controller.signal.aborted && !timedOut) return;
    fallback(timedOut ? "Hết thời gian tạo giọng Google." : error.message);
  } finally {
    clearTimeout(timer);
  }
}
$("#tts-provider").value = settings.ttsProvider;
$("#tts-provider").onchange = (event) => {
  stopSpeech(); settings.ttsProvider = event.target.value;
  persist("durianSettings", settings); loadVoices();
};
function speakDiagnosis() {
  if (!state.diagnosis) return;
  const care = careGuideFor(state.diagnosis);
  const summary = state.diagnosis.uncertain ? "Kết quả nhận diện chưa chắc chắn." : state.diagnosis.summary;
  const guidance = care ? `${care.summary} ${care.caution} Nguồn tham khảo có dưới kết quả.` : state.diagnosis.guidance;
  speak(`${summary} Độ tin cậy ${Math.round(state.diagnosis.confidence * 100)} phần trăm. ${guidance}`);
}
$("#speak-button").onclick = speakDiagnosis;
persist("durianSettings", settings);
$("#auto-speak").checked = Boolean(settings.autoSpeak);
$("#speech-rate").value = settings.rate;
$("#rate-value").textContent = `${Number(settings.rate).toFixed(2)}×`;
$("#auto-speak").onchange = (event) => { settings.autoSpeak = event.target.checked; persist("durianSettings", settings); };
$("#speech-rate").oninput = (event) => {
  settings.rate = Number(event.target.value);
  $("#rate-value").textContent = `${settings.rate.toFixed(2)}×`;
  cloudAudio.playbackRate = settings.rate;
  persist("durianSettings", settings);
};
$("#voice-select").onchange = (event) => { settings.voice = event.target.value; persist("durianSettings", settings); };
$("#test-voice").onclick = () => speak("Xin chào, Durian AI hỗ trợ bạn kiểm tra dấu hiệu bệnh trên lá sầu riêng.");
$("#settings-model").textContent = `Mô hình mặc định: ${window.DURIAN_CONFIG.defaultBackend}`;
function loadVoices() {
  const supported = "speechSynthesis" in window;
  const voices = supported ? speechSynthesis.getVoices().filter((voice) => voice.lang.toLowerCase().startsWith("vi")) : [];
  const select = $("#voice-select");
  select.replaceChildren(new Option("Tự chọn giọng tiếng Việt", ""));
  voices.forEach((voice) => select.add(new Option(voice.name, voice.voiceURI)));
  select.value = voices.some((voice) => voice.voiceURI === settings.voice) ? settings.voice : "";
  $("#voice-status").textContent = !supported ? "Trình duyệt chưa hỗ trợ giọng đọc." : voices.length ? "Giọng đọc do thiết bị cung cấp, không cần Google Cloud." : "Thiết bị chưa cung cấp giọng tiếng Việt. Bạn có thể thêm giọng trong cài đặt hệ điều hành; chất lượng giọng mặc định tùy thiết bị.";
  if (settings.ttsProvider === "gtts") {
    $("#voice-status").textContent = window.DURIAN_CONFIG.tts?.available
      ? "Giọng Google tiếng Việt qua gTTS. Cần Internet khi tạo âm thanh mới; MP3 đã lưu được dùng lại."
      : "Máy chủ chưa cài gTTS. Tạm dùng giọng thiết bị.";
  }
  $("#voice-select").disabled = !supported;
}
if ("speechSynthesis" in window) speechSynthesis.addEventListener("voiceschanged", loadVoices);
loadVoices();

let chatHistory = [];
let chatController = null;
let chatAnswer = "";
let chatDiagnosis = {};
function openAgent() {
  $("#agent-panel").classList.remove("hidden");
  if (!chatHistory.length) chatDiagnosis = state.diagnosis || {};
  $("#agent-context").textContent = chatDiagnosis.disease_name ? `Đang trao đổi về kết quả: ${chatDiagnosis.disease_name}. Tạo cuộc trò chuyện mới để dùng kết quả mới nhất.` : "Bạn có thể hỏi về triệu chứng, chăm sóc, nước tưới và dinh dưỡng sầu riêng.";
  $("#agent-question").focus();
}
function chatMessage(role, text) {
  const message = document.createElement("div");
  message.className = `chat-message chat-${role}`;
  const name = document.createElement("b"); name.textContent = role === "user" ? "Bạn" : "Trợ lý";
  const body = document.createElement("p"); body.textContent = text;
  message.append(name, body); $("#agent-answer").append(message);
  message.scrollIntoView?.({ block: "nearest" });
  return message;
}
$("#agent-reset").onclick = () => {
  chatController?.abort(); chatController = null; stopSpeech();
  chatHistory = []; chatAnswer = "";
  $("#agent-answer").textContent = "Bạn muốn hỏi gì về sầu riêng?";
  $("#agent-status").textContent = ""; $("#agent-listen").disabled = true;
  $("#agent-send").disabled = false; openAgent();
};
$("#close-agent").onclick = () => { stopSpeech(); $("#agent-panel").classList.add("hidden"); };
$("#agent-listen").onclick = () => { if (chatAnswer) speak(chatAnswer.slice(0, 2000)); };
$("#agent-stop").onclick = stopSpeech;
$("#agent-form").onsubmit = async (event) => {
  event.preventDefault();
  const question = $("#agent-question").value.trim();
  if (!question || chatController) return;
  if (!chatHistory.length) $("#agent-answer").textContent = "";
  const userMessage = chatMessage("user", question);
  const controller = new AbortController(); chatController = controller;
  const timer = setTimeout(() => controller.abort(), 45000);
  $("#agent-send").disabled = true; $("#agent-status").textContent = "Gemini đang trả lời…";
  try {
    const response = await fetch("/api/agent", { method: "POST", signal: controller.signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, diagnosis: chatDiagnosis, history: chatHistory.slice(-12) }) });
    const payload = await response.json();
    if (chatController !== controller) return;
    if (!response.ok || !payload.ok) throw new Error(payload.message || "Không thể gửi câu hỏi.");
    if (payload.source !== "gemini") throw new Error(payload.answer || "Trợ lý chưa sẵn sàng.");
    chatAnswer = payload.answer;
    const message = chatMessage("assistant", chatAnswer);
    for (const source of payload.sources || []) { const link = document.createElement("a"); if (!source.url?.startsWith("https://")) continue; link.href = source.url; link.target = "_blank"; link.rel = "noopener noreferrer"; link.textContent = `[${source.id}] ${source.title}`; message.append(link); }
    chatHistory.push({role: "user", content: question}, {role: "assistant", content: chatAnswer});
    chatHistory = chatHistory.slice(-12);
    $("#agent-question").value = ""; $("#agent-listen").disabled = false;
    $("#agent-status").textContent = "Đã trả lời bằng Gemini. Bạn có thể hỏi tiếp hoặc nhấn Nghe.";
  } catch (error) {
    if (chatController !== controller) return;
    userMessage.remove();
    $("#agent-status").textContent = error.name === "AbortError" ? "Chờ quá lâu. Hãy thử gửi lại." : error.message;
  } finally {
    clearTimeout(timer);
    if (chatController === controller) { chatController = null; $("#agent-send").disabled = false; }
  }
};

async function makeThumbnail(file) {
  if (!file) return "";
  const bitmap = await createImageBitmap(file);
  try {
    const canvas = document.createElement("canvas");
    const ratio = Math.min(1, 480 / Math.max(bitmap.width, bitmap.height));
    canvas.width = Math.round(bitmap.width * ratio); canvas.height = Math.round(bitmap.height * ratio);
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.72);
  } finally { bitmap.close(); }
}
async function saveLocalHistory(result, file) {
  let image = "";
  try { image = await makeThumbnail(file); } catch { notify("Đã nhận diện; thiết bị không tạo được ảnh lưu lịch sử."); }
  const entry = { id: window.crypto?.randomUUID?.() || String(Date.now()), ...result, image, at: new Date().toISOString() };
  const next = [entry, ...historyItems].slice(0, 20);
  if (persist("durianHistory", next)) { historyItems = next; renderHistory(); }
}
function renderHistory() {
  $("#history-count").textContent = `${historyItems.length} lần nhận diện`;
  $("#clear-all-history").disabled = !historyItems.length;
  $("#clear-history").disabled = !historyItems.length;
  for (const [selector, items] of [["#history-list", historyItems.slice(0, 3)], ["#full-history-list", historyItems]]) {
    const list = $(selector);
    list.replaceChildren();
    list.className = items.length ? "" : "empty-state";
    if (!items.length) { list.textContent = "Chưa có lần nhận diện nào. Chụp hoặc chọn ảnh để bắt đầu."; continue; }
    items.forEach((item) => {
      const article = document.createElement("article"); article.className = "history-item";
      const open = document.createElement("button"); open.className = "history-open";
      const image = document.createElement("img");
      image.src = typeof item.image === "string" && item.image.startsWith("data:image/") ? item.image : "/static/icons/03_lich_su.png";
      image.alt = ""; image.className = "history-thumb";
      const content = document.createElement("div");
      const title = document.createElement("b"); title.textContent = item.disease_name;
      const meta = document.createElement("small"); meta.textContent = `${Math.round(item.confidence * 100)}% · ${item.model}`;
      const date = document.createElement("small"); date.textContent = new Date(item.at).toLocaleString("vi-VN");
      content.append(title, meta, date); open.append(image, content);
      open.onclick = () => {
        if (!item.probabilities || !item.guidance) return notify("Bản ghi cũ chỉ có thông tin tóm tắt. Những lần nhận diện mới sẽ lưu đầy đủ.");
        state.diagnosis = item; state.resultImage = item.image; state.resultOrigin = selector === "#history-list" ? "home" : "history";
        renderResult(item);
      };
      const remove = document.createElement("button"); remove.className = "delete-history"; remove.textContent = "Xóa";
      remove.setAttribute("aria-label", `Xóa lần nhận diện ${item.disease_name}`);
      remove.onclick = () => {
        if (!confirm("Xóa lần nhận diện này?")) return;
        const next = historyItems.filter((entry) => entry.id !== item.id);
        if (persist("durianHistory", next)) { historyItems = next; renderHistory(); }
      };
      article.append(open, remove); list.append(article);
    });
  }
}
function catalogGuide(raw) {
  if (!raw) return null;
  return {
    ...raw, checked_at: careCatalog.checked_at,
    sources: Object.entries(raw.source_sections || {}).map(([id, section]) =>
      ({ ...careCatalog.sources[id], id, section })),
  };
}
function careGuideFor(result) {
  // Never show a disease-specific treatment path for low-confidence results.
  if (result.uncertain || !careCatalog.guides[result.label]) {
    return catalogGuide(careCatalog.unconfirmed);
  }
  return result.care_guide || catalogGuide(careCatalog.guides[result.label]);
}
function sourceLink(source, text) {
  const link = document.createElement("a");
  if (typeof source.url === "string" && source.url.startsWith("https://")) link.href = source.url;
  link.target = "_blank"; link.rel = "noopener noreferrer";
  link.textContent = text;
  link.title = `${source.publisher} · ${source.section || ""}`;
  return link;
}
function renderCareGuide(container, guide, detailed = false) {
  container.replaceChildren();
  if (!guide) return;
  const intro = document.createElement("p"); intro.className = "care-intro";
  intro.textContent = guide.intro; container.append(intro);
  const sources = guide.sources || [];
  for (const [key, title, numbered] of [
    [detailed && guide.appearance ? "appearance" : "symptoms",
      detailed ? "Hình dạng và dấu hiệu nhận biết" : "Đối chiếu dấu hiệu", false],
    ["steps", detailed ? "Hướng dẫn xử lý từng bước" : "Việc nên làm", true],
  ]) {
    const heading = document.createElement("h4"); heading.textContent = title;
    const list = document.createElement(numbered ? "ol" : "ul"); list.className = "care-list";
    for (const entry of guide[key] || []) {
      const item = document.createElement("li");
      const text = document.createElement("span");
      if (entry.title) {
        const label = document.createElement("strong"); label.textContent = entry.title + ": ";
        item.append(label);
      }
      text.textContent = entry.text; item.append(text);
      for (const id of entry.source_ids || []) {
        const index = sources.findIndex(source => source.id === id);
        if (index >= 0) {
          const link = sourceLink(sources[index], ` [${index + 1}]`);
          link.className = "care-citation"; item.append(link);
        }
      }
      if (!(entry.source_ids || []).length) {
        const tag = document.createElement("small"); tag.className = "care-editorial";
        tag.textContent = "Gợi ý ghi nhận của ứng dụng"; item.append(tag);
      }
      list.append(item);
    }
    container.append(heading, list);
  }
  if (detailed && guide.differential) {
    const title = document.createElement("h4"); title.textContent = "Điểm dễ nhầm";
    const text = document.createElement("p"); text.className = "care-caution";
    text.textContent = guide.differential; container.append(title, text);
  }
  const help = document.createElement("div"); help.className = "care-help";
  const helpHeading = document.createElement("h4"); helpHeading.textContent = "Khi nào cần hỗ trợ?";
  const helpText = document.createElement("p"); helpText.textContent = guide.seek_help;
  help.append(helpHeading, helpText);
  const caution = document.createElement("p"); caution.className = "care-caution"; caution.textContent = guide.caution;
  container.append(help, caution);
  const references = document.createElement("details"); references.className = "care-references";
  const summary = document.createElement("summary"); summary.textContent = `Nguồn tham khảo (${sources.length})`;
  references.append(summary);
  sources.forEach((source, index) => {
    const block = document.createElement("div"); block.className = "care-source";
    block.append(sourceLink(source, `[${index + 1}] ${source.title}`));
    for (const value of [source.publisher, source.published, source.section, source.scope]) {
      if (!value) continue;
      const line = document.createElement("small"); line.textContent = value; block.append(line);
    }
    references.append(block);
  });
  const checked = document.createElement("small"); checked.className = "care-reviewed";
  checked.textContent = `Nguồn được đối chiếu ngày ${guide.checked_at || careCatalog.checked_at}. Nội dung tham khảo, chưa thay thế kiểm tra tại vườn.`;
  references.append(checked);
  container.append(references);
}
function renderDiseases() {
  const term = $("#disease-search").value.trim().toLocaleLowerCase("vi");
  const list = $("#disease-list"); list.replaceChildren();
  Object.values(careCatalog.guides)
    .filter(entry => [entry.name, entry.intro, entry.summary, ...entry.symptoms.map(v => v.text),
      ...entry.steps.map(v => v.text), ...(entry.appearance || []).map(v => v.text), entry.differential || ""].join(" ").toLocaleLowerCase("vi").includes(term))
    .forEach(raw => {
      const card = document.createElement("details"); card.className = "info-card";
      const title = document.createElement("summary"); title.className = "disease-summary";
      const name = document.createElement("span"); name.className = "disease-name"; name.textContent = raw.name;
      const preview = document.createElement("small"); preview.className = "disease-preview";
      preview.textContent = raw.short_description;
      const hint = document.createElement("small"); hint.className = "disease-hint";
      hint.textContent = "Mở để xem dấu hiệu và cách xử lý";
      title.append(name, preview, hint);
      const content = document.createElement("div");
      renderCareGuide(content, catalogGuide(raw), true);
      card.append(title, content); list.append(card);
    });
  if (!list.children.length) {
    const empty = document.createElement("p"); empty.className = "empty-state";
    empty.textContent = "Không tìm thấy nội dung phù hợp."; list.append(empty);
  }
}
$("#disease-search").oninput = renderDiseases;
renderDiseases();
renderHistory();

// Use DOM text nodes so source titles and summaries are never interpreted as HTML.
const articles = window.DURIAN_ARTICLES || [];
function articleCard(article, compact = false) {
  const card = document.createElement("article");
  card.className = "info-card article-card" + (compact ? " compact" : "");
  const badge = document.createElement("span"); badge.className = "article-badge";
  badge.textContent = article.category + " · " + article.type;
  const title = document.createElement("h3"); title.textContent = article.title;
  const source = document.createElement("p"); source.className = "article-source";
  source.textContent = article.source + " · " + (article.date || "Ngày đăng: chưa xác minh");
  const summary = document.createElement("p"); summary.textContent = article.summary;
  const link = document.createElement("a");
  link.href = article.url; link.target = "_blank"; link.rel = "noopener noreferrer";
  link.textContent = article.type.includes("PDF") ? "Mở tài liệu gốc (PDF) ↗" : "Đọc bài gốc ↗";
  link.setAttribute("aria-label", "Đọc bài gốc: " + article.title + " (mở tab mới)");
  card.append(badge, title, source);
  if (!compact) card.append(summary);
  card.append(link);
  return card;
}
function renderArticles() {
  const query = $("#article-search").value.trim().toLocaleLowerCase("vi");
  const category = $("#article-category").value;
  const filtered = articles.filter((article) =>
    (!category || article.category === category) &&
    [article.title, article.source, article.summary, ...article.tags].join(" ").toLocaleLowerCase("vi").includes(query));
  const list = $("#article-list"); list.replaceChildren();
  $("#article-count").textContent = filtered.length + " bài phù hợp";
  filtered.forEach((article) => list.append(articleCard(article)));
  if (!filtered.length) {
    const empty = document.createElement("p"); empty.className = "empty-state";
    empty.textContent = "Chưa có bài phù hợp. Thử từ khóa khác hoặc chọn tất cả chủ đề.";
    list.append(empty);
  }
}
function renderRelatedArticles(result) {
  const list = $("#related-reading"); list.replaceChildren();
  const related = articles.filter((article) => article.tags.includes(result.disease_name)).slice(0, 2);
  if (!related.length) return;
  const title = document.createElement("h3"); title.textContent = "Đọc thêm về kết quả";
  list.append(title);
  related.forEach((article) => list.append(articleCard(article, true)));
}
$("#article-search").oninput = renderArticles;
$("#article-category").onchange = renderArticles;
renderArticles();
