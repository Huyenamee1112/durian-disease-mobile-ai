const $ = (selector) => document.querySelector(selector);
const state = { file: null, imageUrl: "", diagnosis: null, resultImage: "", resultOrigin: "home" };
function readStored(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function persist(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); return true; }
  catch { notify("Không thể lưu dữ liệu trên thiết bị. Kiểm tra dung lượng hoặc quyền lưu trữ."); return false; }
}
const settings = { autoSpeak: true, rate: 0.9, voice: "", ...readStored("durianSettings", {}) };
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
  if ("speechSynthesis" in window) speechSynthesis.cancel();
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
  $("#guidance").textContent = result.guidance;
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

function speak(text) {
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
function speakDiagnosis() {
  if (!state.diagnosis) return;
  speak(`${state.diagnosis.summary} Độ tin cậy ${Math.round(state.diagnosis.confidence * 100)} phần trăm. ${state.diagnosis.guidance}`);
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
  ["#auto-speak", "#speech-rate", "#voice-select", "#test-voice"].forEach((id) => $(id).disabled = !supported);
}
if ("speechSynthesis" in window) speechSynthesis.addEventListener("voiceschanged", loadVoices);
loadVoices();

function openAgent() { $("#agent-panel").classList.remove("hidden"); $("#agent-question").focus(); }
$("#agent-form").onsubmit = async (event) => {
  event.preventDefault();
  const question = $("#agent-question").value.trim();
  if (!question) return;
  $("#agent-answer").textContent = "Đang chuẩn bị câu trả lời…";
  try {
    const response = await fetch("/api/agent", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, diagnosis: state.diagnosis || {} }) });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload.message || "Không thể gửi câu hỏi.");
    $("#agent-answer").textContent = payload.answer || payload.message;
  } catch (_) { $("#agent-answer").textContent = "Không kết nối được trợ lý. Vui lòng thử lại."; }
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
const diseases = [
  ["Đốm rong", "Các đốm trên bề mặt lá cần được quan sát rõ cả hai mặt.", "Theo dõi vùng đốm, giữ vườn thông thoáng và nhờ cán bộ nông nghiệp kiểm tra trước khi xử lý."],
  ["Cháy lá", "Vùng lá khô hoặc cháy cần được đối chiếu với độ ẩm và tình trạng toàn cây.", "Kiểm tra độ ẩm, khả năng thoát nước và mức độ lan rộng trên các lá lân cận. Tên lớp không xác định được tác nhân gây hại."],
  ["Thán thư", "Vết tổn thương trên lá có thể giống các bệnh khác; ảnh đơn lẻ chưa đủ để xác nhận.", "Ghi nhận diễn biến và trao đổi với người có chuyên môn để xác nhận nguyên nhân, biện pháp xử lý."],
  ["Lá khỏe", "Mô hình chưa nhận thấy dấu hiệu thuộc ba nhóm bệnh được hỗ trợ.", "Tiếp tục theo dõi cây. Chụp lại nếu xuất hiện triệu chứng mới; kết quả này không loại trừ mọi bệnh hoặc thiếu dinh dưỡng."],
];
function renderDiseases() {
  const term = $("#disease-search").value.trim().toLocaleLowerCase("vi");
  const list = $("#disease-list"); list.replaceChildren();
  diseases.filter((entry) => entry.join(" ").toLocaleLowerCase("vi").includes(term)).forEach(([name, description, guidance]) => {
    const card = document.createElement("details"); card.className = "info-card";
    const title = document.createElement("summary"); title.textContent = name;
    const descriptionNode = document.createElement("p"); descriptionNode.textContent = description;
    const guidanceNode = document.createElement("p"); guidanceNode.textContent = guidance;
    card.append(title, descriptionNode, guidanceNode); list.append(card);
  });
  if (!list.children.length) { const empty = document.createElement("p"); empty.className = "empty-state"; empty.textContent = "Không tìm thấy nội dung phù hợp."; list.append(empty); }
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
