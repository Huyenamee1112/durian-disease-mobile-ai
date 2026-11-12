// Run with: node --test apps/recognition_app/test_speech.cjs
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const careData = JSON.parse(fs.readFileSync(path.join(__dirname, "knowledge_base/care_guides.json"), "utf8"));
const audioURL = "/api/tts/audio/" + "a".repeat(64) + ".mp3";

function setup(saved = {}) {
  class Element {
    constructor() {
      this.children = []; this.value = ""; this.dataset = {}; this.disabled = false;
      this.classes = new Set();
      this.classList = {
        add: name => this.classes.add(name), remove: name => this.classes.delete(name),
        toggle: (name, on) => on ? this.classes.add(name) : this.classes.delete(name),
      };
      this.playCalls = 0;
    }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    add(node) { this.children.push(node); }
    addEventListener() {} setAttribute() {}
    removeAttribute(name) { delete this[name]; }
    pause() { this.paused = true; }
    load() { if (this.src) queueMicrotask(() => this.oncanplay?.()); }
    play() { this.playCalls++; this.paused = false; return Promise.resolve(); }
  }
  const elements = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const timers = new Map();
  let timerId = 0;
  const spoken = [], requests = [];
  const storage = new Map([["durianSettings", JSON.stringify(saved)]]);
  const context = {
    console, AbortController, DOMException,
    setTimeout: (fn, delay) => { timers.set(++timerId, { fn, delay }); return timerId; },
    clearTimeout: id => timers.delete(id),
    Option: function(label, value) { this.label = label; this.value = value; },
    document: { querySelector: get, querySelectorAll: () => [], createElement: () => new Element() },
    window: { DURIAN_CARE: careData, DURIAN_CONFIG: { defaultBackend: "dinov2", tts: { available: true } }, scrollTo() {} },
    localStorage: { getItem: k => storage.get(k) || null, setItem: (k, v) => storage.set(k, v) },
    confirm: () => true, alert: message => { throw new Error(message); },
    URL: {}, FormData: class { append() {} },
    SpeechSynthesisUtterance: function(text) { this.text = text; },
    speechSynthesis: { cancel() {}, getVoices: () => [], addEventListener() {}, speak: u => spoken.push(u.text) },
  };
  context.window.speechSynthesis = context.speechSynthesis;
  context.fetch = async (url, options) => {
    requests.push({ url, options });
    return { ok: true, json: async () => ({ ok: true, audio_url: audioURL }) };
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "static/app.js"), "utf8"), context);
  return { context, get, spoken, requests, timers, run: code => vm.runInContext(code, context) };
}
const tick = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };

test("migrates FPT setting to gTTS and preserves explicit device selection", () => {
  assert.equal(setup({ ttsProvider: "fpt" }).run("settings.ttsProvider"), "gtts");
  assert.equal(setup({ ttsProvider: "device" }).run("settings.ttsProvider"), "device");
});
test("Google audio plays via local URL; no voice/key sent", async () => {
  const h = setup();
  await h.run('speak("Xin chào")');
  assert.equal(h.get("#cloud-audio").playCalls, 1);
  assert.equal(h.spoken.length, 0);
  assert.deepEqual(JSON.parse(h.requests[0].options.body), { text: "Xin chào" });
  assert.equal(h.get("#speech-status").textContent, "Đang đọc bằng giọng Google.");
});
test("autoplay restriction leaves native play controls without device fallback", async () => {
  const h = setup();
  h.get("#cloud-audio").play = () => Promise.reject(new DOMException("Blocked", "NotAllowedError"));
  await h.run('speak("Xin chào")');
  assert.equal(h.spoken.length, 0);
  assert.equal(h.get("#cloud-audio").classes.has("hidden"), false);
  assert.match(h.get("#speech-status").textContent, /Nhấn ▶/);
});
test("network failure uses device voice", async () => {
  const h = setup();
  h.context.fetch = async () => { throw new Error("Offline"); };
  await h.run('speak("Xin chào")');
  assert.deepEqual(h.spoken, ["Xin chào"]);
});
test("switching screens cancels request and ignores late response", async () => {
  const h = setup();
  let respond;
  h.context.fetch = () => new Promise(resolve => { respond = resolve; });
  const pending = h.run('speak("Giọng cũ")');
  h.run('navigate("home")');
  respond({ ok: true, json: async () => ({ ok: true, audio_url: audioURL }) });
  await pending;
  assert.equal(h.get("#cloud-audio").playCalls, 0);
  assert.equal(h.spoken.length, 0);
  assert.equal(h.get("#speech-status").textContent, "");
});
test("stop during pending play promise cannot change status or trigger fallback", async () => {
  const h = setup();
  let finish;
  h.get("#cloud-audio").play = () => new Promise(resolve => { finish = resolve; });
  const pending = h.run('speak("Xin chào")');
  await tick();
  h.run("stopSpeech()");
  finish();
  await pending;
  assert.equal(h.get("#speech-status").textContent, "");
  assert.equal(h.spoken.length, 0);
});
test("request timeout aborts and falls back; explicit stop does not", async () => {
  const h = setup();
  h.context.fetch = (_url, opts) => new Promise((_resolve, reject) =>
    opts.signal.addEventListener("abort", () => reject(new DOMException("Cancelled", "AbortError"))));
  const pending = h.run('speak("Xin chào")');
  [...h.timers.values()].find(t => t.delay === 40000).fn();
  await pending;
  assert.deepEqual(h.spoken, ["Xin chào"]);
});
test("new request supersedes old audio without overlapping speech", async () => {
  const h = setup();
  const resolvers = [];
  h.context.fetch = () => new Promise(resolve => resolvers.push(resolve));
  const old = h.run('speak("Cũ")');
  const current = h.run('speak("Mới")');
  const response = { ok: true, json: async () => ({ ok: true, audio_url: audioURL }) };
  resolvers[1](response); await current;
  resolvers[0](response); await old;
  assert.equal(h.get("#cloud-audio").playCalls, 1);
  assert.equal(h.spoken.length, 0);
});
test("diagnosis and history finish while TTS remains pending", async () => {
  const h = setup();
  h.context.fetch = (url, options) => {
    h.requests.push({ url, options });
    if (url === "/api/tts") return new Promise(() => {});
    return Promise.resolve({ ok: true, json: async () => ({ ok: true, result: {
      label: "Leaf_Healthy", disease_name: "Lá khỏe", confidence: .9, summary: "Lá khỏe",
      guidance: "Theo dõi cây.", uncertain: false, probabilities: [], model: "test",
    } }) });
  };
  h.run('state.file = { type: "image/jpeg" }; state.imageUrl = "blob:test"');
  await h.run("predict()");
  assert.equal(h.get("#disease-name").textContent, "Lá khỏe");
  assert.equal(h.run("historyItems.length"), 1);
  assert.equal(h.get("#predict-button").disabled, false);
  assert.equal(h.requests.filter(r => r.url === "/api/tts").length, 1);
  h.run("stopSpeech()");
});


test("sourced care renders links and shares catalog with disease lookup", () => {
  const h = setup();
  const guide = h.run('careGuideFor({label:"Leaf_Algal",uncertain:false})');
  h.run('renderCareGuide($("#care-details"), careGuideFor({label:"Leaf_Algal",uncertain:false}))');
  assert(guide.sources.length >= 2);
  assert(h.get("#care-details").children.length > 5);
  assert.equal(h.get("#disease-list").children.length, 4);
});
test("old history and low-confidence results get appropriate sourced guidance", () => {
  const h = setup();
  assert.equal(h.run('careGuideFor({label:"Leaf_Healthy"}).name'), "Lá khỏe");
  assert.equal(h.run('careGuideFor({label:"Leaf_Healthy",uncertain:true}).name'), "Cần kiểm tra thêm");
  assert.equal(h.run('careGuideFor({label:"Leaf_Colletotrichum",uncertain:true,care_guide:{name:"wrong"}}).name'), "Cần kiểm tra thêm");
});
test("voice for uncertain diagnosis reads generic guidance, not old treatment text", async () => {
  const h = setup({ttsProvider:"device"});
  h.run('state.diagnosis={label:"Leaf_Colletotrichum",uncertain:true,summary:"Cũ",confidence:.3,guidance:"Nội dung xử lý cũ"};speakDiagnosis()');
  assert.match(h.spoken[0], /chưa chắc chắn/);
  assert(!h.spoken[0].includes("Nội dung xử lý cũ"));
  assert(!h.spoken[0].includes("https://"));
});

test("disease cards show descriptions, full morphology and searchable signs", () => {
  const h = setup();
  const card = h.get("#disease-list").children[0];
  assert.equal(card.children[0].children[0].textContent, "Đốm rong");
  assert.match(card.children[0].children[1].textContent, /gạch tôm/);
  const headings = card.children[1].children.map(node => node.textContent).filter(Boolean);
  assert(headings.includes("Hình dạng và dấu hiệu nhận biết"));
  assert(headings.includes("Hướng dẫn xử lý từng bước"));
  assert(headings.includes("Điểm dễ nhầm"));
  h.get("#disease-search").value = "đồng tâm";
  h.run("renderDiseases()");
  assert.equal(h.get("#disease-list").children.length, 1);
  assert.equal(h.get("#disease-list").children[0].children[0].children[0].textContent, "Thán thư");
});
