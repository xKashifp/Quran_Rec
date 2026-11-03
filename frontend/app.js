const TARGET_RATE = 16000;
const MAX_SECONDS = 90;

const $ = (id) => document.getElementById(id);
const recordBtn = $("record");
const statusEl = $("status");

let ctx = null;
let stream = null;
let node = null;
let source = null;
let chunks = [];
let startedAt = 0;
let timerId = null;
let recording = false;

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("tap", Tap);
`;

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = "status " + kind;
}

async function checkHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    if (h.ready) {
      setStatus("Ready. Press Start and recite.");
      recordBtn.disabled = false;
    } else {
      setStatus("Server is up but the model is not loaded: " + h.error, "error");
    }
  } catch {
    setStatus("Cannot reach the server.", "error");
  }
}

async function start() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
  });
  ctx = new AudioContext();
  const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
  await ctx.audioWorklet.addModule(url);
  source = ctx.createMediaStreamSource(stream);
  node = new AudioWorkletNode(ctx, "tap");
  chunks = [];
  node.port.onmessage = (e) => {
    chunks.push(e.data);
    let peak = 0;
    for (const v of e.data) peak = Math.max(peak, Math.abs(v));
    $("level").style.width = Math.min(100, peak * 140) + "%";
  };
  source.connect(node);
  recording = true;
  startedAt = performance.now();
  timerId = setInterval(tick, 200);
  recordBtn.classList.add("on");
  $("record-label").textContent = "Stop";
  setStatus("Recording... press Stop when you finish.");
}

function tick() {
  const s = (performance.now() - startedAt) / 1000;
  $("timer").textContent = `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  if (s >= MAX_SECONDS) stop();
}

async function stop() {
  if (!recording) return;
  recording = false;
  clearInterval(timerId);
  source.disconnect();
  node.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  const rate = ctx.sampleRate;
  await ctx.close();
  recordBtn.classList.remove("on");
  $("record-label").textContent = "Start";
  $("level").style.width = "0%";

  const length = chunks.reduce((n, c) => n + c.length, 0);
  const pcm = new Float32Array(length);
  let o = 0;
  for (const c of chunks) { pcm.set(c, o); o += c.length; }
  if (pcm.every((v) => Math.abs(v) < 1e-3)) {
    setStatus("No sound was recorded. Check that your microphone is on and selected in the browser.", "error");
    return;
  }
  if (length < rate * 0.5) {
    setStatus("Recording is too short. Try again.", "error");
    return;
  }
  const mono16k = await resample(pcm, rate);
  await send(encodeWav(mono16k, TARGET_RATE));
}

async function resample(pcm, rate) {
  if (rate === TARGET_RATE) return pcm;
  const frames = Math.ceil(pcm.length * TARGET_RATE / rate);
  const off = new OfflineAudioContext(1, frames, TARGET_RATE);
  const buf = off.createBuffer(1, pcm.length, rate);
  buf.copyToChannel(pcm, 0);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}

async function decodeFile(file) {
  const ac = new AudioContext();
  const audio = await ac.decodeAudioData(await file.arrayBuffer());
  await ac.close();
  const mono = new Float32Array(audio.length);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const d = audio.getChannelData(c);
    for (let i = 0; i < d.length; i++) mono[i] += d[i] / audio.numberOfChannels;
  }
  return resample(mono, audio.sampleRate);
}

function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

async function send(wav) {
  setStatus("Recognising...", "busy");
  recordBtn.disabled = true;
  try {
    const form = new FormData();
    form.append("audio", wav, "recitation.wav");
    const r = await fetch("/api/recognize", { method: "POST", body: form });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    render(data);
  } catch (e) {
    setStatus("Error: " + e.message, "error");
  } finally {
    recordBtn.disabled = false;
  }
}

function ref(c) {
  const first = c.ayahs[0], last = c.ayahs[c.ayahs.length - 1];
  if (c.ayahs.length === 1) return `Surah ${first.surah}, Ayah ${first.ayah}`;
  if (first.surah === last.surah) return `Surah ${first.surah}, Ayat ${first.ayah}–${last.ayah}`;
  return `${first.surah}:${first.ayah} – ${last.surah}:${last.ayah}`;
}

function pct(x) { return Math.round(x * 100) + "%"; }

function showBest(c, data) {
  $("best-name-en").textContent = c.primary.surah_name_en;
  $("best-name-ar").textContent = c.primary.surah_name_ar;
  $("best-ref").textContent = ref(c);
  $("best-score").textContent = pct(c.score);
  const text = $("best-text");
  text.innerHTML = "";
  for (const a of c.ayahs) {
    const span = document.createElement("span");
    span.textContent = a.text + " ";
    const num = document.createElement("span");
    num.className = "ayah-num";
    num.textContent = "۝" + a.ayah.toLocaleString("ar-EG");
    text.append(span, num, " ");
  }
  const notes = [];
  if (c.ayahs.length > 1) notes.push(`The recitation spans ${c.ayahs.length} ayat.`);
  else if (c.coverage < 0.85) notes.push(`About ${pct(c.coverage)} of this ayah was recited.`);
  if (data) notes.push(`Confidence: ${data.confidence}.`);
  $("best-note").textContent = notes.join(" ");
  $("ph-ref").textContent = c.reference_phonemes;
}

function render(data) {
  if (!data.results.length) {
    setStatus("No match found. Try reciting a little more clearly or for longer.", "error");
    $("result").hidden = true;
    return;
  }
  setStatus(`Recognised ${data.phoneme_count} phonemes from ${data.duration_seconds}s of audio.`);
  $("result").hidden = false;
  showBest(data.results[0], data);
  const list = $("candidates");
  list.innerHTML = "";
  data.results.forEach((c, i) => {
    const li = document.createElement("li");
    li.className = i === 0 ? "active" : "";
    li.innerHTML = `<div class="c-main"><span class="c-name"></span><span class="c-ref"></span></div>
      <div class="c-bar"><div style="width:${pct(c.score)}"></div></div><span class="c-score">${pct(c.score)}</span>`;
    li.querySelector(".c-name").textContent = `${c.primary.surah_name_en} ${c.primary.surah_name_ar}`;
    li.querySelector(".c-ref").textContent = ref(c);
    li.onclick = () => {
      list.querySelectorAll("li").forEach((x) => x.classList.remove("active"));
      li.classList.add("active");
      showBest(c, i === 0 ? data : null);
    };
    list.appendChild(li);
  });
  $("ph-rec").textContent = data.recognized_phonemes;
  $("timing").textContent = `recognition ${data.timing_ms.recognition} ms · search ${data.timing_ms.search} ms`;
}

recordBtn.onclick = () => (recording ? stop() : start().catch((e) => setStatus("Microphone error: " + e.message, "error")));
$("file").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  setStatus("Decoding file...", "busy");
  try {
    await send(encodeWav(await decodeFile(f), TARGET_RATE));
  } catch (err) {
    setStatus("Could not read that file: " + err.message, "error");
  }
  e.target.value = "";
};

checkHealth();
