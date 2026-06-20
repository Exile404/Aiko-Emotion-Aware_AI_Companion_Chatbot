// Aiko avatar — Live2D viewer + WebSocket-driven speech, lipsync, expressions.

const MODEL_URL =
  "https://cdn.jsdelivr.net/gh/guansss/pixi-live2d-display/test/assets/haru/haru_greeter_t03.model3.json";
const WS_URL = "ws://localhost:8765";
const MOUTH_GAIN = 5;       // was 7
const MOUTH_CURVE = 0.9;    // was 0.45 — stop over-lifting; let the mouth close between syllables     // was 4
const MOUTH_GATE = 0.003;   // was 0.006 — keep the quiet tail of each utterance moving the mouth
const EXPR_LERP  = 0.12;   // expression smoothing per frame (~300ms)
const VOWEL_GAIN  = 6;     // brightness deviation → mouth-form sensitivity (main tuning knob)
const VOWEL_DEPTH = 0.3;   // how far vowels move ParamMouthForm around the emotion baseline
const statusEl = document.getElementById("status");
const setStatus = (s) => { statusEl.textContent = s; console.log("[aiko]", s); };

let model = null;
let audioCtx = null;
let mouthValue = 0;
let lipsyncFrame = null;   // set during speech; the bake hook calls it each frame

// --- speech queue: play streamed sentences back-to-back, no overlap ---
let speakQueue = [];
let qBusy = false;
function enqueueSpeak(audioUrl, emotion) {
  speakQueue.push({ audioUrl, emotion });
  if (!qBusy) playNext();
}
function playNext() {
  if (speakQueue.length === 0) { qBusy = false; return; }
  qBusy = true;
  const { audioUrl, emotion } = speakQueue.shift();
  if (emotion) setEmotion(emotion);
  aikoSpeak(audioUrl);
}
window.enqueueSpeak = enqueueSpeak;

// --- emotion -> face param targets. ParamMouthOpenY is a managed BASELINE here;
// during speech the lipsync ticker runs later and overwrites it (so voice wins). ---
const NEUTRAL = {
  ParamMouthForm: 0, ParamMouthOpenY: 0,
  ParamEyeLSmile: 0, ParamEyeRSmile: 0,
  ParamBrowLY: 0, ParamBrowRY: 0, ParamBrowLAngle: 0, ParamBrowRAngle: 0,
  ParamBrowLForm: 0, ParamBrowRForm: 0, ParamEyeForm: 0, ParamTear: 0,
};
const EXPRESSIONS = {
  neutral:   { ...NEUTRAL },
  happy:     { ...NEUTRAL, ParamMouthForm: 1, ParamEyeLSmile: 1, ParamEyeRSmile: 1, ParamBrowLY: 0.4, ParamBrowRY: 0.4 },
  sad:       { ...NEUTRAL, ParamMouthForm: -0.8, ParamBrowLY: -0.3, ParamBrowRY: -0.3, ParamBrowLForm: 0.9, ParamBrowRForm: 0.9, ParamEyeForm: 0.3 },
  // angry: brows fully down + angled, sharp narrowed eyes, tense frown
  angry:     { ...NEUTRAL, ParamMouthForm: -0.7, ParamBrowLY: -1, ParamBrowRY: -1, ParamBrowLForm: -1, ParamBrowRForm: -1, ParamBrowLAngle: 1, ParamBrowRAngle: 1, ParamEyeForm: -1, ParamEyeLOpen: 0.85, ParamEyeROpen: 0.85 },
  // fearful: wide surprised-style eyes + round "ooo" open mouth + raised worried brows
  fearful:   { ...NEUTRAL, ParamMouthForm: -1, ParamMouthOpenY: 0.7, ParamBrowLY: 0.9, ParamBrowRY: 0.9, ParamBrowLForm: 0.4, ParamBrowRForm: 0.4, ParamEyeForm: 0, ParamEyeLOpen: 2, ParamEyeROpen: 2 },
  // surprised: neutral mouth SHAPE but held open + wide eyes + raised brows
  surprised: { ...NEUTRAL, ParamMouthForm: 0, ParamMouthOpenY: 0.7, ParamBrowLY: 1, ParamBrowRY: 1, ParamEyeForm: 0, ParamEyeLOpen: 2, ParamEyeROpen: 2 },
  disgusted: { ...NEUTRAL, ParamMouthForm: -0.7, ParamBrowLY: -0.4, ParamBrowRY: -0.4, ParamBrowLForm: -0.5, ParamBrowRForm: -0.5, ParamEyeForm: -0.5 },
  unknown:   { ...NEUTRAL },
  other:     { ...NEUTRAL },
};
let exprTarget  = { ...EXPRESSIONS.neutral };
let exprCurrent = { ...EXPRESSIONS.neutral };

function setEmotion(name) {
  exprTarget = { ...(EXPRESSIONS[name] || EXPRESSIONS.neutral) };
  console.log("[aiko] emotion:", name);
}
window.setEmotion = setEmotion;

function exprTick() {
  if (!model) return;
  const core = model.internalModel.coreModel;
  for (const id in exprTarget) {
    // seed from the live value so wide-eyes/open-mouth don't snap from 0
    const cur = exprCurrent[id] ?? core.getParameterValueById(id);
    const next = cur + (exprTarget[id] - cur) * EXPR_LERP;
    exprCurrent[id] = next;
    core.setParameterValueById(id, next);       // runs after motion update -> wins
  }
}

async function aikoSpeak(audioUrl) {
  if (!model || !audioCtx) { console.warn("[aiko] not ready"); playNext(); return; }
  const core = model.internalModel.coreModel;

  let buffer;
  try {
    const resp = await fetch(audioUrl);
    buffer = await audioCtx.decodeAudioData(await resp.arrayBuffer());
  } catch (e) { console.warn("[aiko] audio load failed", audioUrl, e.message); playNext(); return; }

  const srcNode = audioCtx.createBufferSource();
  srcNode.buffer = buffer;
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0;
  srcNode.connect(analyser);
  analyser.connect(audioCtx.destination);

  const wave = new Uint8Array(analyser.fftSize);
  const freq = new Uint8Array(analyser.frequencyBinCount);
  const binHz = audioCtx.sampleRate / analyser.fftSize;
  const binOf = (hz) => Math.min(freq.length - 1, Math.round(hz / binHz));
  const LO0 = binOf(300), LO1 = binOf(1200), HI1 = binOf(3000);

  let mouthForm = exprCurrent.ParamMouthForm ?? 0;
  let brightAvg = 0.15;
  let rmsPeak = 0, rmsSum = 0, frames = 0;

  const tick = () => {
    analyser.getByteTimeDomainData(wave);
    let s = 0;
    for (let i = 0; i < wave.length; i++) { const v = (wave[i] - 128) / 128; s += v * v; }
    const rms = Math.sqrt(s / wave.length);
    rmsPeak = Math.max(rmsPeak, rms); rmsSum += rms; frames++;
    const gated = Math.max(0, rms - MOUTH_GATE);
    const openTarget = Math.min(1, Math.pow(gated * MOUTH_GAIN, MOUTH_CURVE));
    mouthValue += (openTarget - mouthValue) * (openTarget > mouthValue ? 0.9 : 0.4);
    core.setParameterValueById("ParamMouthOpenY", mouthValue);

    analyser.getByteFrequencyData(freq);
    let lo = 0, hi = 0;
    for (let i = LO0; i < LO1; i++) lo += freq[i];
    for (let i = LO1; i < HI1; i++) hi += freq[i];
    const bright = (hi + lo) > 0 ? hi / (hi + lo) : brightAvg;
    const speaking = rms > MOUTH_GATE;
    if (speaking) brightAvg += (bright - brightAvg) * 0.05;
    const vowel = speaking ? Math.max(-1, Math.min(1, (bright - brightAvg) * VOWEL_GAIN)) : 0;
    const baseForm = exprCurrent.ParamMouthForm ?? 0;
    const formTarget = Math.max(-1, Math.min(1, baseForm + vowel * VOWEL_DEPTH));
    mouthForm += (formTarget - mouthForm) * 0.25;
    core.setParameterValueById("ParamMouthForm", mouthForm);
  };

  const stop = () => {
    lipsyncFrame = null;
    mouthValue = 0;
    core.setParameterValueById("ParamMouthOpenY", 0);
    try { srcNode.disconnect(); analyser.disconnect(); } catch (_) {}
    if (frames) console.log(`[aiko] lipsync rms — peak ${rmsPeak.toFixed(3)} avg ${(rmsSum/frames).toFixed(3)} (${frames} frames)`);
    playNext();                 // ← play the next queued sentence, if any
  };

  srcNode.onended = stop;
  await audioCtx.resume();
  lipsyncFrame = tick;          // bake hook calls this each frame, right before the mesh bake
  srcNode.start();
}
window.aikoSpeak = aikoSpeak;
function connectWS() {
  const ws = new WebSocket(WS_URL);
  ws.onopen  = () => { console.log("[ws] linked"); setStatus("linked to Python ✓"); };
  ws.onclose = () => { console.log("[ws] closed, retry 2s"); setTimeout(connectWS, 2000); };
  ws.onerror = () => console.log("[ws] error");
  ws.onmessage = (e) => {
    let m; try { m = JSON.parse(e.data); } catch { return; }
    if (m.type === "speak") {
      console.log("[ws] speak:", m.audio, "| emotion:", m.emotion);
      enqueueSpeak(m.audio, m.emotion);
    } else if (m.type === "emotion") {
      setEmotion(m.emotion);
    }
  };
  window._ws = ws;
}

function showEnable() {
  const btn = document.createElement("button");
  btn.textContent = "▶ Enable Aiko";
  btn.style.cssText = "position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:99;font:20px monospace;padding:16px 28px;cursor:pointer";
  btn.onclick = () => {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtx.resume();
    btn.remove();
    connectWS();
    setStatus("audio enabled, WS connecting…");
  };
  document.body.appendChild(btn);
}

(async () => {
  try {
    if (!window.PIXI || !PIXI.live2d) { setStatus("ERROR: PIXI/live2d not loaded"); return; }
    const app = new PIXI.Application({ view: document.getElementById("canvas"), resizeTo: window, backgroundAlpha: 0, antialias: true });
    setStatus("loading model…");
    model = await PIXI.live2d.Live2DModel.from(MODEL_URL);
    app.stage.addChild(model);
    model.anchor.set(0.5, 0.5);
    const fit = () => {
      const s = Math.min(window.innerWidth / model.internalModel.width, window.innerHeight / model.internalModel.height) * 0.9;
      model.scale.set(s); model.position.set(window.innerWidth / 2, window.innerHeight / 2);
    };
    fit(); window.addEventListener("resize", fit);
    // Drive expressions + lipsync from the model's OWN bake step, not a plain
    // ticker — a ticker write gets wiped by Cubism's load/motion/save/bake cycle,
    // which is why the mouth never opened despite the value reading back correct.
    (function installAikoHook() {
      const core = model.internalModel.coreModel;
      if (core.__aikoHook) return;
      const orig = core.update.bind(core);
      core.update = () => {
        exprTick();                        // emotion brows/eyes + mouth baseline
        if (lipsyncFrame) lipsyncFrame();  // voice overrides the mouth during speech
        orig();                            // bake — reads the values we just set
      };
      core.__aikoHook = true;
    })();
    window.aikoModel = model;
    window.aikoApp = app;   // debug handle: lets a single frame be rendered manually
    setStatus("model loaded ✓ — click ▶ Enable Aiko");
    showEnable();
  } catch (e) { setStatus("ERROR: " + (e.message || e)); console.error(e); }
})();