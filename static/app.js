/* Hands-Free Voice-to-Voice Console for Ezee Flights (Zaya)
   With strict speaker-echo gating so Zaya never hears her own TTS voice. */

let activeSessionId = null;
let callStartTime = null;
let lastActivityTime = null;
let timerInterval = null;

// Voice & Turn-Taking State
let micMuted = false;
let canListen = false; // True ONLY when it is the user's turn to speak
let turnGenId = 0; // Incremented every turn to discard any stale STT events

let recognition = null;
let recognitionRunning = false;
let webSpeechSupported = false;
let webSpeechFailed = false;
let osMicLoopActive = false;

let agentIsSpeaking = false;
let agentIsThinking = false;
let lastAgentSpokenText = "";
let lastAgentSpeakEndTs = 0;
let activeHtmlAudio = null;
window.__activeUtterance = null;

let pendingFinalText = "";
let pendingInterimText = "";
let endpointTimer = null;

const ENDPOINT_DELAY_MS = 1600; // Wait 1.6s of silence after user speaks before sending

let ambientAudioCtx = null;
let ambientGainNode = null;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initConsoleControls();
  initDispatchControls();
  initKbControls();
  initCallsControls();
  initEnvModal();

  if ("speechSynthesis" in window) {
    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
  }

  loadSpecInfo();
  loadCallsList();
  runKbSearch();
});

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      buttons.forEach((b) => b.classList.remove("active"));
      panels.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const panel = document.getElementById(`tab-${target}`);
      if (panel) panel.classList.add("active");
    });
  });
}

async function loadSpecInfo() {
  try {
    const res = await fetch("/api/spec");
    const data = await res.json();

    document.getElementById("pillLlm").textContent = `LLM: ${data.llm.model} (${data.llm.api_key_configured ? "OpenAI Live" : "Built-in Spec Engine"})`;
    document.getElementById("pillTts").textContent = `TTS: ${data.tts.model} (${data.tts.api_key_configured ? "ElevenLabs Live" : "Browser Voice"})`;
    document.getElementById("pillStt").textContent = `STT: ${data.stt.provider} (${data.stt.language_hints.join("/")}, ${data.stt.keyterms_count} terms)`;
    document.getElementById("pillOverrides").textContent = `${data.applied_overrides.length} Overrides Active`;

    const overridesList = document.getElementById("overridesList");
    overridesList.innerHTML = "";
    data.applied_overrides.forEach((ov) => {
      const li = document.createElement("li");
      li.textContent = `✓ ${ov}`;
      overridesList.appendChild(li);
    });

    const configCopy = {
      workflow_id: data.workflow_id,
      workflow_name: data.workflow_name,
      llm: data.llm,
      tts: data.tts,
      stt: {
        provider: data.stt.provider,
        language_hints: data.stt.language_hints,
        max_endpoint_delay_ms: data.stt.max_endpoint_delay_ms,
        keyterms_count: data.stt.keyterms_count,
      },
      turn_taking: data.turn_taking,
      limits: data.limits,
      bg_noise: data.bg_noise,
      fillers: data.fillers,
    };
    document.getElementById("specConfigViewer").textContent = JSON.stringify(configCopy, null, 2);
    document.getElementById("promptPreviewBox").textContent = data.rendered_prompt_preview;
  } catch (err) {
    console.error("Failed to load spec:", err);
  }
}

/* ============================================================================
   API KEYS (.ENV) MODAL
   ============================================================================ */

function initEnvModal() {
  const modal = document.getElementById("envModal");
  const openBtn = document.getElementById("btnOpenEnvModal");
  const closeBtn = document.getElementById("btnCloseEnvModal");
  const cancelBtn = document.getElementById("btnCancelEnvModal");
  const form = document.getElementById("envForm");

  openBtn.addEventListener("click", () => (modal.style.display = "grid"));
  closeBtn.addEventListener("click", () => (modal.style.display = "none"));
  cancelBtn.addEventListener("click", () => (modal.style.display = "none"));

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const payload = {
      OPENAI_API_KEY: document.getElementById("envOpenaiKey").value.trim(),
      ELEVEN_API_KEY: document.getElementById("envElevenKey").value.trim(),
      ELEVEN_VOICE_ID: document.getElementById("envElevenVoice").value.trim(),
      DEEPGRAM_API_KEY: document.getElementById("envDeepgramKey").value.trim(),
    };
    await fetch("/api/config/env", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    modal.style.display = "none";
    await loadSpecInfo();
  });
}

/* ============================================================================
   TAB 1: VOICE-TO-VOICE CALL CONSOLE
   ============================================================================ */

function initConsoleControls() {
  document.querySelectorAll(".route-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("consoleRoute").value = btn.getAttribute("data-route");
    });
  });

  document.querySelectorAll(".quick-turn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-text");
      if (!activeSessionId) {
        startCallSession().then(() => submitCallerSpeechOrText(text));
      } else {
        stopAgentSpeech();
        submitCallerSpeechOrText(text);
      }
    });
  });

  document.getElementById("btnStartCall").addEventListener("click", () => startCallSession());
  document.getElementById("btnEndCall").addEventListener("click", () => endCallSession("user_hangup"));

  // Clicking the Voice Orb:
  // - If not in call: starts call
  // - If Zaya is speaking: interrupts Zaya and switches to listening to you
  // - If hearing buffered words: sends your speech immediately
  const voiceOrb = document.getElementById("voiceOrb");
  voiceOrb.style.cursor = "pointer";
  voiceOrb.addEventListener("click", () => {
    if (!activeSessionId) {
      startCallSession();
    } else if (agentIsSpeaking) {
      stopAgentSpeech();
      setTimeout(() => beginUserListeningTurn(), 250);
    } else if ((pendingFinalText + " " + pendingInterimText).trim()) {
      flushPendingSpeech();
    } else {
      beginUserListeningTurn();
    }
  });

  document.getElementById("btnMuteMic").addEventListener("click", () => {
    if (!activeSessionId) return;
    micMuted = !micMuted;
    const muteBtn = document.getElementById("btnMuteMic");
    if (micMuted) {
      muteBtn.textContent = "🔇 Mic Muted (Click to Unmute)";
      muteBtn.className = "btn btn-sm btn-danger";
      abortRecognitionImmediately();
      setVoiceStageState("muted", "Microphone is muted — click 'Mic Muted' to unmute and speak.");
    } else {
      muteBtn.textContent = "🎙️ Mic Active";
      muteBtn.className = "btn btn-sm btn-secondary";
      if (!agentIsSpeaking && !agentIsThinking) {
        beginUserListeningTurn();
      }
    }
  });

  document.getElementById("btnForceSendSpeech").addEventListener("click", () => {
    flushPendingSpeech();
  });

  document.getElementById("consoleLang").addEventListener("change", () => {
    if (activeSessionId && canListen && !micMuted) {
      abortRecognitionImmediately();
      setTimeout(() => beginUserListeningTurn(), 150);
    }
  });

  document.getElementById("toggleOfficeNoise").addEventListener("change", (e) => {
    if (!activeSessionId) return;
    if (e.target.checked) {
      startOfficeAmbience();
    } else {
      stopOfficeAmbience();
    }
  });

  document.getElementById("chatForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = document.getElementById("chatInput");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    stopAgentSpeech();
    submitCallerSpeechOrText(text);
  });

  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  webSpeechSupported = Boolean(SpeechRec);
  webSpeechFailed = !webSpeechSupported;
}

function setVoiceStageState(mode, subtitleText) {
  const orb = document.getElementById("voiceOrb");
  const badge = document.getElementById("voiceModeBadge");
  const hearing = document.getElementById("liveHearingText");
  const dot = document.getElementById("callStateDot");
  const meterFill = document.getElementById("micMeterFill");

  orb.className = "voice-orb";
  badge.className = "voice-mode-badge";

  if (mode === "listening") {
    orb.classList.add("listening");
    orb.textContent = "🎙️";
    badge.classList.add("listening");
    badge.textContent = "🟢 YOUR TURN — SPEAK INTO YOUR MICROPHONE";
    dot.className = "pulse-dot listening";
    meterFill.style.width = "20%";
  } else if (mode === "hearing") {
    orb.classList.add("listening");
    orb.textContent = "🗣️";
    badge.classList.add("listening");
    badge.textContent = "🗣️ HEARING YOU SPEAK... (Pause 1.5s to send, or click Orb)";
    dot.className = "pulse-dot listening";
    meterFill.style.width = `${Math.floor(50 + Math.random() * 45)}%`;
  } else if (mode === "speaking") {
    orb.classList.add("speaking");
    orb.textContent = "🔊";
    badge.classList.add("speaking");
    badge.textContent = "🔊 ZAYA IS SPEAKING (Mic paused to prevent speaker echo — Click 🔊 Orb to interrupt)";
    dot.className = "pulse-dot speaking";
    meterFill.style.width = "0%";
  } else if (mode === "thinking") {
    orb.classList.add("thinking");
    orb.textContent = "⚡";
    badge.classList.add("thinking");
    badge.textContent = "⚡ ZAYA IS THINKING...";
    dot.className = "pulse-dot live";
    meterFill.style.width = "0%";
  } else if (mode === "muted") {
    orb.textContent = "🔇";
    badge.textContent = "🔇 MICROPHONE MUTED";
    dot.className = "pulse-dot";
    meterFill.style.width = "0%";
  } else {
    orb.textContent = "🎙️";
    badge.textContent = "IDLE — CLICK 'START VOICE-TO-VOICE CALL'";
    dot.className = "pulse-dot";
    meterFill.style.width = "0%";
  }

  if (subtitleText !== undefined) {
    hearing.textContent = subtitleText;
  }
}

/* ============================================================================
   ECHO-FREE TURN-BASED SPEECH RECOGNITION
   ============================================================================ */

function isLikelyAgentEcho(heardText) {
  if (!lastAgentSpokenText || !heardText) return false;
  // Only check within 2.5 seconds of Zaya finishing speaking
  if (Date.now() - lastAgentSpeakEndTs > 2500) return false;

  const normHeard = heardText.toLowerCase().replace(/[^a-z0-9\u0600-\u06FF\s]/g, "").trim();
  const normAgent = lastAgentSpokenText.toLowerCase().replace(/[^a-z0-9\u0600-\u06FF\s]/g, "").trim();
  if (normHeard.length >= 12 && normAgent.includes(normHeard)) {
    return true;
  }
  return false;
}

function abortRecognitionImmediately() {
  canListen = false;
  turnGenId++;
  if (endpointTimer) {
    clearTimeout(endpointTimer);
    endpointTimer = null;
  }
  pendingFinalText = "";
  pendingInterimText = "";
  document.getElementById("btnForceSendSpeech").style.display = "none";

  if (recognition) {
    try {
      recognition.onresult = null;
      recognition.onend = null;
      recognition.onerror = null;
      recognition.abort();
    } catch (_) {}
    recognition = null;
  }
  recognitionRunning = false;
}

function beginUserListeningTurn() {
  if (!activeSessionId || micMuted || agentIsSpeaking || agentIsThinking) return;

  abortRecognitionImmediately();
  canListen = true;
  const myGenId = ++turnGenId;

  setVoiceStageState(
    "listening",
    "Speak naturally now — Zaya is listening to you..."
  );

  if (!webSpeechSupported || webSpeechFailed) {
    startOsMicLoop(myGenId);
    return;
  }

  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  const rec = new SpeechRec();
  recognition = rec;
  rec.continuous = true;
  rec.interimResults = true;
  rec.maxAlternatives = 1;
  rec.lang = document.getElementById("consoleLang").value || "en-US";

  rec.onstart = () => {
    if (myGenId !== turnGenId) return;
    recognitionRunning = true;
  };

  rec.onspeechstart = () => {
    if (myGenId !== turnGenId || !canListen) return;
    touchActivity();
    document.getElementById("micMeterFill").style.width = "70%";
  };

  rec.onspeechend = () => {
    if (myGenId !== turnGenId || !canListen) return;
    document.getElementById("micMeterFill").style.width = "20%";
  };

  rec.onerror = (event) => {
    if (myGenId !== turnGenId) return;
    if (
      event.error === "not-allowed" ||
      event.error === "service-not-allowed" ||
      event.error === "network" ||
      event.error === "audio-capture"
    ) {
      webSpeechFailed = true;
      recognitionRunning = false;
      if (activeSessionId && canListen && !micMuted) {
        startOsMicLoop(myGenId);
      }
    }
  };

  rec.onend = () => {
    if (myGenId !== turnGenId) return;
    recognitionRunning = false;

    // If user spoke words and recognition ended naturally, flush and send them!
    const buffered = `${pendingFinalText} ${pendingInterimText}`.trim();
    if (buffered && canListen && activeSessionId && !agentIsSpeaking && !agentIsThinking) {
      flushPendingSpeech();
      return;
    }

    // Otherwise, if still waiting for the user to speak, keep the mic open!
    if (activeSessionId && canListen && !micMuted && !webSpeechFailed && !agentIsSpeaking && !agentIsThinking) {
      setTimeout(() => {
        if (myGenId === turnGenId && activeSessionId && canListen && !micMuted) {
          beginUserListeningTurn();
        }
      }, 150);
    }
  };

  rec.onresult = (event) => {
    // Discard immediately if we are no longer in this user listening turn!
    if (myGenId !== turnGenId || !canListen || !activeSessionId || micMuted || agentIsSpeaking || agentIsThinking) {
      return;
    }

    touchActivity();

    let interimChunk = "";
    let newFinalChunk = "";

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const res = event.results[i];
      const transcript = res[0].transcript;
      if (res.isFinal) {
        newFinalChunk += " " + transcript;
      } else {
        interimChunk += transcript;
      }
    }

    if (newFinalChunk.trim()) {
      pendingFinalText = `${pendingFinalText} ${newFinalChunk.trim()}`.trim();
    }
    pendingInterimText = interimChunk.trim();

    const fullHeard = `${pendingFinalText} ${pendingInterimText}`.trim();
    if (!fullHeard) return;

    // Guard against residual speaker echo
    if (isLikelyAgentEcho(fullHeard)) {
      pendingFinalText = "";
      pendingInterimText = "";
      return;
    }

    setVoiceStageState("hearing", `Hearing you: "${fullHeard}"`);
    document.getElementById("btnForceSendSpeech").style.display = "inline-block";

    if (endpointTimer) clearTimeout(endpointTimer);
    if (document.getElementById("toggleHandsFree").checked) {
      const wordCount = fullHeard.split(/\s+/).filter(Boolean).length;
      const delayMs = wordCount <= 3 && pendingFinalText ? 1050 : ENDPOINT_DELAY_MS;
      endpointTimer = setTimeout(() => {
        if (myGenId === turnGenId && canListen) {
          flushPendingSpeech();
        }
      }, delayMs);
    }
  };

  try {
    rec.start();
  } catch (_) {}
}

async function startOsMicLoop(myGenId) {
  if (osMicLoopActive || !activeSessionId || micMuted || !canListen) return;
  osMicLoopActive = true;

  while (activeSessionId && !micMuted && canListen && myGenId === turnGenId) {
    if (agentIsSpeaking || agentIsThinking) {
      await new Promise((r) => setTimeout(r, 250));
      continue;
    }

    setVoiceStageState(
      "listening",
      "Listening via Windows Microphone — speak naturally now..."
    );

    try {
      const res = await fetch("/api/session/listen-mic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ timeout_sec: 6 }),
      });
      const data = await res.json();
      if (!activeSessionId || micMuted || !canListen || myGenId !== turnGenId) break;

      const text = (data.text || "").trim();
      if (text && !isLikelyAgentEcho(text)) {
        setVoiceStageState("hearing", `Heard you: "${text}"`);
        osMicLoopActive = false;
        await submitCallerSpeechOrText(text);
        return;
      }
    } catch (_) {
      await new Promise((r) => setTimeout(r, 800));
    }
  }

  osMicLoopActive = false;
}

function flushPendingSpeech() {
  const fullText = `${pendingFinalText} ${pendingInterimText}`.trim();
  abortRecognitionImmediately();

  if (!fullText || !activeSessionId) return;
  submitCallerSpeechOrText(fullText);
}

/* ============================================================================
   OFFICE AMBIENCE (BGNoise_Office @ 20% VOLUME)
   ============================================================================ */

function startOfficeAmbience() {
  if (!document.getElementById("toggleOfficeNoise").checked) return;
  stopOfficeAmbience();
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    ambientAudioCtx = new AudioCtx();
    const bufferSize = ambientAudioCtx.sampleRate * 2;
    const buffer = ambientAudioCtx.createBuffer(1, bufferSize, ambientAudioCtx.sampleRate);
    const data = buffer.getChannelData(0);
    let lastOut = 0.0;
    for (let i = 0; i < bufferSize; i++) {
      const white = Math.random() * 2 - 1;
      data[i] = (lastOut + 0.02 * white) / 1.02;
      lastOut = data[i];
      data[i] *= 0.025;
    }
    const noise = ambientAudioCtx.createBufferSource();
    noise.buffer = buffer;
    noise.loop = true;

    const filter = ambientAudioCtx.createBiquadFilter();
    filter.type = "bandpass";
    filter.frequency.value = 420;

    ambientGainNode = ambientAudioCtx.createGain();
    ambientGainNode.gain.value = 0.12;

    noise.connect(filter);
    filter.connect(ambientGainNode);
    ambientGainNode.connect(ambientAudioCtx.destination);
    noise.start();
  } catch (_) {}
}

function stopOfficeAmbience() {
  if (ambientAudioCtx) {
    try {
      ambientAudioCtx.close();
    } catch (_) {}
    ambientAudioCtx = null;
  }
}

/* ============================================================================
   CALL SESSION LIFECYCLE & TURN-TAKING
   ============================================================================ */

function touchActivity() {
  lastActivityTime = Date.now();
}

async function startCallSession() {
  const caller_name = document.getElementById("consoleName").value.trim();
  const enquiry_route = document.getElementById("consoleRoute").value.trim();
  const phone_number = document.getElementById("consolePhone").value.trim();
  const provider = document.getElementById("consoleProvider").value;

  abortRecognitionImmediately();
  stopAgentSpeech();

  document.getElementById("btnStartCall").disabled = true;
  document.getElementById("transcriptBox").innerHTML = "";
  document.getElementById("toolCallsList").innerHTML = '<p class="muted">No <code>search_flights</code> calls executed yet.</p>';
  document.getElementById("liveSummaryBox").innerHTML = '<p class="muted">Call in progress...</p>';

  micMuted = false;

  const res = await fetch("/api/session/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ caller_name, customer_name: caller_name, enquiry_route, phone_number, provider }),
  });
  const data = await res.json();

  activeSessionId = data.session_id;
  callStartTime = Date.now();
  touchActivity();

  document.getElementById("btnEndCall").disabled = false;
  document.getElementById("chatInput").disabled = false;
  document.getElementById("btnSendTurn").disabled = false;
  document.getElementById("btnMuteMic").disabled = false;
  document.getElementById("btnMuteMic").textContent = "🎙️ Mic Active";
  document.getElementById("btnMuteMic").className = "btn btn-sm btn-secondary";

  document.getElementById("callStatusTitle").textContent = `Live Voice Call: ${activeSessionId}`;
  document.getElementById("callStatusSub").textContent = `Engine: ${data.mode} | Route: ${enquiry_route || "Unspecified"}`;

  startOfficeAmbience();

  if (timerInterval) clearInterval(timerInterval);
  timerInterval = setInterval(updateCallWatchdog, 1000);

  // 1. Zaya speaks her opening line while the mic is strictly OFF (preventing speaker echo)
  appendTranscriptMessage("assistant", data.opening);
  await speakText(data.opening);

  // 2. Wait 350ms for speaker room reverb to decay, then open the mic for the user's turn!
  await new Promise((r) => setTimeout(r, 350));
  if (activeSessionId && !micMuted) {
    beginUserListeningTurn();
  }
}

function updateCallWatchdog() {
  if (!activeSessionId) return;
  const elapsedSec = Math.floor((Date.now() - callStartTime) / 1000);
  if (agentIsSpeaking || agentIsThinking || pendingInterimText || pendingFinalText) {
    touchActivity();
  }
  const idleSec = Math.floor((Date.now() - lastActivityTime) / 1000);

  const mins = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
  const secs = String(elapsedSec % 60).padStart(2, "0");
  document.getElementById("durationTimer").textContent = `${mins}:${secs}`;
  document.getElementById("silenceTimer").textContent = `${idleSec}s`;

  if (elapsedSec >= 600) {
    endCallSession("max_call_duration_reached");
  } else if (idleSec >= 30 && !agentIsSpeaking && !agentIsThinking) {
    endCallSession("silence_timeout");
  }
}

async function submitCallerSpeechOrText(text) {
  if (!activeSessionId || !text.trim()) return;
  const currentSession = activeSessionId;

  // Immediately shut off mic recognition while Zaya processes and speaks her reply!
  abortRecognitionImmediately();

  touchActivity();
  agentIsThinking = true;
  appendTranscriptMessage("user", text);
  setVoiceStageState("thinking", `You said: "${text}" — Zaya is replying...`);

  try {
    const res = await fetch("/api/session/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: currentSession, message: text }),
    });
    const data = await res.json();
    agentIsThinking = false;
    touchActivity();

    if (activeSessionId !== currentSession) return;

    if (data.bridge) {
      appendTranscriptMessage("assistant", `${data.bridge} (bridge line)`);
    }

    if (data.tool_calls && data.tool_calls.length > 0) {
      data.tool_calls.forEach((tc) => {
        appendTranscriptMessage("tool", `🔧 ${tc.tool}(query="${tc.query}")`);
        prependToolCard(tc);
      });
    }

    if (data.reply) {
      appendTranscriptMessage("assistant", data.reply);
      // Speak Zaya's reply with mic strictly OFF so she never hears herself!
      await speakText(data.reply);
    }

    // Wait 350ms after Zaya finishes speaking so speaker echo decays, then start listening for user!
    await new Promise((r) => setTimeout(r, 350));
    if (activeSessionId === currentSession && !micMuted) {
      beginUserListeningTurn();
    }
  } catch (err) {
    agentIsThinking = false;
    console.error("Error sending caller turn:", err);
    if (activeSessionId === currentSession && !micMuted) {
      beginUserListeningTurn();
    }
  }
}

function stopAgentSpeech() {
  agentIsSpeaking = false;
  lastAgentSpeakEndTs = Date.now();
  if (activeHtmlAudio) {
    try {
      activeHtmlAudio.pause();
      activeHtmlAudio.src = "";
    } catch (_) {}
    activeHtmlAudio = null;
  }
  if ("speechSynthesis" in window) {
    try {
      window.speechSynthesis.cancel();
    } catch (_) {}
  }
}

async function speakText(text) {
  if (!document.getElementById("toggleTtsAudio").checked || !text) return;

  // Ensure microphone recognition is completely aborted while Zaya is speaking!
  abortRecognitionImmediately();

  touchActivity();
  agentIsSpeaking = true;
  lastAgentSpokenText = text;
  setVoiceStageState("speaking", text);

  // 1. Try ElevenLabs TTS (/api/session/tts) if configured
  try {
    const res = await fetch("/api/session/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (res.status === 200) {
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      await new Promise((resolve) => {
        const audio = new Audio(url);
        activeHtmlAudio = audio;
        audio.onended = () => {
          activeHtmlAudio = null;
          resolve();
        };
        audio.onerror = () => {
          activeHtmlAudio = null;
          resolve();
        };
        audio.play().catch(() => {
          activeHtmlAudio = null;
          resolve();
        });
      });
      agentIsSpeaking = false;
      lastAgentSpeakEndTs = Date.now();
      touchActivity();
      return;
    }
  } catch (_) {}

  // 2. Browser Web Speech Synthesis
  if ("speechSynthesis" in window) {
    await new Promise((resolve) => {
      let settled = false;
      let pollTimer = null;
      let safetyTimer = null;

      const finish = () => {
        if (settled) return;
        settled = true;
        if (pollTimer) clearInterval(pollTimer);
        if (safetyTimer) clearTimeout(safetyTimer);
        window.__activeUtterance = null;
        resolve();
      };

      const utter = new SpeechSynthesisUtterance(text);
      window.__activeUtterance = utter;

      const hasArabic = /[\u0600-\u06FF]/.test(text);
      utter.lang = hasArabic ? "ar-AE" : "en-GB";
      utter.rate = 1.04;

      const voices = window.speechSynthesis.getVoices() || [];
      if (voices.length > 0) {
        const preferred = voices.find((v) =>
          hasArabic
            ? v.lang.startsWith("ar")
            : (v.name.includes("Sonia") ||
               v.name.includes("Hazel") ||
               v.name.includes("Aria") ||
               v.name.includes("Zira") ||
               v.name.includes("Google UK English Female")) &&
              v.lang.startsWith("en")
        );
        if (preferred) utter.voice = preferred;
      }

      const wordCount = text.split(/\s+/).length;
      const maxWaitMs = Math.max(3000, wordCount * 420);
      safetyTimer = setTimeout(finish, maxWaitMs);

      utter.onend = finish;
      utter.onerror = finish;

      window.speechSynthesis.speak(utter);

      const startedAt = Date.now();
      pollTimer = setInterval(() => {
        if (!agentIsSpeaking) {
          finish();
          return;
        }
        if (Date.now() - startedAt > 550 && !window.speechSynthesis.speaking) {
          finish();
        }
      }, 90);
    });
  }

  agentIsSpeaking = false;
  lastAgentSpeakEndTs = Date.now();
  touchActivity();
}

async function endCallSession(reason = "completed") {
  if (!activeSessionId) return;
  const sessionId = activeSessionId;
  activeSessionId = null;

  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }

  abortRecognitionImmediately();
  stopAgentSpeech();
  stopOfficeAmbience();

  document.getElementById("btnStartCall").disabled = false;
  document.getElementById("btnEndCall").disabled = true;
  document.getElementById("chatInput").disabled = true;
  document.getElementById("btnSendTurn").disabled = true;
  document.getElementById("btnMuteMic").disabled = true;
  document.getElementById("callStatusTitle").textContent = `Call Ended (${reason})`;
  setVoiceStageState("idle", "Call ended. Post-call summary has been generated on the right.");

  const res = await fetch("/api/session/end", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, reason }),
  });
  const data = await res.json();

  if (data.summary) {
    document.getElementById("liveSummaryBox").innerHTML = `
      <pre class="json-viewer">${escapeHtml(JSON.stringify(data.summary, null, 2))}</pre>
    `;
    loadCallsList();
  }
}

function appendTranscriptMessage(role, content) {
  const box = document.getElementById("transcriptBox");
  const div = document.createElement("div");

  if (role === "tool") {
    div.className = "msg msg-tool";
    div.textContent = content;
  } else {
    div.className = `msg msg-${role}`;
    const roleLabel = role === "assistant" ? "Zaya (Ezee Flights)" : "Caller (You)";
    div.innerHTML = `<div class="msg-role">${escapeHtml(roleLabel)}</div><div>${escapeHtml(content)}</div>`;
  }

  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function prependToolCard(tc) {
  const container = document.getElementById("toolCallsList");
  if (container.querySelector(".muted")) {
    container.innerHTML = "";
  }
  const card = document.createElement("div");
  card.className = "tool-call-card";
  card.innerHTML = `
    <strong><code>${escapeHtml(tc.tool)}</code></strong>
    <p style="margin: 0.3rem 0;">Query: <code>${escapeHtml(tc.query)}</code></p>
    <pre class="json-viewer" style="max-height: 180px;">${escapeHtml(tc.result)}</pre>
  `;
  container.prepend(card);
}

/* ============================================================================
   TAB 2: OUTBOUND SIP DISPATCH
   ============================================================================ */

function initDispatchControls() {
  const updateCliPreview = () => {
    const phone = document.getElementById("dispPhone").value.trim() || "+9715XXXXXXX";
    const name = document.getElementById("dispName").value.trim() || "Omair";
    const route = document.getElementById("dispRoute").value.trim() || "Dubai to London";
    document.getElementById("dispatchCliPreview").textContent =
      `python dispatch_call.py --to ${phone} --name "${name}" --route "${route}"`;
  };

  ["dispPhone", "dispName", "dispRoute"].forEach((id) => {
    document.getElementById(id).addEventListener("input", updateCliPreview);
  });

  document.getElementById("dispatchForm").addEventListener("submit", (e) => {
    e.preventDefault();
    submitDispatch(false);
  });

  document.getElementById("btnDispatchDry").addEventListener("click", () => {
    submitDispatch(true);
  });
}

async function submitDispatch(dryRun) {
  const phone_number = document.getElementById("dispPhone").value.trim();
  const caller_name = document.getElementById("dispName").value.trim();
  const enquiry_route = document.getElementById("dispRoute").value.trim();
  const trunk_id = document.getElementById("dispTrunk").value.trim();

  const res = await fetch("/api/dispatch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phone_number, caller_name, enquiry_route, trunk_id, dry_run: dryRun }),
  });
  const data = await res.json();
  document.getElementById("dispatchOutput").textContent = JSON.stringify(data, null, 2);
}

/* ============================================================================
   TAB 3: FLIGHT KB & VECTOR SEARCH EXPLORER
   ============================================================================ */

function initKbControls() {
  document.getElementById("kbSearchForm").addEventListener("submit", (e) => {
    e.preventDefault();
    runKbSearch();
  });

  document.querySelectorAll(".kb-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("kbQueryInput").value = btn.getAttribute("data-q");
      runKbSearch();
    });
  });
}

async function runKbSearch() {
  const query = document.getElementById("kbQueryInput").value.trim();
  const provider = document.getElementById("kbProviderMode").value;
  if (!query) return;

  const res = await fetch("/api/search-flights", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, provider }),
  });
  const data = await res.json();

  document.getElementById("kbRawOutput").textContent = data.formatted_output || "";

  const chunksBox = document.getElementById("kbChunksOutput");
  chunksBox.innerHTML = "";
  (data.chunks || []).forEach((ch, idx) => {
    const div = document.createElement("div");
    div.className = "chunk-card";
    div.innerHTML = `
      <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
        <strong>#${idx + 1} — ${escapeHtml(ch.title)}</strong>
        <span class="pill pill-info">Score: ${ch.score}</span>
      </div>
      <div class="muted" style="margin-bottom:0.35rem;">${escapeHtml(ch.section)}</div>
      <div style="font-size:0.8rem; color:#cbd5e1;">${escapeHtml(ch.text)}</div>
    `;
    chunksBox.appendChild(div);
  });
}

/* ============================================================================
   TAB 4: CALL SUMMARIES & 5 TEST SCENARIOS
   ============================================================================ */

function initCallsControls() {
  document.getElementById("btnRefreshCalls").addEventListener("click", () => loadCallsList());
  document.getElementById("btnRunAllTests").addEventListener("click", async () => {
    const btn = document.getElementById("btnRunAllTests");
    btn.disabled = true;
    btn.textContent = "⏳ Running 5 Scenarios...";
    try {
      await fetch("/api/run-tests", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: "kb" }),
      });
      await loadCallsList();
    } finally {
      btn.disabled = false;
      btn.textContent = "▶️ Run 5 Text-Only Test Conversations";
    }
  });
}

async function loadCallsList() {
  const res = await fetch("/api/calls");
  const data = await res.json();
  const container = document.getElementById("callsListContainer");
  container.innerHTML = "";

  const calls = data.calls || [];
  if (calls.length === 0) {
    container.innerHTML = '<p class="muted">No call summaries recorded yet.</p>';
    return;
  }

  calls.forEach((call, index) => {
    const item = document.createElement("div");
    item.className = "call-item";
    const meta = call.metadata || {};
    item.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <strong>${escapeHtml(call.call_id)}</strong>
        <span class="pill pill-success">${escapeHtml(call.outcome || "unknown")}</span>
      </div>
      <div class="muted" style="margin-top:0.25rem;">
        Caller: <strong>${escapeHtml(meta.customer_name || meta.caller_name || "Unknown")}</strong> |
        Route: <strong>${escapeHtml(meta.enquiry_route || "N/A")}</strong>
      </div>
    `;
    item.addEventListener("click", () => {
      document.querySelectorAll(".call-item").forEach((el) => el.classList.remove("selected"));
      item.classList.add("selected");
      renderCallDetail(call);
    });
    container.appendChild(item);

    if (index === 0) {
      item.classList.add("selected");
      renderCallDetail(call);
    }
  });
}

function renderCallDetail(call) {
  const box = document.getElementById("callDetailContainer");
  const transcriptHtml = (call.transcript || [])
    .map(
      (t) => `
      <div style="margin-bottom:0.45rem; padding:0.45rem; background:#090e1c; border-radius:6px; font-size:0.81rem;">
        <strong style="color:var(--accent); text-transform:uppercase;">${escapeHtml(t.role)}:</strong>
        <span>${escapeHtml(t.content)}</span>
      </div>
    `
    )
    .join("");

  box.innerHTML = `
    <div class="chunk-card">
      <p><strong>Call ID:</strong> <code>${escapeHtml(call.call_id)}</code></p>
      <p><strong>Outcome:</strong> <code>${escapeHtml(call.outcome || "N/A")}</code></p>
      <p><strong>Chosen Option:</strong> ${escapeHtml(call.chosen_option || "None")}</p>
      <p><strong>Callback Time:</strong> ${escapeHtml(call.callback_time || "None")}</p>
      <p><strong>Consultant Notes:</strong> ${escapeHtml(call.notes_for_consultant || "")}</p>
    </div>
    <h4 style="margin:0.75rem 0 0.4rem; font-size:0.86rem;">Transcript (${(call.transcript || []).length} events)</h4>
    <div style="max-height:280px; overflow-y:auto; margin-bottom:0.75rem;">${transcriptHtml}</div>
    <h4 style="margin:0.5rem 0 0.4rem; font-size:0.86rem;">Raw JSON (<code>calls/${escapeHtml(call.call_id)}.json</code>)</h4>
    <pre class="json-viewer" style="max-height:220px;">${escapeHtml(JSON.stringify(call, null, 2))}</pre>
  `;
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
