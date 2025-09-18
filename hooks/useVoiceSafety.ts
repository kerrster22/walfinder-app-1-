// hooks/useVoiceSafety.ts
"use client";

import { useEffect, useRef, useState } from "react";
import { coarseSignature, summarizeEvent } from "@/lib/dangerNLG";

// Local copies so we don't depend on app-global types
export type Danger = "none" | "low" | "med" | "high";
export type Box = { x: number; y: number; w: number; h: number };
export type Det = {
  class: string;
  conf: number;
  box: Box;
  danger: Danger;
  count?: number; // for "team"
};

// Accept ANY event shape with ts + detections; meta is flexible/optional
export type EventLike = {
  ts: number;
  detections: Det[];
  meta?: Partial<{
    fps: number;
    max_danger: Danger | "team" | string;
    signature: string;
  }> & Record<string, unknown>;
};

type Options = {
  enabled?: boolean;

  /** Speak at most this often for normal changes (ms) */
  cooldownMs?: number;

  /** How long a change must hold before we consider it "real" (ms) */
  stabilityMs?: number;

  /** If danger jumps (MED→HIGH or team appears), cut through and speak immediately */
  interruptOnSpike?: boolean;

  /** Prefer voices whose name contains this (e.g., "Natural", "Microsoft", "Google") */
  voiceNameContains?: string;

  /** TTS tuning */
  rate?: number;
  pitch?: number;
  volume?: number;

  /** How coarse a "new object" is (bin size for position, 0–1). Smaller = more sensitive. */
  newKeyBin?: number;

  /** Extra cooldown after HIGH danger (ms) to reduce chatter right after urgent alerts */
  postHighCooldownMs?: number;
};

const dangerRank = (x: Danger | "team" | string | undefined) =>
  x === "team" || x === "high" ? 3 : x === "med" ? 2 : x === "low" ? 1 : 0;

/** Build a compact key for a threat to detect "new object appeared" events. */
function threatKey(d: Det, bin = 0.1) {
  const bx = d.box;
  const cx = bx.x + bx.w / 2;
  const cy = bx.y + bx.h / 2;
  const q = (v: number) => Math.round(v / bin);
  // include class + coarse center + coarse size
  return `${d.class}:${q(cx)}:${q(cy)}:${q(bx.w)}:${q(bx.h)}`;
}

/** Pick the nicest-sounding voice we can find. */
function pickBestVoice(
  voices: SpeechSynthesisVoice[],
  hint = "Natural"
): SpeechSynthesisVoice | undefined {
  if (!voices.length) return undefined;

  // Score voices
  const scored = voices.map((v) => {
    let score = 0;
    const name = v.name || "";
    const lang = (v.lang || "").toLowerCase();

    if (/natural|neural/i.test(name)) score += 5000;
    if (hint && name.toLowerCase().includes(hint.toLowerCase())) score += 3000;

    // Prefer Microsoft on Windows Edge, Google on Chrome
    if (/microsoft/i.test(name)) score += 1500;
    if (/google/i.test(name)) score += 1200;

    // Prefer English voices
    if (/^en(-|_|$)/i.test(lang)) score += 800;

    // Prefer non-"default" placeholders
    if (!/default/i.test(name)) score += 200;

    return { v, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored[0]?.v || voices[0];
}

export function useVoiceSafety(event: EventLike | null, opts?: Options) {
  const {
    enabled = true,
    cooldownMs = 6000,        // slightly less strict than before
    stabilityMs = 400,        // quicker to react to a real new object
    interruptOnSpike = true,
    voiceNameContains = "Natural", // try "Natural" first; falls back to "Microsoft"/"Google"
    rate = 0.98,              // slower, more natural cadence
    pitch = 1.04,             // touch brighter
    volume = 1.0,
    newKeyBin = 0.08,         // more sensitive bins for "new object"
    postHighCooldownMs = 9000 // extra quiet period after high-danger alert
  } = opts || {};

  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const lastStableSigRef = useRef<string>("");
  const pendingSigRef = useRef<string>("");
  const pendingSinceRef = useRef<number>(0);
  const lastSpokenAtRef = useRef<number>(0);
  const lastDangerRankRef = useRef<number>(0);
  const lastKeysRef = useRef<Set<string>>(new Set());
  const lastWasHighRef = useRef<boolean>(false);

  // Load available voices (Edge/Chrome expose system voices)
  useEffect(() => {
    const synth = window.speechSynthesis;
    const load = () => setVoices(synth.getVoices());
    load();
    synth.onvoiceschanged = load;
  }, []);

  const speak = (text: string, interrupt = false) => {
    const synth = window.speechSynthesis;
    if (!synth) return;

    // If we need to cut through for danger spikes
    if (interrupt && synth.speaking) synth.cancel();

    const u = new SpeechSynthesisUtterance(text);
    const preferred = pickBestVoice(voices, voiceNameContains) || voices[0];
    if (preferred) u.voice = preferred;

    // Tweak prosody
    u.rate = rate;
    u.pitch = pitch;
    u.volume = volume;

    // Subtle punctuation spacing helps some voices
    u.text = text.replace(/, /g, ", ").replace(/: /g, ": ");

    synth.speak(u);
  };

  useEffect(() => {
    if (!enabled || !event) return;

    // Keep only serious threats (med/high or team)
    const threats = (event.detections || []).filter(
      (d) => d.danger === "med" || d.danger === "high" || d.class === "team"
    );
    if (!threats.length) return;

    const now = Date.now();

    // Signature for "situation changed" (coarse)
    const signature = (event.meta?.signature as string | undefined) ?? coarseSignature(threats);

    // Compute "new threat" keys vs last known keys (finer than signature)
    const keysNow = new Set(threats.map((d) => threatKey(d, newKeyBin)));
    const wasNewThreat = Array.from(keysNow).some((k) => !lastKeysRef.current.has(k));

    // Spike detection: prefer backend-provided max_danger if available
    const metaMax = event.meta?.max_danger as Danger | "team" | string | undefined;
    const currentRank = metaMax
      ? dangerRank(metaMax)
      : Math.max(...threats.map((d) => dangerRank(d.class === "team" ? "team" : d.danger)));
    const spike = currentRank > lastDangerRankRef.current;

    // Stability: require change to hold a short time
    if (signature !== pendingSigRef.current) {
      pendingSigRef.current = signature;
      pendingSinceRef.current = now;
    }
    const stable = now - pendingSinceRef.current >= stabilityMs;

    // Base cooldown; extend after a HIGH alert to keep things calm
    const baseCooldown = lastWasHighRef.current ? Math.max(cooldownMs, postHighCooldownMs) : cooldownMs;

    // Allowed to speak again?
    const cooledDown = now - lastSpokenAtRef.current >= baseCooldown;

    // Speak if:
    //  - a danger spike happens (can interrupt), OR
    //  - a genuinely NEW threat appears (keys) and we're cooled down, OR
    //  - the situation changed (signature) stably and we're cooled down
    const changedVsLastSpoken = stable && signature !== lastStableSigRef.current;

    const shouldSpeak =
      spike ||
      (wasNewThreat && cooledDown) ||
      (changedVsLastSpoken && cooledDown);

    if (!shouldSpeak) {
      // Update memory of keys even if we didn't speak (prevents immediate repeats)
      lastKeysRef.current = keysNow;
      return;
    }

    const text = summarizeEvent(threats);
    if (text) {
      const urgent = interruptOnSpike && spike;
      speak(text, urgent);

      // Update trackers
      lastStableSigRef.current = signature;
      lastSpokenAtRef.current = now;
      lastDangerRankRef.current = currentRank;
      lastKeysRef.current = keysNow;
      lastWasHighRef.current = currentRank >= 3; // high or team
    }
  }, [
    enabled,
    event,
    voices,
    cooldownMs,
    stabilityMs,
    interruptOnSpike,
    voiceNameContains,
    rate,
    pitch,
    volume,
    newKeyBin,
    postHighCooldownMs,
  ]);
}
