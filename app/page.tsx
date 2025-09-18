"use client"

import React, { useEffect, useState } from "react"
import { LogoBox } from "@/components/logo-box"
import { AIDetectionFeed } from "@/components/ai-detection-feed"
import { CameraFeed } from "@/components/camera-feed"
import { ModeToggle } from "@/components/mode-toggle"

import { useDetections } from "@/hooks/useDetections"
import { useVoiceSafety } from "@/hooks/useVoiceSafety"

export default function Home() {
  const [isDevelopmentMode, setIsDevelopmentMode] = useState(false)
  const [backgroundColor, setBackgroundColor] = useState("#ffffff")
  const [selectedElement, setSelectedElement] = useState<string | null>(null)
  const [voiceOn, setVoiceOn] = useState(true)
  const [voiceFilter, setVoiceFilter] = useState("Microsoft") // try "Natural" if available

  const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://127.0.0.1:8000/ws/detections"
  const { event } = useDetections(WS_URL)

  // Speak only on meaningful change or danger spike; no heartbeat chatter.
  useVoiceSafety(event ?? null, {
    enabled: voiceOn,
    cooldownMs: 8000,        // won't speak again for at least 8s (unless spike)
    stabilityMs: 900,        // change must persist ~0.9s
    interruptOnSpike: true,  // cut through if danger jumps
    voiceNameContains: voiceFilter,
    rate: 1.0,
    pitch: 1.03,
  })

  // For the voice picker dropdown, read available voices (just for display)
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([])
  useEffect(() => {
    const synth = window.speechSynthesis
    const load = () => setVoices(synth.getVoices())
    load()
    synth.onvoiceschanged = load
  }, [])

  const distinctVendors = Array.from(
    new Set(
      voices.map(v => {
        // vendor-ish token from voice name
        const m = v.name.match(/^[A-Za-z]+/)
        return m?.[0] || v.name
      })
    )
  ).sort()

  return (
    <div
      className="min-h-screen p-6 transition-colors duration-300"
      style={{ backgroundColor }}
      onClick={() => setSelectedElement(null)}
    >
      {/* Top bar: Mode + Voice controls */}
      <div className="flex items-center justify-between gap-4">
        <ModeToggle isDevelopmentMode={isDevelopmentMode} onToggle={setIsDevelopmentMode} />

        <div className="flex items-center gap-3">
          <label className="text-sm text-zinc-500">Voice</label>
          <select
            value={voiceFilter}
            onChange={(e) => setVoiceFilter(e.target.value)}
            className="px-2 py-1 rounded border bg-white text-sm"
            title="Pick a voice family to prefer"
          >
            {[...new Set(["Natural", "Microsoft", "Google", ...distinctVendors])].map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>

          <button
            onClick={() => setVoiceOn(v => !v)}
            className={`px-3 py-2 rounded-lg text-sm font-medium border ${
              voiceOn
                ? "bg-green-600 text-white border-green-600 hover:bg-green-700"
                : "bg-zinc-800 text-zinc-100 border-zinc-700 hover:bg-zinc-700"
            }`}
            aria-pressed={voiceOn}
            title="Toggle voice guidance"
          >
            {voiceOn ? "Voice: On" : "Voice: Off"}
          </button>
        </div>
      </div>

      {/* Main Layout */}
      <div className="flex gap-6 h-[calc(100vh-3rem)] mt-4">
        {/* Left Sidebar */}
        <div className="flex flex-col gap-6 w-80">
          <div className="h-32">
            <LogoBox
              isDevelopmentMode={isDevelopmentMode}
              isSelected={selectedElement === "logo"}
              onSelect={() => setSelectedElement("logo")}
            />
          </div>

          <div className="flex-1">
            <AIDetectionFeed
              isDevelopmentMode={isDevelopmentMode}
              isSelected={selectedElement === "ai-feed"}
              onSelect={() => setSelectedElement("ai-feed")}
            />
          </div>
        </div>

        {/* Camera Feed */}
        <div className="flex-1">
          <CameraFeed
            isDevelopmentMode={isDevelopmentMode}
            isSelected={selectedElement === "camera"}
            onSelect={() => setSelectedElement("camera")}
          />
        </div>
      </div>

      {/* Development Mode Background Color Control */}
      {isDevelopmentMode && (
        <div className="fixed bottom-6 left-6 bg-white p-4 rounded-xl border-2 border-blue-500 shadow-xl z-50">
          <label className="block text-sm font-semibold mb-2 text-gray-700">Background Color</label>
          <div className="flex items-center gap-2">
            <input
              type="color"
              value={backgroundColor}
              onChange={(e) => setBackgroundColor(e.target.value)}
              className="w-12 h-8 rounded-lg border-2 border-gray-300 cursor-pointer"
            />
            <input
              type="text"
              value={backgroundColor}
              onChange={(e) => setBackgroundColor(e.target.value)}
              className="px-2 py-1 text-xs border rounded font-mono"
              placeholder="#ffffff"
            />
          </div>
        </div>
      )}
    </div>
  )
}
