"use client"

import { useEffect, useRef, useState } from "react"
import { Type } from "lucide-react"
import { useDetections } from "@/hooks/useDetections"

interface AIDetectionFeedProps {
  isDevelopmentMode: boolean
  isSelected: boolean
  onSelect: () => void
}

type LogEntry = { id: string; timestamp: Date; message: string }

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/detections"

export function AIDetectionFeed({ isDevelopmentMode, isSelected, onSelect }: AIDetectionFeedProps) {
  const { event, detections } = useDetections(WS_URL)

  const [logs, setLogs] = useState<LogEntry[]>([])
  const [isExpanded, setIsExpanded] = useState(false)
  const [fontFamily, setFontFamily] = useState("Inter")
  const [fontSize, setFontSize] = useState(14)
  const [fontWeight, setFontWeight] = useState("normal")
  const [textColor, setTextColor] = useState("#1f2937")
  const [textAlign, setTextAlign] = useState<"left" | "center" | "right">("left")
  const [expandButtonPosition, setExpandButtonPosition] = useState<"top-left" | "top-right" | "bottom-left" | "bottom-right">("top-right")
  const [showHistoryButton, setShowHistoryButton] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Dev mode: keep your fake generator
  useEffect(() => {
    if (!isDevelopmentMode) return
    const interval = setInterval(() => {
      const messages = [
        "Dev: Person entering frame",
        "Dev: Motion in zone A",
        "Dev: Confidence 94%",
      ]
      const entry: LogEntry = {
        id: `${Date.now()}`,
        timestamp: new Date(),
        message: messages[Math.floor(Math.random() * messages.length)],
      }
      setLogs((prev) => [...prev.slice(-99), entry])
    }, 1500)
    return () => clearInterval(interval)
  }, [isDevelopmentMode])

  // Production mode: append real detections to the feed
  useEffect(() => {
    if (isDevelopmentMode || !event) return
    const t = new Date(event.ts)
    if (detections.length === 0) return
    setLogs((prev) => [
      ...prev.slice(-90),
      ...detections.map((d) => ({
        id: d.id,
        timestamp: t,
        message: `${d.class} ${(d.conf * 100).toFixed(0)}% • danger: ${d.danger}`,
      })),
    ])
  }, [event, detections, isDevelopmentMode])

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current && !isExpanded) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs, isExpanded])

  const handleClick = (e: React.MouseEvent) => {
    if (isDevelopmentMode) {
      e.stopPropagation()
      onSelect()
    }
  }

  return (
    <div className="relative h-full">
      <div
        className={`w-full h-[753px] bg-white rounded-xl border-2 transition-all duration-200 ${
          isDevelopmentMode && isSelected ? "border-blue-600 shadow-lg ring-2 ring-blue-200" : "border-blue-500"
        }`}
        onClick={handleClick}
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          <h3 className="font-semibold text-gray-800">AI Detection Feed</h3>
        </div>

        {/* Feed Content */}
        <div
          ref={scrollRef}
          className={`flex-1 p-4 overflow-y-auto space-y-2 ${isExpanded ? "h-96" : "h-[calc(100%-4rem)]"}`}
          style={{ fontFamily, fontSize: `${fontSize}px`, fontWeight, color: textColor, textAlign }}
        >
          {logs.map((log) => (
            <div key={log.id} className="text-sm leading-relaxed">
              <span className="text-gray-500 text-xs font-mono">{log.timestamp.toLocaleTimeString()}</span>{" "}
              <span>{log.message}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Dev settings panel (unchanged controls) */}
      {isDevelopmentMode && isSelected && (
        <div className="absolute -right-2 top-0 translate-x-full bg-white p-4 rounded-xl border-2 border-blue-500 shadow-xl z-50 w-72">
          <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
            <Type size={16} />
            Feed Settings
          </h4>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Font Family</label>
              <select value={fontFamily} onChange={(e) => setFontFamily(e.target.value)} className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                <option value="Inter">Inter</option>
                <option value="Arial">Arial</option>
                <option value="Helvetica">Helvetica</option>
                <option value="Georgia">Georgia</option>
                <option value="monospace">Monospace</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Font Size: {fontSize}px</label>
              <input type="range" min="10" max="24" value={fontSize} onChange={(e) => setFontSize(Number.parseInt(e.target.value))} className="w-full" />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Font Weight</label>
              <select value={fontWeight} onChange={(e) => setFontWeight(e.target.value)} className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                <option value="300">Light</option>
                <option value="normal">Normal</option>
                <option value="500">Medium</option>
                <option value="600">Semi Bold</option>
                <option value="bold">Bold</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Text Color</label>
              <div className="flex items-center gap-2">
                <input type="color" value={textColor} onChange={(e) => setTextColor(e.target.value)} className="w-12 h-8 rounded-lg border border-gray-300 cursor-pointer" />
                <input type="text" value={textColor} onChange={(e) => setTextColor(e.target.value)} className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded font-mono" />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Text Alignment</label>
              <select value={textAlign} onChange={(e) => setTextAlign(e.target.value as "left" | "center" | "right")} className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                <option value="left">Left</option>
                <option value="center">Center</option>
                <option value="right">Right</option>
              </select>
            </div>

            <div>
              <label className="flex items-center space-x-2">
                <input type="checkbox" checked={showHistoryButton} onChange={(e) => setShowHistoryButton(e.target.checked)} className="rounded" />
                <span className="text-sm text-gray-700">Show History Button</span>
              </label>
            </div>

            {showHistoryButton && (
              <div>
                <label className="block text-sm font-medium mb-1 text-gray-700">History Button Position</label>
                <select value={expandButtonPosition} onChange={(e) => setExpandButtonPosition(e.target.value as any)} className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500">
                  <option value="top-left">Top Left</option>
                  <option value="top-right">Top Right</option>
                  <option value="bottom-left">Bottom Left</option>
                  <option value="bottom-right">Bottom Right</option>
                </select>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
