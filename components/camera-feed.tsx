"use client"

import type React from "react"
import { useEffect, useRef, useState } from "react"
import { Maximize2, Minimize2, X, Camera, Settings } from "lucide-react"
import { useDetections } from "@/hooks/useDetections"

interface CameraFeedProps {
  isDevelopmentMode: boolean
  isSelected: boolean
  onSelect: () => void
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/detections"
const VIDEO_URL = process.env.NEXT_PUBLIC_VIDEO_URL ?? "http://localhost:8000/preview.mjpg"

export function CameraFeed({ isDevelopmentMode, isSelected, onSelect }: CameraFeedProps) {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showOverlay, setShowOverlay] = useState(true)
  const [overlayPosition, setOverlayPosition] = useState({ x: 50, y: 50 })
  const [overlaySize, setOverlaySize] = useState({ width: 120, height: 120 })
  const [overlayRadius, setOverlayRadius] = useState(12)
  const [overlayBorderThickness, setOverlayBorderThickness] = useState(2)
  const [isDragging, setIsDragging] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })

  // NEW: live detections + drawing
  const { detections, fps, connected } = useDetections(WS_URL)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  // draw boxes whenever detections update
  useEffect(() => {
    const img = imgRef.current
    const canvas = canvasRef.current
    if (!img || !canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // sync canvas to rendered size
    const rect = img.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    detections.forEach((d) => {
      const x = d.box.x * canvas.width
      const y = d.box.y * canvas.height
      const w = d.box.w * canvas.width
      const h = d.box.h * canvas.height

      ctx.lineWidth = 3
      ctx.strokeStyle = d.danger === "high" ? "#ef4444" : d.danger === "med" ? "#f59e0b" : "#22c55e"
      ctx.strokeRect(x, y, w, h)

      const label = `${d.class} ${(d.conf * 100).toFixed(0)}%`
      ctx.font = "14px ui-sans-serif, system-ui"
      ctx.fillStyle = "rgba(0,0,0,0.6)"
      const tw = ctx.measureText(label).width + 8
      ctx.fillRect(x, Math.max(0, y - 18), tw, 18)
      ctx.fillStyle = "#fff"
      ctx.fillText(label, x + 4, Math.max(12, y - 6))
    })
  }, [detections])

  const toggleFullscreen = () => setIsFullscreen((v) => !v)

  const handleClick = (e: React.MouseEvent) => {
    if (isDevelopmentMode) {
      e.stopPropagation()
      onSelect()
    }
  }

  const handleOverlayMouseDown = (e: React.MouseEvent) => {
    if (isDevelopmentMode) {
      e.preventDefault()
      e.stopPropagation()
      setIsDragging(true)
      const rect = e.currentTarget.getBoundingClientRect()
      setDragOffset({ x: e.clientX - rect.left, y: e.clientY - rect.top })
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && isDevelopmentMode) {
      const container = e.currentTarget.getBoundingClientRect()
      setOverlayPosition({
        x: Math.max(0, Math.min(container.width - overlaySize.width, e.clientX - container.left - dragOffset.x)),
        y: Math.max(0, Math.min(container.height - overlaySize.height, e.clientY - container.top - dragOffset.y)),
      })
    }
  }

  const handleMouseUp = () => setIsDragging(false)

  const cameraContent = (
    <div
      className="relative w-full h-full bg-gray-100 rounded-xl border-2 border-blue-500 overflow-hidden"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Live video (backend MJPEG) */}
      <img
        ref={imgRef}
        src={VIDEO_URL}
        alt="camera"
        className="w-full h-full object-cover select-none"
        draggable={false}
      />

      {/* Detection canvas (separate from your dev overlay) */}
      <canvas className="absolute inset-0 pointer-events-none" ref={canvasRef} />

      {/* Connection pill */}
      <div className="absolute left-4 top-4 z-30 px-2 py-1 text-xs rounded-md bg-black/60 text-white">
        {connected ? `WS: connected • ${fps} fps` : "WS: connecting..."}
      </div>

      {/* Fullscreen Toggle */}
      <button
        onClick={toggleFullscreen}
        className="absolute top-4 right-4 w-12 h-12 bg-black/60 text-white rounded-xl flex items-center justify-center hover:bg-black/80 transition-all duration-200 z-30 backdrop-blur-sm"
        aria-label="Toggle fullscreen"
      >
        {isFullscreen ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
      </button>

      {/* YOUR dev overlay (unchanged behavior) */}
      {showOverlay && (
        <div
          className={`absolute bg-black/80 flex items-center justify-center z-20 transition-all duration-200 ${
            isDevelopmentMode ? "cursor-move hover:bg-black/90" : ""
          } ${isDragging ? "cursor-grabbing" : ""}`}
          style={{
            left: overlayPosition.x,
            top: overlayPosition.y,
            width: overlaySize.width,
            height: overlaySize.height,
            borderRadius: `${overlayRadius}px`,
            border: isDevelopmentMode ? `${overlayBorderThickness}px solid #3b82f6` : "none",
          }}
          onMouseDown={handleOverlayMouseDown}
        >
          <X size={32} className="text-white" />
        </div>
      )}
    </div>
  )

  if (isFullscreen) {
    return <div className="fixed inset-0 z-50 bg-black transition-all duration-300 ease-in-out">{cameraContent}</div>
  }

  return (
    <div className="relative h-full">
      <div
        className={`h-full transition-all duration-200 ${isDevelopmentMode && isSelected ? "ring-2 ring-blue-200 rounded-xl" : ""}`}
        onClick={handleClick}
      >
        {cameraContent}
      </div>

      {/* Settings panel (unchanged) */}
      {isDevelopmentMode && isSelected && (
        <div className="absolute -left-2 top-0 -translate-x-full bg-white p-4 rounded-xl border-2 border-blue-500 shadow-xl z-50 w-64">
          <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
            <Settings size={16} />
            Camera Settings
          </h4>

          <div className="space-y-3">
            <div>
              <label className="flex items-center space-x-2">
                <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} className="rounded" />
                <span className="text-sm text-gray-700">Show Overlay (dev)</span>
              </label>
            </div>

            {showOverlay && (
              <>
                <div>
                  <label className="block text-sm font-medium mb-1 text-gray-700">
                    Overlay Width: {overlaySize.width}px
                  </label>
                  <input type="range" min="50" max="300" value={overlaySize.width}
                    onChange={(e) => setOverlaySize((p) => ({ ...p, width: Number.parseInt(e.target.value) }))} className="w-full" />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1 text-gray-700">
                    Overlay Height: {overlaySize.height}px
                  </label>
                  <input type="range" min="50" max="300" value={overlaySize.height}
                    onChange={(e) => setOverlaySize((p) => ({ ...p, height: Number.parseInt(e.target.value) }))} className="w-full" />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1 text-gray-700">
                    Corner Radius: {overlayRadius}px
                  </label>
                  <input type="range" min="0" max="50" value={overlayRadius}
                    onChange={(e) => setOverlayRadius(Number.parseInt(e.target.value))} className="w-full" />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1 text-gray-700">
                    Border Thickness: {overlayBorderThickness}px
                  </label>
                  <input type="range" min="0" max="8" value={overlayBorderThickness}
                    onChange={(e) => setOverlayBorderThickness(Number.parseInt(e.target.value))} className="w-full" />
                </div>

                <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
                  <strong>Tip:</strong> Click and drag the overlay to reposition it
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
