"use client"

import { useEffect, useMemo, useRef, useState } from "react"

export type Danger = "none" | "low" | "med" | "high"

export interface DetectionBox { x: number; y: number; w: number; h: number }
export interface Detection {
  id: string
  class: string
  conf: number
  box: DetectionBox
  distance_m: number | null
  danger: Danger
}
export interface DetectionEvent {
  v: number
  stream_id: string
  ts: number
  detections: Detection[]
  meta: { fps: number }
}

export function useDetections(wsUrl: string) {
  const [event, setEvent] = useState<DetectionEvent | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg?.detections) setEvent(msg as DetectionEvent)
      } catch {}
    }

    return () => ws.close()
  }, [wsUrl])

  // tiny helper so consumers don’t have to null-check
  const api = useMemo(() => ({
    event,
    detections: event?.detections ?? [],
    fps: event?.meta?.fps ?? 0,
    ts: event?.ts ?? 0,
    connected
  }), [event, connected])

  return api
}
