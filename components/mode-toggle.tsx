"use client"

import { Settings, Eye } from "lucide-react"

interface ModeToggleProps {
  isDevelopmentMode: boolean
  onToggle: (mode: boolean) => void
}

export function ModeToggle({ isDevelopmentMode, onToggle }: ModeToggleProps) {
  return (
    <div className="fixed top-4 right-4 z-50">
      <button
        onClick={() => onToggle(!isDevelopmentMode)}
        className={`flex items-center space-x-2 px-4 py-2 rounded-lg border-2 font-medium transition-all duration-200 ${
          isDevelopmentMode
            ? "bg-orange-500 text-white border-orange-500 hover:bg-orange-600"
            : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
        }`}
      >
        {isDevelopmentMode ? <Settings size={18} /> : <Eye size={18} />}
        <span className="text-sm">{isDevelopmentMode ? "Development" : "Production"}</span>
      </button>
    </div>
  )
}
