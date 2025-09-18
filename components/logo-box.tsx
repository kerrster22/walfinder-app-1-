"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Upload, Palette } from "lucide-react"

interface LogoBoxProps {
  isDevelopmentMode: boolean
  isSelected: boolean
  onSelect: () => void
}

export function LogoBox({ isDevelopmentMode, isSelected, onSelect }: LogoBoxProps) {
  const [logoText, setLogoText] = useState("wAlfinder")
  const [logoImage, setLogoImage] = useState<string | null>(null)
  const [backgroundColor, setBackgroundColor] = useState("#ffffff")
  const [borderColor, setBorderColor] = useState("#3b82f6")
  const fileInputRef = useRef<HTMLInputElement>(null)


  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setLogoImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleClick = (e: React.MouseEvent) => {
    if (isDevelopmentMode) {
      e.stopPropagation()
      onSelect()
    }
  }

  return (
    <div className="relative h-full">
      <div
        className={`w-full h-full rounded-xl border-2 transition-all duration-200 cursor-pointer ${
          isDevelopmentMode && isSelected ? "border-blue-600 shadow-lg ring-2 ring-blue-200" : "border-blue-500"
        }`}
        style={{ backgroundColor }}
        onClick={handleClick}
      >
        <div className="w-full h-full flex items-center justify-center p-4">

            <img src={"/waifinder_logo.png"} alt="Logo" className="max-w-full max-h-full object-contain" />

        </div>
      </div>

      {isDevelopmentMode && isSelected && (
        <div className="absolute -right-2 top-0 translate-x-full bg-white p-4 rounded-xl border-2 border-blue-500 shadow-xl z-50 w-64">
          <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
            <Palette size={16} />
            Logo Settings
          </h4>

          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Logo Text</label>
              <input
                type="text"
                value={logoText}
                onChange={(e) => setLogoText(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter logo text"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Logo Image</label>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2"
              >
                <Upload size={16} />
                Upload Logo
              </button>
              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Background Color</label>
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={backgroundColor}
                  onChange={(e) => setBackgroundColor(e.target.value)}
                  className="w-12 h-8 rounded-lg border border-gray-300 cursor-pointer"
                />
                <input
                  type="text"
                  value={backgroundColor}
                  onChange={(e) => setBackgroundColor(e.target.value)}
                  className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded font-mono"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Border Color</label>
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  value={borderColor}
                  onChange={(e) => setBorderColor(e.target.value)}
                  className="w-12 h-8 rounded-lg border border-gray-300 cursor-pointer"
                />
                <input
                  type="text"
                  value={borderColor}
                  onChange={(e) => setBorderColor(e.target.value)}
                  className="flex-1 px-2 py-1 text-xs border border-gray-300 rounded font-mono"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
