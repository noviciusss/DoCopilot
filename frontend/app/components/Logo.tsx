"use client";

import React from "react";

interface LogoProps {
  size?: "sm" | "md" | "lg";
  showText?: boolean;
  showSubtitle?: boolean;
}

export default function Logo({ size = "md", showText = true, showSubtitle = true }: LogoProps) {
  const iconSizes = {
    sm: "w-7 h-7 rounded-lg",
    md: "w-9 h-9 rounded-xl",
    lg: "w-12 h-12 rounded-2xl",
  };

  const svgSizes = {
    sm: "w-4 h-4",
    md: "w-5 h-5",
    lg: "w-7 h-7",
  };

  const textSizes = {
    sm: "text-sm",
    md: "text-base",
    lg: "text-2xl",
  };

  return (
    <div className="flex items-center gap-3 select-none group">
      {/* Brand Icon Badge */}
      <div className={`relative flex items-center justify-center bg-gradient-to-tr from-violet-600 via-indigo-500 to-cyan-400 text-white shadow-lg shadow-indigo-500/20 ring-1 ring-white/20 transition-transform duration-300 group-hover:scale-105 ${iconSizes[size]}`}>
        {/* Subtle background glow */}
        <div className="absolute inset-0 rounded-inherit bg-gradient-to-tr from-violet-600 to-cyan-400 blur-md opacity-40 group-hover:opacity-75 transition-opacity" />

        {/* Custom SVG Emblem: Overlapping Document Sheet + Copilot Star */}
        <svg className={`relative z-10 ${svgSizes[size]}`} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          {/* Back document sheet */}
          <path d="M7 3H15L19 7V17C19 18.1046 18.1046 19 17 19H7C5.89543 19 5 18.1046 5 17V5C5 3.89543 5.89543 3 7 3Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.6"/>
          {/* Front document sheet fold */}
          <path d="M14 3V8H19" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.8"/>
          {/* Glowing AI Sparkle emblem */}
          <path d="M12 11C12 9.34315 13.3431 8 15 8C13.3431 8 12 6.65685 12 5C12 6.65685 10.6569 8 9 8C10.6569 8 12 9.34315 12 11Z" fill="currentColor"/>
          <path d="M8 15C8 14.1716 8.67157 13.5 9.5 13.5C8.67157 13.5 8 12.8284 8 12C8 12.8284 7.32843 13.5 6.5 13.5C7.32843 13.5 8 14.1716 8 15Z" fill="currentColor" opacity="0.9"/>
        </svg>
      </div>

      {/* Brand Typography */}
      {showText && (
        <div className="flex flex-col leading-none">
          <div className="flex items-center gap-1.5">
            <span className={`font-bold tracking-tight text-white ${textSizes[size]}`}>
              Do<span className="bg-gradient-to-r from-violet-400 via-indigo-300 to-cyan-300 bg-clip-text text-transparent">Copilot</span>
            </span>
            <span className="px-1.5 py-0.5 text-[9px] font-semibold tracking-wider uppercase rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              RAG 2.0
            </span>
          </div>
          {showSubtitle && (
            <span className="text-[10px] text-zinc-400 font-medium tracking-wide mt-0.5">
              Enterprise Document Intelligence
            </span>
          )}
        </div>
      )}
    </div>
  );
}
