"use client";

import React from "react";

interface LogoProps {
  size?: "sm" | "md" | "lg";
  showText?: boolean;
  collapsed?: boolean;
}

export default function Logo({ size = "md", showText = true, collapsed = false }: LogoProps) {
  const iconDim = { sm: 28, md: 32, lg: 40 }[size];
  const textClass = { sm: "text-sm", md: "text-sm", lg: "text-lg" }[size];

  return (
    <div className="flex items-center gap-2.5 select-none" aria-label="Docopilot">
      {/* Mark: stacked document shape */}
      <svg
        width={iconDim}
        height={iconDim}
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
        style={{ flexShrink: 0 }}
      >
        <rect width="32" height="32" rx="7" fill="var(--surface-2)" />
        <rect x="1" y="1" width="30" height="30" rx="6" stroke="var(--border)" strokeWidth="1" />
        {/* Back doc */}
        <rect x="9" y="8" width="13" height="16" rx="1.5" fill="var(--border)" opacity="0.5" />
        {/* Front doc */}
        <rect x="7" y="6" width="13" height="16" rx="1.5" fill="var(--text-2)" opacity="0.9" />
        {/* Fold corner */}
        <path d="M17 6 L20 9 L17 9 Z" fill="var(--surface-2)" />
        <path d="M17 6 L20 9 H17 Z" stroke="var(--border)" strokeWidth="0.75" />
        {/* Lines */}
        <rect x="9" y="12" width="7" height="1" rx="0.5" fill="var(--ink)" opacity="0.35" />
        <rect x="9" y="14.5" width="9" height="1" rx="0.5" fill="var(--ink)" opacity="0.25" />
        <rect x="9" y="17" width="5" height="1" rx="0.5" fill="var(--ink)" opacity="0.2" />
        {/* Cobalt dot — active indicator */}
        <circle cx="23" cy="23" r="4" fill="var(--cobalt)" />
        <path d="M21.5 23 L22.5 24 L24.5 22" stroke="white" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
      </svg>

      {showText && !collapsed && (
        <span
          className={`font-semibold tracking-tight leading-none ${textClass}`}
          style={{ color: "var(--text-1)" }}
        >
          Docopilot
        </span>
      )}
    </div>
  );
}
