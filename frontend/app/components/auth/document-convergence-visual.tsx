"use client";

import { useReducedMotion, motion, MotionConfig } from "motion/react";

// ─── Fragment definitions (all coordinates in viewBox 0 0 1000 660) ──────────
interface Fragment {
  id: string;
  label: string;
  ext: string;
  cx: number;
  cy: number;
  w: number;
  h: number;
  rot: number;
  delay: number;
  tablet?: boolean;
}

const FRAGMENTS: Fragment[] = [
  { id: "research", label: "RESEARCH",  ext: ".PDF",  cx: 108, cy: 148, w: 150, h: 98,  rot: -7,  delay: 0,    tablet: true  },
  { id: "notes",    label: "NOTES",     ext: ".TXT",  cx: 878, cy: 132, w: 138, h: 90,  rot: 6,   delay: 0.07, tablet: true  },
  { id: "archive",  label: "ARCHIVE",   ext: ".MD",   cx: 100, cy: 510, w: 148, h: 96,  rot: 4,   delay: 0.11, tablet: false },
  { id: "report",   label: "REPORT",    ext: ".DOCX", cx: 882, cy: 502, w: 148, h: 96,  rot: -5,  delay: 0.15, tablet: false },
  { id: "summary",  label: "SUMMARY",   ext: ".TXT",  cx: 108, cy: 330, w: 136, h: 88,  rot: 2,   delay: 0.19, tablet: false },
  { id: "brief",    label: "BRIEF",     ext: ".PDF",  cx: 880, cy: 320, w: 136, h: 88,  rot: -3,  delay: 0.22, tablet: false },
];

const INDEX_CX = 500;
const INDEX_CY = 330;
const INDEX_R  = 32;

function buildPath(f: Fragment): string {
  const dx = INDEX_CX - f.cx;
  const dy = INDEX_CY - f.cy;
  const len = Math.sqrt(dx * dx + dy * dy);
  const nx = dx / len;
  const ny = dy / len;
  const startX = f.cx + nx * (f.w / 2 + 6);
  const startY = f.cy + ny * (f.h / 2 + 6);
  const endX = INDEX_CX - nx * (INDEX_R + 4);
  const endY = INDEX_CY - ny * (INDEX_R + 4);
  const cpX = (startX + endX) / 2 - ny * 40;
  const cpY = (startY + endY) / 2 + nx * 40;
  return `M ${startX} ${startY} Q ${cpX} ${cpY} ${endX} ${endY}`;
}

// ─── Single fragment (SVG group) ──────────────────────────────────────────────
function FragmentEl({ f, reduced }: { f: Fragment; reduced: boolean }) {
  const x = f.cx - f.w / 2;
  const y = f.cy - f.h / 2;
  return (
    <motion.g
      initial={reduced ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={
        reduced
          ? { duration: 0 }
          : { duration: 0.4, delay: f.delay, ease: [0.16, 1, 0.3, 1] }
      }
    >
      <g transform={`rotate(${f.rot}, ${f.cx}, ${f.cy})`}>
        {/* Card body */}
        <rect
          x={x} y={y} width={f.w} height={f.h} rx={5}
          fill="var(--surface)" stroke="var(--border)" strokeWidth={0.75}
        />
        {/* Fold corner */}
        <path
          d={`M ${x + f.w - 16} ${y} L ${x + f.w} ${y + 16} L ${x + f.w - 16} ${y + 16} Z`}
          fill="var(--ink)" opacity={0.5}
        />
        <path
          d={`M ${x + f.w - 16} ${y} L ${x + f.w} ${y + 16}`}
          stroke="var(--border)" strokeWidth={0.6} fill="none"
        />
        {/* Generic text lines */}
        <rect x={x+12} y={y+22} width={f.w*0.52} height={1.25} rx={0.6} fill="var(--border-light)" />
        <rect x={x+12} y={y+29} width={f.w*0.68} height={1.25} rx={0.6} fill="var(--border)" />
        <rect x={x+12} y={y+36} width={f.w*0.48} height={1.25} rx={0.6} fill="var(--border)" />
        <rect x={x+12} y={y+43} width={f.w*0.60} height={1.25} rx={0.6} fill="var(--border)" opacity={0.6} />
        {/* Label */}
        <text
          x={x + 12} y={y + f.h - 12}
          fontFamily="var(--font-sans)" fontSize={7.5} fontWeight={600}
          letterSpacing="0.08em" fill="var(--text-3)"
        >
          {f.label}
          <tspan fill="var(--cobalt)" fontWeight={500} opacity={0.8}>{f.ext}</tspan>
        </text>
      </g>
    </motion.g>
  );
}

// ─── Main export ──────────────────────────────────────────────────────────────
export default function DocumentConvergenceVisual() {
  const reduced = useReducedMotion() ?? false;

  return (
    <MotionConfig reducedMotion="user">
      {/*
       * Fixed full-screen layer, behind auth card (z-0).
       * pointer-events:none so the card and form remain fully interactive.
       * aria-hidden — purely decorative.
       */}
      <div
        aria-hidden="true"
        role="presentation"
        className="fixed inset-0 overflow-hidden select-none"
        style={{ zIndex: 0, pointerEvents: "none" }}
      >
        {/* ── Desktop + tablet: full convergence SVG ── */}
        <svg
          viewBox="0 0 1000 660"
          width="100%"
          height="100%"
          preserveAspectRatio="xMidYMid slice"
          xmlns="http://www.w3.org/2000/svg"
          className="hidden md:block"
        >
          {/* Radial fade mask: composition is fully transparent at center (behind card),
              fades to full opacity toward the outer edges where fragments live.
              This means no text or path lines bleed through behind the card. */}
          <defs>
            <radialGradient id="center-fade" cx="50%" cy="50%" r="50%">
              <stop offset="0%"   stopColor="black" stopOpacity="1" />
              <stop offset="28%"  stopColor="black" stopOpacity="1" />
              <stop offset="52%"  stopColor="black" stopOpacity="0.35" />
              <stop offset="70%"  stopColor="black" stopOpacity="0" />
            </radialGradient>
            <mask id="fade-mask">
              {/* White = show, black = hide. Invert: use rect minus center circle */}
              <rect width="1000" height="660" fill="white" />
              <rect width="1000" height="660" fill="url(#center-fade)" />
            </mask>
          </defs>

          {/* All composition elements masked — center fades to transparent */}
          <g mask="url(#fade-mask)">
          {/* Connector paths first (rendered behind fragments & ring) */}
          {FRAGMENTS.map((f) => (
            <motion.path
              key={`path-${f.id}`}
              d={buildPath(f)}
              stroke="var(--cobalt)"
              strokeWidth={0.75}
              strokeOpacity={0.38}
              fill="none"
              strokeLinecap="round"
              className={!f.tablet ? "hidden lg:block" : undefined}
              initial={reduced ? false : { pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={
                reduced
                  ? { duration: 0 }
                  : {
                      pathLength: { duration: 0.65, delay: f.delay + 0.28, ease: [0.16, 1, 0.3, 1] },
                      opacity: { duration: 0.15, delay: f.delay + 0.28 },
                    }
              }
            />
          ))}
          </g>

          {/* Index ring — behind the auth card in visual z-order */}
          <motion.g
            initial={reduced ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={reduced ? { duration: 0 } : { duration: 0.35, delay: 0.5 }}
          >
            {/* Outer dashed orbit */}
            <circle
              cx={INDEX_CX} cy={INDEX_CY} r={INDEX_R + 14}
              fill="none" stroke="var(--cobalt)" strokeWidth={0.4}
              strokeOpacity={0.12} strokeDasharray="3 6"
            />
            {/* Main ring */}
            <circle
              cx={INDEX_CX} cy={INDEX_CY} r={INDEX_R}
              fill="var(--surface)" stroke="var(--cobalt)"
              strokeWidth={0.85} strokeOpacity={0.45}
            />
            {/* Inner fill disc */}
            <circle
              cx={INDEX_CX} cy={INDEX_CY} r={INDEX_R - 8}
              fill="var(--cobalt)" fillOpacity={0.06}
            />
            {/* Index glyph — 3 stacked lines */}
            <rect x={INDEX_CX-9} y={INDEX_CY-6} width={18} height={1.5} rx={0.75} fill="var(--cobalt)" opacity={0.65} />
            <rect x={INDEX_CX-7} y={INDEX_CY-2} width={14} height={1.5} rx={0.75} fill="var(--cobalt)" opacity={0.45} />
            <rect x={INDEX_CX-5} y={INDEX_CY+2} width={10} height={1.5} rx={0.75} fill="var(--cobalt)" opacity={0.3}  />
          </motion.g>

          {/* Fragment cards — rendered outside mask so they appear at full opacity */}
          {FRAGMENTS.map((f) => (
            <g
              key={f.id}
              className={!f.tablet ? "hidden lg:block" : undefined}
            >
              <FragmentEl f={f} reduced={reduced} />
            </g>
          ))}

          {/* No center labels — they were positioned behind the auth card */}
        </svg>
      </div>
    </MotionConfig>
  );
}
