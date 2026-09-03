"use client";

import { useReducedMotion, motion, MotionConfig } from "motion/react";

// --- Document card definitions ----------------------------------------------
interface DocCard {
  id: string;
  label: string;
  ext: string;
  cx: number;
  cy: number;
  w: number;
  h: number;
  rotDeg: number;
  delay: number;
  hasAmberHighlight: boolean;
}

const CARDS: DocCard[] = [
  {
    id: "research",
    label: "RESEARCH",
    ext: ".PDF",
    cx: 110,
    cy: 135,
    w: 148,
    h: 104,
    rotDeg: -6,
    delay: 0,
    hasAmberHighlight: false,
  },
  {
    id: "notes",
    label: "NOTES",
    ext: ".TXT",
    cx: 370,
    cy: 148,
    w: 140,
    h: 96,
    rotDeg: 5,
    delay: 0.08,
    hasAmberHighlight: false,
  },
  {
    id: "report",
    label: "REPORT",
    ext: ".DOC",
    cx: 100,
    cy: 385,
    w: 148,
    h: 96,
    rotDeg: 3,
    delay: 0.16,
    hasAmberHighlight: true,
  },
  {
    id: "brief",
    label: "BRIEF",
    ext: ".PDF",
    cx: 380,
    cy: 375,
    w: 136,
    h: 92,
    rotDeg: -4,
    delay: 0.24,
    hasAmberHighlight: false,
  },
];

const INDEX_CX = 240;
const INDEX_CY = 258;
const INDEX_R = 28;

const LABELS = [
  { text: "INDEXED", x: INDEX_CX - 8, y: INDEX_CY - INDEX_R - 14, anchor: "middle" as const },
  { text: "RETRIEVED CONTEXT", x: INDEX_CX + INDEX_R + 12, y: INDEX_CY + 4, anchor: "start" as const },
  { text: "READY TO QUERY", x: INDEX_CX - INDEX_R - 12, y: INDEX_CY + 24, anchor: "end" as const },
];

function buildPath(card: DocCard): string {
  const dx = INDEX_CX - card.cx;
  const dy = INDEX_CY - card.cy;
  const len = Math.sqrt(dx * dx + dy * dy);
  const nx = dx / len;
  const ny = dy / len;
  const startX = card.cx + nx * 52;
  const startY = card.cy + ny * 42;
  const endX = INDEX_CX - nx * (INDEX_R + 3);
  const endY = INDEX_CY - ny * (INDEX_R + 3);
  const cpX = (startX + endX) / 2 - ny * 30;
  const cpY = (startY + endY) / 2 + nx * 30;
  return `M ${startX} ${startY} Q ${cpX} ${cpY} ${endX} ${endY}`;
}

function DocCardEl({ card, reduced }: { card: DocCard; reduced: boolean }) {
  const x = card.cx - card.w / 2;
  const y = card.cy - card.h / 2;

  return (
    <motion.g
      initial={reduced ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={
        reduced
          ? { duration: 0 }
          : { duration: 0.45, delay: card.delay, ease: [0.16, 1, 0.3, 1] }
      }
    >
      <g transform={`rotate(${card.rotDeg}, ${card.cx}, ${card.cy})`}>
        <rect
          x={x} y={y} width={card.w} height={card.h} rx={6}
          fill="var(--surface)" stroke="var(--border)" strokeWidth={1}
        />
        <path
          d={`M ${x + card.w - 18} ${y} L ${x + card.w} ${y + 18} L ${x + card.w - 18} ${y + 18} Z`}
          fill="var(--ink)" opacity={0.6}
        />
        <path
          d={`M ${x + card.w - 18} ${y} L ${x + card.w} ${y + 18}`}
          stroke="var(--border)" strokeWidth={0.75} fill="none"
        />
        <rect x={x + 14} y={y + 26} width={card.w * 0.55} height={1.5} rx={0.75} fill="var(--border-light)" />
        <rect x={x + 14} y={y + 33} width={card.w * 0.7} height={1.5} rx={0.75} fill="var(--border)" />
        <rect x={x + 14} y={y + 40} width={card.w * 0.5} height={1.5} rx={0.75} fill="var(--border)" />
        <rect x={x + 14} y={y + 47} width={card.w * 0.65} height={1.5} rx={0.75} fill="var(--border)" opacity={0.6} />

        {card.hasAmberHighlight && (
          <motion.rect
            x={x + 12} y={y + 31} width={card.w * 0.72} height={19} rx={2}
            fill="var(--amber)"
            animate={reduced ? { opacity: 0.12 } : { opacity: [0.08, 0.18, 0.08] }}
            transition={
              reduced
                ? { duration: 0 }
                : { duration: 3.5, repeat: Infinity, ease: "easeInOut", delay: 0.6 }
            }
          />
        )}

        <text
          x={x + 14} y={y + card.h - 14}
          fontFamily="var(--font-sans)" fontSize={8} fontWeight={600}
          letterSpacing="0.08em" fill="var(--text-3)"
        >
          {card.label}
          <tspan fill="var(--cobalt)" fontWeight={500}>{card.ext}</tspan>
        </text>
      </g>
    </motion.g>
  );
}

function ConnectorPathEl({ card, reduced }: { card: DocCard; reduced: boolean }) {
  const d = buildPath(card);
  return (
    <motion.path
      d={d}
      stroke="var(--cobalt)" strokeWidth={0.75} strokeOpacity={0.35} fill="none"
      initial={reduced ? false : { pathLength: 0, opacity: 0 }}
      animate={{ pathLength: 1, opacity: 1 }}
      transition={
        reduced
          ? { duration: 0 }
          : {
              pathLength: { duration: 0.7, delay: card.delay + 0.3, ease: [0.16, 1, 0.3, 1] },
              opacity: { duration: 0.2, delay: card.delay + 0.3 },
            }
      }
    />
  );
}

export default function DocumentIndexVisual() {
  const reduced = useReducedMotion() ?? false;

  return (
    <MotionConfig reducedMotion="user">
      <div
        aria-hidden="true"
        role="presentation"
        className="w-full h-full flex items-center justify-center select-none"
      >
        <svg
          viewBox="0 0 480 520"
          width="100%"
          height="100%"
          preserveAspectRatio="xMidYMid meet"
          style={{ maxWidth: 420, maxHeight: 460 }}
          xmlns="http://www.w3.org/2000/svg"
        >
          {CARDS.map((card) => (
            <ConnectorPathEl key={card.id} card={card} reduced={reduced} />
          ))}

          <motion.g
            initial={reduced ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={reduced ? { duration: 0 } : { duration: 0.4, delay: 0.5 }}
          >
            <circle
              cx={INDEX_CX} cy={INDEX_CY} r={INDEX_R + 10}
              fill="none" stroke="var(--cobalt)" strokeWidth={0.5}
              strokeOpacity={0.15} strokeDasharray="3 5"
            />
            <circle
              cx={INDEX_CX} cy={INDEX_CY} r={INDEX_R}
              fill="var(--surface)" stroke="var(--cobalt)"
              strokeWidth={1} strokeOpacity={0.6}
            />
            <circle
              cx={INDEX_CX} cy={INDEX_CY} r={INDEX_R - 7}
              fill="var(--cobalt)" fillOpacity={0.08}
            />
            <rect x={INDEX_CX - 8} y={INDEX_CY - 5} width={16} height={1.5} rx={0.75} fill="var(--cobalt)" opacity={0.7} />
            <rect x={INDEX_CX - 6} y={INDEX_CY - 1.5} width={12} height={1.5} rx={0.75} fill="var(--cobalt)" opacity={0.5} />
            <rect x={INDEX_CX - 5} y={INDEX_CY + 2} width={10} height={1.5} rx={0.75} fill="var(--cobalt)" opacity={0.35} />
          </motion.g>

          {CARDS.map((card) => (
            <DocCardEl key={card.id} card={card} reduced={reduced} />
          ))}

          <motion.g
            initial={reduced ? false : { opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={reduced ? { duration: 0 } : { duration: 0.4, delay: 0.7 }}
          >
            {LABELS.map((lbl) => (
              <text
                key={lbl.text}
                x={lbl.x} y={lbl.y}
                textAnchor={lbl.anchor}
                fontFamily="var(--font-sans)"
                fontSize={7} fontWeight={500}
                letterSpacing="0.07em"
                fill="var(--text-3)"
              >
                {lbl.text}
              </text>
            ))}
          </motion.g>
        </svg>
      </div>
    </MotionConfig>
  );
}
