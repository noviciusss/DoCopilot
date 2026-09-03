"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../../lib/hooks/useAuth";
import Logo from "../components/Logo";
import DocumentConvergenceVisual from "../components/auth/document-convergence-visual";
import { ArrowRight, Loader2 } from "lucide-react";

// Small static index mark shown only on mobile (replaces full visualization)
function MobileIndexMark() {
  return (
    <div aria-hidden="true" className="flex flex-col items-center gap-2 mb-5 md:hidden">
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="3"  y="3"  width="15" height="19" rx="2" fill="var(--surface)" stroke="var(--border)" strokeWidth="0.75" transform="rotate(-8,10,12)" />
        <rect x="30" y="4"  width="15" height="19" rx="2" fill="var(--surface)" stroke="var(--border)" strokeWidth="0.75" transform="rotate(7,37,13)" />
        <rect x="5"  y="28" width="15" height="17" rx="2" fill="var(--surface)" stroke="var(--border)" strokeWidth="0.75" transform="rotate(5,12,36)" />
        <rect x="28" y="26" width="15" height="17" rx="2" fill="var(--surface)" stroke="var(--border)" strokeWidth="0.75" transform="rotate(-5,35,34)" />
        <line x1="11" y1="20" x2="23" y2="24" stroke="var(--cobalt)" strokeWidth="0.6" strokeOpacity="0.45" />
        <line x1="37" y1="21" x2="26" y2="24" stroke="var(--cobalt)" strokeWidth="0.6" strokeOpacity="0.45" />
        <line x1="13" y1="29" x2="23" y2="25" stroke="var(--cobalt)" strokeWidth="0.6" strokeOpacity="0.45" />
        <line x1="35" y1="28" x2="26" y2="25" stroke="var(--cobalt)" strokeWidth="0.6" strokeOpacity="0.45" />
        <circle cx="24" cy="24" r="6" fill="var(--surface)" stroke="var(--cobalt)" strokeWidth="0.85" strokeOpacity="0.55" />
        <rect x="19" y="23" width="10" height="1.25" rx="0.6" fill="var(--cobalt)" opacity="0.65" />
        <rect x="20" y="25.5" width="7.5" height="1.25" rx="0.6" fill="var(--cobalt)" opacity="0.4" />
      </svg>
      <span style={{ fontSize: "8.5px", letterSpacing: "0.1em", color: "var(--text-3)", fontFamily: "var(--font-sans)" }}>
        DOCUMENT INDEX
      </span>
    </div>
  );
}

export default function LoginPage() {
  const router = useRouter();
  const { isLoggedIn, isHydrated, loading, error, login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  // Redirect if already authenticated
  useEffect(() => {
    if (isHydrated && isLoggedIn) router.push("/");
  }, [isLoggedIn, isHydrated, router]);

  // Auth handlers — all logic unchanged
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const ok =
      mode === "login"
        ? await login(email, password)
        : await register(email, password);
    if (ok) router.push("/");
  };

  return (
    <div
      className="relative min-h-screen flex flex-col items-center justify-center px-4 py-10"
      style={{ color: "var(--text-1)", zIndex: 1 }}
    >
      {/* Decorative convergence visual — fixed, behind this content */}
      <DocumentConvergenceVisual />

      {/* Subtle center-focus vignette: edges darker, center lighter.
          Adds atmospheric depth without any color/glow. pointer-events:none. */}
      <div
        aria-hidden="true"
        className="fixed inset-0 pointer-events-none"
        style={{
          zIndex: 1,
          background: "radial-gradient(ellipse 60% 55% at 50% 50%, transparent 0%, rgba(18,19,21,0.55) 100%)",
        }}
      />

      {/* Mobile-only index mark */}
      <MobileIndexMark />

      {/*
       * Auth card — the primary visual anchor.
       * position:relative + z-index so it sits above the fixed SVG layer.
       * email-input is the first meaningful keyboard focus target.
       */}
      <div
        className="relative w-full max-w-sm space-y-6"
        style={{ zIndex: 2 }}
      >
        {/* Brand */}
        <div className="flex flex-col items-center gap-3">
          <Logo size="md" />
          <p className="text-xs text-center max-w-xs" style={{ color: "var(--text-2)" }}>
            {mode === "login"
              ? "Sign in to your document research workspace"
              : "Create an account to start indexing documents"}
          </p>
        </div>

        {/* Auth form card */}
        <div
          className="rounded-xl p-6 space-y-4"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            boxShadow: "0 8px 48px rgba(0,0,0,0.6), 0 1px 0 rgba(255,255,255,0.04) inset",
          }}
        >
          <h1 className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>
            {mode === "login" ? "Sign in" : "Create account"}
          </h1>

          <form onSubmit={handleSubmit} className="space-y-3">
            {/* Email — first keyboard focus target */}
            <div className="space-y-1">
              <label
                htmlFor="email-input"
                className="text-xs font-medium"
                style={{ color: "var(--text-2)" }}
              >
                Email address
              </label>
              <input
                id="email-input"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full rounded-md px-3 py-2 text-xs transition-colors"
                style={{
                  background: "var(--surface-2)",
                  border: "1px solid var(--border)",
                  color: "var(--text-1)",
                  outline: "none",
                  fontFamily: "var(--font-sans)",
                }}
                onFocus={(e) => (e.currentTarget.style.borderColor = "var(--cobalt)")}
                onBlur={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
              />
            </div>

            {/* Password */}
            <div className="space-y-1">
              <label
                htmlFor="password-input"
                className="text-xs font-medium"
                style={{ color: "var(--text-2)" }}
              >
                Password
              </label>
              <input
                id="password-input"
                type="password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full rounded-md px-3 py-2 text-xs transition-colors"
                style={{
                  background: "var(--surface-2)",
                  border: "1px solid var(--border)",
                  color: "var(--text-1)",
                  outline: "none",
                  fontFamily: "var(--font-sans)",
                }}
                onFocus={(e) => (e.currentTarget.style.borderColor = "var(--cobalt)")}
                onBlur={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
              />
            </div>

            {/* Error */}
            {error && (
              <div
                className="text-xs rounded-md px-3 py-2 leading-snug"
                style={{
                  background: "var(--red-dim)",
                  border: "1px solid var(--red)",
                  color: "var(--red)",
                }}
                role="alert"
              >
                {error}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-2 text-xs font-medium rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                background: "var(--cobalt)",
                color: "#fff",
              }}
              onMouseEnter={(e) =>
                !e.currentTarget.disabled &&
                (e.currentTarget.style.background = "var(--cobalt-hover)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = "var(--cobalt)")
              }
            >
              {loading ? (
                <>
                  <Loader2 size={13} className="animate-spin" aria-hidden="true" />
                  <span>{mode === "login" ? "Signing in…" : "Creating account…"}</span>
                </>
              ) : (
                <>
                  <span>{mode === "login" ? "Sign in" : "Create account"}</span>
                  <ArrowRight size={13} aria-hidden="true" />
                </>
              )}
            </button>
          </form>

          {/* Toggle login ↔ register */}
          <div className="pt-3" style={{ borderTop: "1px solid var(--border)" }}>
            <button
              type="button"
              onClick={() => {
                setMode((m) => (m === "login" ? "register" : "login"));
                setEmail("");
                setPassword("");
              }}
              className="text-xs transition-colors"
              style={{ color: "var(--text-3)" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-1)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-3)")}
            >
              {mode === "login"
                ? "Don't have an account? Create one"
                : "Already have an account? Sign in"}
            </button>
          </div>
        </div>

        <p
          className="text-center"
          style={{ fontSize: "10px", color: "var(--text-3)" }}
        >
          Docopilot · Document intelligence workspace
        </p>
      </div>
    </div>
  );
}
