"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../../lib/hooks/useAuth";
import Logo from "../components/Logo";
import { ArrowRight, Loader2 } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const { isLoggedIn, isHydrated, loading, error, login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  useEffect(() => {
    if (isHydrated && isLoggedIn) router.push("/");
  }, [isLoggedIn, isHydrated, router]);

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
      className="min-h-screen flex flex-col items-center justify-center px-4"
      style={{ background: "var(--ink)", color: "var(--text-1)" }}
    >
      <div className="w-full max-w-sm space-y-7">
        {/* Brand */}
        <div className="flex flex-col items-center gap-3">
          <Logo size="md" />
          <p className="text-xs text-center max-w-xs" style={{ color: "var(--text-2)" }}>
            {mode === "login"
              ? "Sign in to your document research workspace"
              : "Create an account to start indexing documents"}
          </p>
        </div>

        {/* Form card */}
        <div
          className="rounded-xl p-6 space-y-4"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
          }}
        >
          <h1 className="text-sm font-semibold" style={{ color: "var(--text-1)" }}>
            {mode === "login" ? "Sign in" : "Create account"}
          </h1>

          <form onSubmit={handleSubmit} className="space-y-3">
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
                  <span>
                    {mode === "login" ? "Signing in…" : "Creating account…"}
                  </span>
                </>
              ) : (
                <>
                  <span>
                    {mode === "login" ? "Sign in" : "Create account"}
                  </span>
                  <ArrowRight size={13} aria-hidden="true" />
                </>
              )}
            </button>
          </form>

          <div
            className="pt-3"
            style={{ borderTop: "1px solid var(--border)" }}
          >
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
