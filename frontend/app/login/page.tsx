"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../../lib/hooks/useAuth";
import Logo from "../components/Logo";
import { ArrowRight, Lock, Mail, Sparkles } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const { isLoggedIn, isHydrated, loading, error, login, register } = useAuth();
  const [mode, setMode]         = useState<"login" | "register">("login");
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");

  // Redirect to home if already logged in (only after hydration completes)
  useEffect(() => {
    if (isHydrated && isLoggedIn) router.push("/");
  }, [isLoggedIn, isHydrated, router]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const ok = mode === "login"
      ? await login(email, password)
      : await register(email, password);
    if (ok) router.push("/");
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col items-center justify-center px-4 relative overflow-hidden selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* Background ambient lighting effects */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-indigo-600/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 left-1/3 w-80 h-80 bg-cyan-600/10 rounded-full blur-3xl pointer-events-none" />

      <div className="w-full max-w-sm space-y-8 relative z-10">

        {/* Logo & Header */}
        <div className="flex flex-col items-center justify-center text-center space-y-3">
          <Logo size="lg" showSubtitle={false} />
          <p className="text-xs text-zinc-400 max-w-xs">
            {mode === "login"
              ? "Sign in to access your document intelligence workspace"
              : "Create an account to start indexing & querying your documents"}
          </p>
        </div>

        {/* Form Card */}
        <div className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 backdrop-blur-xl p-7 space-y-5 shadow-2xl shadow-indigo-950/40">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-zinc-300 flex items-center gap-1.5">
                <Mail className="w-3.5 h-3.5 text-zinc-400" />
                Email Address
              </label>
              <input
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 px-4 py-2.5 text-xs text-zinc-100 placeholder:text-zinc-600 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 transition-all shadow-inner"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-zinc-300 flex items-center gap-1.5">
                <Lock className="w-3.5 h-3.5 text-zinc-400" />
                Password
              </label>
              <input
                type="password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full rounded-xl border border-zinc-800 bg-zinc-950/80 px-4 py-2.5 text-xs text-zinc-100 placeholder:text-zinc-600 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 transition-all shadow-inner"
              />
            </div>

            {error && (
              <div className="rounded-xl bg-rose-950/40 border border-rose-900/60 p-3 text-xs text-rose-300 flex items-start gap-2">
                <span className="font-semibold text-rose-400">Error:</span>
                <span className="flex-1">{error}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl bg-gradient-to-r from-indigo-600 via-indigo-500 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-medium py-2.5 text-xs transition-all shadow-lg shadow-indigo-600/25 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 border border-indigo-400/20"
            >
              {loading ? (
                <>
                  <span className="inline-block h-3.5 w-3.5 rounded-full border-2 border-white border-t-transparent animate-spin" />
                  <span>{mode === "login" ? "Signing in…" : "Creating workspace…"}</span>
                </>
              ) : (
                <>
                  <span>{mode === "login" ? "Sign in to workspace" : "Create account"}</span>
                  <ArrowRight className="w-3.5 h-3.5 text-indigo-200" />
                </>
              )}
            </button>
          </form>

          <div className="border-t border-zinc-800/80 pt-4 text-center">
            <button
              type="button"
              onClick={() => { setMode(m => m === "login" ? "register" : "login"); setEmail(""); setPassword(""); }}
              className="text-xs text-zinc-400 hover:text-zinc-200 transition-colors flex items-center justify-center gap-1 mx-auto"
            >
              <span>
                {mode === "login"
                  ? "Don't have an account? Create one"
                  : "Already have an account? Sign in"}
              </span>
              <Sparkles className="w-3 h-3 text-indigo-400" />
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
