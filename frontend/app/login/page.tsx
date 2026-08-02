"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../../lib/hooks/useAuth";

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
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex items-center justify-center px-4">
      <div className="w-full max-w-sm space-y-8">

        {/* Logo */}
        <div className="text-center space-y-2">
          <span className="inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-zinc-100 text-xl font-bold text-zinc-950 shadow-lg">
            D
          </span>
          <h1 className="text-2xl font-semibold tracking-tight">DoCopilot</h1>
          <p className="text-xs text-zinc-500">
            {mode === "login" ? "Sign in to your workspace" : "Create a new workspace"}
          </p>
        </div>

        {/* Form */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 backdrop-blur-sm p-8 space-y-5">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-zinc-400">Email</label>
              <input
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none focus:border-zinc-600 transition-colors"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-zinc-400">Password</label>
              <input
                type="password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-2.5 text-sm text-zinc-100 placeholder:text-zinc-600 outline-none focus:border-zinc-600 transition-colors"
              />
            </div>

            {error && (
              <div className="rounded-lg bg-red-950/30 border border-red-900/50 px-3 py-2.5 text-xs text-red-400">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl bg-zinc-100 hover:bg-white text-zinc-950 font-semibold py-2.5 text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <span className="inline-block h-4 w-4 rounded-full border-2 border-zinc-950 border-t-transparent animate-spin" />
                  {mode === "login" ? "Signing in…" : "Creating account…"}
                </>
              ) : mode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>

          <div className="border-t border-zinc-800 pt-4 text-center">
            <button
              type="button"
              onClick={() => { setMode(m => m === "login" ? "register" : "login"); setEmail(""); setPassword(""); }}
              className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              {mode === "login"
                ? "No account? Create one →"
                : "Already have an account? Sign in →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
