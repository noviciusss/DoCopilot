"use client";
import { useState, useEffect, useCallback } from "react";
import { apiLogin, apiRegister } from "../api";

export function useAuth() {
  // Initialize token directly from localStorage if in browser environment
  const [token, setToken] = useState<string | null>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("docopilot_token");
    }
    return null;
  });

  const [isHydrated, setIsHydrated] = useState(false);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState<string | null>(null);

  // Sync token on mount and mark as hydrated
  useEffect(() => {
    const storedToken = localStorage.getItem("docopilot_token");
    setToken(storedToken);
    setIsHydrated(true);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiLogin(email, password);
      localStorage.setItem("docopilot_token", data.access_token);
      setToken(data.access_token);
      return true;
    } catch (e: unknown) {
      setError((e as Error).message);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const register = useCallback(async (email: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiRegister(email, password);
      localStorage.setItem("docopilot_token", data.access_token);
      setToken(data.access_token);
      return true;
    } catch (e: unknown) {
      setError((e as Error).message);
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("docopilot_token");
    sessionStorage.removeItem("document_id");
    setToken(null);
  }, []);

  return {
    isLoggedIn: !!token,
    token,
    isHydrated,
    loading: loading || !isHydrated,
    error,
    login,
    register,
    logout
  };
}
