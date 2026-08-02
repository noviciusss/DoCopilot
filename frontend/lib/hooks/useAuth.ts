"use client";
import { useState, useEffect, useCallback } from "react";
import { apiLogin, apiRegister } from "../api";

export function useAuth() {
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Rehydrate JWT from localStorage on mount
  useEffect(() => {
    setToken(localStorage.getItem("docopilot_token"));
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

  return { isLoggedIn: !!token, token, loading, error, login, register, logout };
}
