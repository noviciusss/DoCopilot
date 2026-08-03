"use client";
import { useState, useCallback, useEffect } from "react";
import { apiGetDocuments, apiDeleteDocument, DocumentLibraryItem } from "../api";

export type { DocumentLibraryItem };

export function useDocuments(isLoggedIn: boolean) {
  const [documents, setDocuments]   = useState<DocumentLibraryItem[]>([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState<string | null>(null);
  const [showAllDocs, setShowAllDocs] = useState(false);

  const refresh = useCallback(async (myDocs = !showAllDocs ? true : false) => {
    if (!isLoggedIn) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiGetDocuments(!showAllDocs);
      setDocuments(data);
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [isLoggedIn, showAllDocs]);

  // Auto-fetch on login or scope toggle
  useEffect(() => {
    if (isLoggedIn) refresh();
  }, [isLoggedIn, showAllDocs]); // eslint-disable-line react-hooks/exhaustive-deps

  const deleteDoc = useCallback(async (docId: string) => {
    try {
      await apiDeleteDocument(docId);
      setDocuments(prev => prev.filter(d => d.id !== docId));
    } catch (e: unknown) {
      setError((e as Error).message);
    }
  }, []);

  const toggleScope = useCallback(() => {
    setShowAllDocs(prev => !prev);
  }, []);

  return { documents, loading, error, refresh, deleteDoc, showAllDocs, toggleScope };
}
