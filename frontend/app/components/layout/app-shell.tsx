"use client";

import { ReactNode } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { motion, MotionConfig } from "motion/react";
import { X } from "lucide-react";

interface Props {
  sidebar: ReactNode;
  workspace: ReactNode;
  chatPanel: ReactNode;
  sidebarCollapsed: boolean;
  // Mobile chat sheet state
  mobileChatOpen: boolean;
  onMobileChatOpenChange: (v: boolean) => void;
}

export default function AppShell({
  sidebar,
  workspace,
  chatPanel,
  sidebarCollapsed,
  mobileChatOpen,
  onMobileChatOpenChange,
}: Props) {
  return (
    <MotionConfig reducedMotion="user">
      <div
        className="flex h-screen w-screen overflow-hidden"
        style={{ background: "var(--ink)" }}
      >
        {/* Left sidebar (desktop: fixed position, mobile: dialog-drawer) */}
        {sidebar}

        {/* Main area — animated offset matching the fixed sidebar width */}
        <motion.div
          className="flex flex-1 min-w-0 overflow-hidden"
          initial={false}
          animate={{ marginLeft: sidebarCollapsed ? 48 : 260 }}
          transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
          // On mobile the sidebar is a dialog overlay, so no margin offset
          style={{ marginLeft: 0 }}
        >
          <style>{`
            @media (max-width: 1023px) {
              /* Reset margin on mobile — sidebar is a modal drawer */
              .sidebar-offset { margin-left: 0 !important; }
            }
          `}</style>

          {/* Inner: workspace + chat panel side-by-side on desktop */}
          <div className="flex flex-1 min-w-0 overflow-hidden w-full">
            {/* Center workspace */}
            <div className="workspace min-w-0">
              {workspace}
            </div>

            {/* Right chat panel — desktop only */}
            <div className="hidden lg:flex" style={{ flexShrink: 0 }}>
              {chatPanel}
            </div>
          </div>
        </motion.div>

        {/* Mobile chat sheet — Radix Dialog, slides from bottom */}
        <Dialog.Root open={mobileChatOpen} onOpenChange={onMobileChatOpenChange}>
          <Dialog.Portal>
            <Dialog.Overlay
              className="fixed inset-0 z-50 bg-black/50 lg:hidden"
              style={{ backdropFilter: "blur(2px)" }}
            />
            <Dialog.Content
              className="fixed z-50 inset-x-0 bottom-0 lg:hidden rounded-t-xl overflow-hidden flex flex-col shadow-2xl"
              style={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                maxHeight: "85vh",
              }}
              aria-describedby="mobile-chat-desc"
            >
              <Dialog.Title className="sr-only">Document assistant</Dialog.Title>
              <Dialog.Description id="mobile-chat-desc" className="sr-only">
                Chat with your document using AI-powered retrieval
              </Dialog.Description>

              {/* Drag handle + close */}
              <div
                className="flex items-center justify-between px-4 py-3 flex-shrink-0"
                style={{ borderBottom: "1px solid var(--border)" }}
              >
                <div
                  className="w-8 h-1 rounded-full"
                  style={{ background: "var(--border)" }}
                  aria-hidden="true"
                />
                <Dialog.Close asChild>
                  <button
                    className="p-1.5 rounded-md"
                    style={{ color: "var(--text-3)" }}
                    aria-label="Close chat"
                  >
                    <X size={15} aria-hidden="true" />
                  </button>
                </Dialog.Close>
              </div>

              {/* Real chat panel inside the sheet */}
              <div className="flex-1 flex flex-col overflow-hidden">
                {chatPanel}
              </div>
            </Dialog.Content>
          </Dialog.Portal>
        </Dialog.Root>
      </div>
    </MotionConfig>
  );
}
