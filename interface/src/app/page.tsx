"use client";

import Image from "next/image";
import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import {
  advanceSpring,
  gazeTarget,
  springIsSettled,
  type Point,
  type SpringAxis,
} from "../lib/companion-gaze";
import { DEFAULT_PET_ID, isPetId, nextPetId, petById, type PetId } from "../lib/pets";
import MediaCapture, { type CompanionState } from "./media-capture";
import styles from "./page.module.css";

type Message = { id: string; role: "user" | "model" | "teacher" | "notice"; text: string };
type ChatResponse = { session_id?: string; message_id?: string | number; answer?: string; error?: string };
const SESSION_KEY = "mlm.apprenticeship.session";
const PET_KEY = "tinydog.selected-pet";

function localId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function firstString(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) if (typeof record[key] === "string") return record[key] as string;
  return undefined;
}

function normalizeHistoryMessage(value: unknown, index: number): Message | null {
  const record = asRecord(value);
  if (!record) return null;
  const rawRole = firstString(record, ["role", "sender"]);
  const role = rawRole === "user" || rawRole === "human" ? "user" : rawRole === "teacher" ? "teacher" : "model";
  const text = firstString(record, ["text", "content", "message", "answer"]);
  if (!text) return null;
  const rawId = record.message_id ?? record.id;
  return {
    id: typeof rawId === "string" || typeof rawId === "number" ? `history-${rawId}` : `history-${index}`,
    role,
    text,
  };
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [composerOpen, setComposerOpen] = useState(false);
  const [companionState, setCompanionState] = useState<CompanionState>("idle");
  const [petId, setPetId] = useState<PetId>(DEFAULT_PET_ID);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const companionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const savedPet = window.localStorage.getItem(PET_KEY);
    if (isPetId(savedPet)) setPetId(savedPet);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const savedSession = window.localStorage.getItem(SESSION_KEY);
    if (!savedSession) {
      setHistoryLoading(false);
      return () => controller.abort();
    }
    setSessionId(savedSession);
    void (async () => {
      try {
        const response = await fetch(`/api/history?session_id=${encodeURIComponent(savedSession)}`, { cache: "no-store", signal: controller.signal });
        const data = (await response.json()) as { messages?: unknown[] };
        if (!response.ok) throw new Error("History unavailable");
        setMessages((data.messages ?? []).map(normalizeHistoryMessage).filter((message): message is Message => message !== null));
      } catch {
        if (!controller.signal.aborted) setMessages([]);
      } finally {
        if (!controller.signal.aborted) setHistoryLoading(false);
      }
    })();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    function handleShortcut(event: globalThis.KeyboardEvent) {
      const target = event.target;
      const editingText = target instanceof HTMLElement
        && (target.isContentEditable || target.matches("input, textarea, select"));
      const key = event.key.toLowerCase();
      const primaryShortcut = event.metaKey !== event.ctrlKey
        && (event.metaKey || event.ctrlKey)
        && !event.shiftKey
        && !event.altKey;

      if (primaryShortcut && key === "b") {
        event.preventDefault();
        setComposerOpen(true);
        window.setTimeout(() => inputRef.current?.focus(), 0);
      } else if (primaryShortcut && key === "v" && !editingText && !event.repeat) {
        event.preventDefault();
        setPetId((current) => {
          const next = nextPetId(current);
          window.localStorage.setItem(PET_KEY, next);
          return next;
        });
      } else if (event.key === "Escape") {
        setComposerOpen(false);
      }
    }
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, []);

  useEffect(() => {
    const companion = companionRef.current;
    if (!companion) return;

    const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let target: Point = { x: 0, y: 0 };
    let eyeX: SpringAxis = { position: 0, velocity: 0 };
    let eyeY: SpringAxis = { position: 0, velocity: 0 };
    let bodyX: SpringAxis = { position: 0, velocity: 0 };
    let bodyY: SpringAxis = { position: 0, velocity: 0 };
    let animationFrame: number | null = null;
    let previousTime = performance.now();

    function renderGaze() {
      companion?.style.setProperty("--eye-x", `${(eyeX.position * 4.8).toFixed(3)}px`);
      companion?.style.setProperty("--eye-y", `${(eyeY.position * 3.4).toFixed(3)}px`);
      companion?.style.setProperty("--body-x", `${(bodyX.position * 7).toFixed(3)}px`);
      companion?.style.setProperty("--body-y", `${(bodyY.position * 4.5).toFixed(3)}px`);
      companion?.style.setProperty("--body-turn", `${(bodyX.position * 1.8).toFixed(3)}deg`);
    }

    function tick(time: number) {
      const delta = time - previousTime;
      previousTime = time;
      eyeX = advanceSpring(eyeX, target.x, delta, 150, 25);
      eyeY = advanceSpring(eyeY, target.y, delta, 150, 25);
      bodyX = advanceSpring(bodyX, target.x, delta, 52, 15);
      bodyY = advanceSpring(bodyY, target.y, delta, 52, 15);
      renderGaze();

      const settled = springIsSettled(eyeX, target.x)
        && springIsSettled(eyeY, target.y)
        && springIsSettled(bodyX, target.x)
        && springIsSettled(bodyY, target.y);
      animationFrame = settled ? null : window.requestAnimationFrame(tick);
    }

    function animateToward(nextTarget: Point) {
      target = nextTarget;
      if (animationFrame === null) {
        previousTime = performance.now();
        animationFrame = window.requestAnimationFrame(tick);
      }
    }

    function handlePointerMove(event: PointerEvent) {
      if (event.pointerType !== "mouse" || !finePointer.matches || reducedMotion.matches) return;
      const rect = companion?.getBoundingClientRect();
      if (!rect) return;
      animateToward(gazeTarget(
        { x: event.clientX, y: event.clientY },
        { left: rect.left, top: rect.top, width: rect.width, height: rect.height },
        { width: window.innerWidth, height: window.innerHeight },
      ));
    }

    function returnToNeutral() {
      animateToward({ x: 0, y: 0 });
    }

    function handlePointerOut(event: PointerEvent) {
      if (event.relatedTarget === null) returnToNeutral();
    }

    function handleMotionPreference() {
      if (!finePointer.matches || reducedMotion.matches) returnToNeutral();
    }

    window.addEventListener("pointermove", handlePointerMove, { passive: true });
    window.addEventListener("pointerout", handlePointerOut, { passive: true });
    window.addEventListener("blur", returnToNeutral);
    finePointer.addEventListener("change", handleMotionPreference);
    reducedMotion.addEventListener("change", handleMotionPreference);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerout", handlePointerOut);
      window.removeEventListener("blur", returnToNeutral);
      finePointer.removeEventListener("change", handleMotionPreference);
      reducedMotion.removeEventListener("change", handleMotionPreference);
      if (animationFrame !== null) window.cancelAnimationFrame(animationFrame);
    };
  }, []);

  async function ask(question: string) {
    const clean = question.trim();
    if (!clean || loading || historyLoading) return;
    setMessages((current) => [...current, { id: localId("user"), role: "user", text: clean }]);
    setInput("");
    setComposerOpen(false);
    setLoading(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: clean, ...(sessionId ? { session_id: sessionId } : {}) }),
      });
      const data = (await response.json()) as ChatResponse;
      if (!response.ok) throw new Error(data.error || "tinydog is unavailable");
      if (data.session_id) {
        setSessionId(data.session_id);
        window.localStorage.setItem(SESSION_KEY, data.session_id);
      }
      setMessages((current) => [...current, {
        id: data.message_id === undefined ? localId("model") : `model-${data.message_id}`,
        role: "model",
        text: data.answer || "…",
      }]);
    } catch (reason) {
      setMessages((current) => [...current, {
        id: localId("notice"),
        role: "notice",
        text: reason instanceof Error ? reason.message : "tinydog is unavailable",
      }]);
    } finally {
      setLoading(false);
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void ask(input);
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  function openComposer() {
    setComposerOpen(true);
    window.setTimeout(() => inputRef.current?.focus(), 0);
  }

  const visibleMessages = messages.slice(-4);
  const dogState = loading && companionState === "idle" ? "thinking" : companionState;
  const selectedPet = petById(petId);
  const petEyeStyle = {
    "--eye-left-x": `${selectedPet.eyes.left[0]}%`,
    "--eye-left-y": `${selectedPet.eyes.left[1]}%`,
    "--eye-right-x": `${selectedPet.eyes.right[0]}%`,
    "--eye-right-y": `${selectedPet.eyes.right[1]}%`,
  } as React.CSSProperties;

  return (
    <main className={styles.world}>
      <header className={styles.minimalHeader}>
        <span className={styles.wordmark}>tinydog.ai</span>
      </header>

      <section className={styles.immersiveMessages} aria-live="polite">
        {visibleMessages.map((message, index) => (
          <p key={message.id} className={`${styles.immersiveMessage} ${styles[message.role]}`}
            style={{ "--message-age": visibleMessages.length - index - 1 } as React.CSSProperties}>
            {message.text}
          </p>
        ))}
      </section>

      <section ref={companionRef} className={styles.companion} data-state={dogState}
        aria-label={`${selectedPet.label} companion is ${dogState}`} style={petEyeStyle}>
        <div className={styles.aura} aria-hidden="true" />
        <button className={styles.trainingBubble} type="button" onClick={openComposer}>Talk to train me.</button>
        <div className={styles.dogLook}>
          <div className={styles.dogStage}>
            <Image key={selectedPet.id} className={styles.dog} src={selectedPet.src} alt={selectedPet.alt}
              width={1254} height={1254} priority />
            <span className={`${styles.gazeEye} ${styles.gazeEyeLeft}`} aria-hidden="true" />
            <span className={`${styles.gazeEye} ${styles.gazeEyeRight}`} aria-hidden="true" />
          </div>
        </div>
        <div className={styles.voiceControl}>
          <MediaCapture sessionId={sessionId} minimal onCompanionState={setCompanionState} onOpenTextInput={openComposer} />
        </div>
      </section>

      {loading && <span className={styles.thinkingDots} aria-label="Thinking">•••</span>}

      {composerOpen && (
        <div className={styles.composerBackdrop} onMouseDown={() => setComposerOpen(false)}>
          <form className={styles.commandComposer} onSubmit={submit} onMouseDown={(event) => event.stopPropagation()}>
            <label className={styles.srOnly} htmlFor="message">Message tinydog</label>
            <textarea ref={inputRef} id="message" rows={1} value={input}
              onChange={(event) => setInput(event.target.value)} onKeyDown={handleComposerKeyDown}
              placeholder="say something…" autoComplete="off" disabled={historyLoading} />
          </form>
        </div>
      )}
    </main>
  );
}
