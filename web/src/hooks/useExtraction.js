import { useCallback, useRef, useState } from "react";

import { EMPTY_GRAPH, extract } from "../api";

/**
 * Run an extraction and track its lifecycle.
 *
 * Pulled out of the view so the comparison panel can hold two of these
 * independently and run both models against the same text at once.
 */
export function useExtraction() {
  const [graph, setGraph] = useState(EMPTY_GRAPH);
  const [meta, setMeta] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | ready | error
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  const run = useCallback(async (text, options = {}) => {
    // Supersede any in-flight request: a user changing model mid-run should
    // see the new answer, not whichever response happens to land last.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStatus("loading");
    setError(null);

    try {
      const result = await extract(text, { ...options, signal: controller.signal });
      if (controller.signal.aborted) return null;
      setGraph({ components: result.components, relations: result.relations });
      setMeta({
        route: result.route,
        model: result.model,
        cached: result.cached,
        latencyMs: result.latency_ms,
        usage: result.usage,
        escalated: result.escalated,
        confidence: result.confidence,
      });
      setStatus("ready");
      return result;
    } catch (caught) {
      if (controller.signal.aborted) return null;
      setError(caught);
      setStatus("error");
      return null;
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setStatus("idle");
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setGraph(EMPTY_GRAPH);
    setMeta(null);
    setError(null);
    setStatus("idle");
  }, []);

  return { graph, meta, status, error, run, cancel, reset };
}
