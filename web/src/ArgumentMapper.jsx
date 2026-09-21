import { useCallback, useEffect, useMemo, useState } from "react";

import { health } from "./api";
import Graph from "./components/Graph";
import HighlightedText from "./components/HighlightedText";
import InspectorPanel from "./components/InspectorPanel";
import ModelSelector from "./components/ModelSelector";
import RunStats from "./components/RunStats";
import TextInputPanel from "./components/TextInputPanel";
import { SAMPLE_PASSAGE } from "./constants";
import { DEFAULT_SELECTION, parseSelection } from "./selection";
import { useExtraction } from "./hooks/useExtraction";

/**
 * Pair components across two graphs by span overlap.
 *
 * The comparison view needs to show where two models agree. Matching on span
 * overlap rather than on text equality mirrors how the offline metrics score
 * these graphs, so what the UI calls a match is what the F1 numbers call one.
 */
function matchAcross(left, right) {
  const leftMatched = new Set();
  const rightMatched = new Set();

  left.forEach((a) => {
    const partner = right.find((b) => {
      if (rightMatched.has(b.id)) return false;
      const overlap = Math.min(a.end, b.end) - Math.max(a.start, b.start);
      if (overlap <= 0) return false;
      const union = Math.max(a.end, b.end) - Math.min(a.start, b.start);
      return overlap / union >= 0.5;
    });
    if (partner) {
      leftMatched.add(a.id);
      rightMatched.add(partner.id);
    }
  });

  return { leftMatched, rightMatched };
}

export default function ArgumentMapper() {
  const [text, setText] = useState("");
  const [selection, setSelection] = useState(DEFAULT_SELECTION);
  const [compareSelection, setCompareSelection] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [showGold, setShowGold] = useState(false);
  const [gold, setGold] = useState([]);
  const [availability, setAvailability] = useState(null);

  const primary = useExtraction();
  const secondary = useExtraction();
  const isComparing = compareSelection !== null;

  useEffect(() => {
    let cancelled = false;
    health()
      .then((body) => {
        if (!cancelled) setAvailability(body.routes);
      })
      .catch(() => {
        if (!cancelled) setAvailability({ claude: false, local: false, cascade: false });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === "Escape") setSelectedId(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const run = useCallback(() => {
    if (!text.trim()) return;
    setSelectedId(null);
    primary.run(text, parseSelection(selection));
    if (compareSelection) {
      secondary.run(text, parseSelection(compareSelection));
    }
  }, [text, selection, compareSelection, primary, secondary]);

  const matches = useMemo(() => {
    if (!isComparing) return { leftMatched: new Set(), rightMatched: new Set() };
    return matchAcross(primary.graph.components, secondary.graph.components);
  }, [isComparing, primary.graph.components, secondary.graph.components]);

  const busy = primary.status === "loading" || secondary.status === "loading";

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <h1>Argument Mapper</h1>
          <p className="app__tagline">
            Extract argument components and their relations, and see what each model costs.
          </p>
        </div>
        <div className="app__toolbar">
          <label className="toggle">
            <input
              type="checkbox"
              checked={showGold}
              onChange={(event) => setShowGold(event.target.checked)}
              disabled={gold.length === 0}
            />
            Gold overlay
            {gold.length === 0 && <span className="toggle__hint">load a dataset document</span>}
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={isComparing}
              onChange={(event) =>
                setCompareSelection(event.target.checked ? DEFAULT_SELECTION : null)
              }
            />
            Compare two models
          </label>
        </div>
      </header>

      <section className="app__controls">
        <ModelSelector
          value={selection}
          onChange={setSelection}
          availability={availability}
          disabled={busy}
        />
        {isComparing && (
          <ModelSelector
            value={compareSelection}
            onChange={setCompareSelection}
            availability={availability}
            disabled={busy}
          />
        )}
      </section>

      <main className={`app__body ${isComparing ? "is-comparing" : ""}`}>
        <div className="panel panel--source">
          <TextInputPanel
            text={text}
            onTextChange={setText}
            onExtract={run}
            onSample={() => setText(SAMPLE_PASSAGE)}
            onLoadGold={setGold}
            processing={busy}
          />
          <HighlightedText
            text={text}
            components={primary.graph.components}
            gold={showGold ? gold : []}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </div>

        <div className="panel panel--graph">
          <GraphPane
            title={primary.meta?.model ?? "Model A"}
            extraction={primary}
            selectedId={selectedId}
            matchedIds={matches.leftMatched}
            onSelect={setSelectedId}
          />
          {isComparing && (
            <GraphPane
              title={secondary.meta?.model ?? "Model B"}
              extraction={secondary}
              selectedId={selectedId}
              matchedIds={matches.rightMatched}
              onSelect={setSelectedId}
            />
          )}
        </div>

        {!isComparing && (
          <InspectorPanel
            selectedId={selectedId}
            graph={primary.graph}
            gold={gold}
            onSelect={setSelectedId}
          />
        )}
      </main>
    </div>
  );
}

function GraphPane({ title, extraction, selectedId, matchedIds, onSelect }) {
  const { graph, meta, status, error } = extraction;

  return (
    <section className="graph-pane">
      <header className="graph-pane__header">
        <h2>{title}</h2>
        <RunStats meta={meta} />
      </header>

      {status === "error" && (
        <div className="notice notice--error" role="alert">
          <strong>{error?.message ?? "Extraction failed."}</strong>
          {error?.detail && <p>{error.detail}</p>}
        </div>
      )}

      {status === "loading" ? (
        <div className="graph-empty">
          <p>Extracting…</p>
        </div>
      ) : (
        <Graph
          components={graph.components}
          relations={graph.relations}
          selectedId={selectedId}
          matchedIds={matchedIds}
          onSelect={onSelect}
          fitViewKey={`${title}-${graph.components.length}`}
        />
      )}
    </section>
  );
}
