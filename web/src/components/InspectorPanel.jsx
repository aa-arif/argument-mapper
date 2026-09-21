import {
  componentColors,
  componentLabels,
  relationColors,
  relationIcons,
  relationLabels,
} from "../constants";

function RelationRow({ relation, otherId, direction, components, onSelect }) {
  const other = components.find((c) => c.id === otherId);
  const color = relationColors[relation.type] ?? "#888";
  return (
    <li className="relation-row">
      <span className="relation-row__icon" style={{ color }}>
        {relationIcons[relation.type] ?? "→"}
      </span>
      <span className="relation-row__label">
        {direction === "out" ? relationLabels[relation.type] : `${relationLabels[relation.type]} by`}
      </span>
      <button type="button" className="relation-row__target" onClick={() => onSelect?.(otherId)}>
        {other ? `${other.id} · ${componentLabels[other.type] ?? other.type}` : otherId}
      </button>
    </li>
  );
}

export default function InspectorPanel({ selectedId, graph, gold, onSelect }) {
  const { components, relations } = graph;

  if (!selectedId) {
    return (
      <aside className="inspector">
        <h2>Inspector</h2>
        <p className="inspector__hint">
          Select a component in the graph or the text to see how it connects.
        </p>
        <Summary components={components} relations={relations} gold={gold} />
      </aside>
    );
  }

  const component = components.find((c) => c.id === selectedId);
  if (!component) {
    return (
      <aside className="inspector">
        <h2>Inspector</h2>
        <p className="inspector__hint">That component is no longer in the graph.</p>
      </aside>
    );
  }

  const outgoing = relations.filter((r) => r.src === selectedId);
  const incoming = relations.filter((r) => r.tgt === selectedId);
  const color = componentColors[component.type] ?? "#888";

  return (
    <aside className="inspector">
      <h2>Inspector</h2>

      <div className="inspector__type" style={{ "--type-color": color }}>
        {componentLabels[component.type] ?? component.type ?? "Untyped"}
        <span className="inspector__id">{component.id}</span>
      </div>

      <blockquote className="inspector__text">{component.text}</blockquote>

      <p className="inspector__span">
        characters {component.start}&ndash;{component.end}
      </p>

      {outgoing.length > 0 && (
        <section>
          <h3>This component</h3>
          <ul className="relation-list">
            {outgoing.map((relation) => (
              <RelationRow
                key={relation.id ?? `${relation.src}-${relation.tgt}`}
                relation={relation}
                otherId={relation.tgt}
                direction="out"
                components={components}
                onSelect={onSelect}
              />
            ))}
          </ul>
        </section>
      )}

      {incoming.length > 0 && (
        <section>
          <h3>Is</h3>
          <ul className="relation-list">
            {incoming.map((relation) => (
              <RelationRow
                key={relation.id ?? `${relation.src}-${relation.tgt}`}
                relation={relation}
                otherId={relation.src}
                direction="in"
                components={components}
                onSelect={onSelect}
              />
            ))}
          </ul>
        </section>
      )}

      {outgoing.length === 0 && incoming.length === 0 && (
        <p className="inspector__hint">This component is not connected to any other.</p>
      )}
    </aside>
  );
}

function Summary({ components, relations, gold }) {
  if (components.length === 0) return null;

  const byType = components.reduce((counts, component) => {
    const key = component.type ?? "Untyped";
    counts[key] = (counts[key] ?? 0) + 1;
    return counts;
  }, {});

  return (
    <section className="inspector__summary">
      <h3>Graph</h3>
      <ul>
        {Object.entries(byType).map(([type, count]) => (
          <li key={type}>
            <span className="dot" style={{ background: componentColors[type] ?? "#888" }} />
            {componentLabels[type] ?? type}: {count}
          </li>
        ))}
        <li>Relations: {relations.length}</li>
        {gold?.length > 0 && <li>Gold components: {gold.length}</li>}
      </ul>
    </section>
  );
}
