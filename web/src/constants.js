// Visual vocabulary for the unified argument schema.
//
// These are the Argument Annotated Essays categories, which is what the
// backend, the metrics and the gold annotations all speak. v1 used a richer
// philosophy-flavoured set (objection, hidden_assumption, weakness); those had
// no gold labels behind them, so nothing could ever be scored against them.

export const componentColors = {
  MajorClaim: "#7BA5C4",
  Claim: "#D4A574",
  Premise: "#8BC47B",
};

export const componentLabels = {
  MajorClaim: "Major claim",
  Claim: "Claim",
  Premise: "Premise",
};

// Short forms for the node badges, where horizontal space is tight.
export const componentAbbrev = {
  MajorClaim: "MC",
  Claim: "C",
  Premise: "P",
};

export const componentOrder = ["MajorClaim", "Claim", "Premise"];

export const relationColors = {
  supports: "#8BC47B",
  attacks: "#C47B7B",
};

export const relationLabels = {
  supports: "supports",
  attacks: "attacks",
};

export const relationIcons = {
  supports: "↑",
  attacks: "⊗",
};

// Gold annotations render in a single neutral colour rather than by type: the
// overlay answers "where are the real components", and colouring it like the
// prediction makes the two hard to tell apart at a glance.
export const GOLD_COLOR = "#9B8AC4";

export const ROUTES = [
  {
    id: "claude",
    model: "claude-sonnet-5",
    label: "Claude Sonnet 5",
    detail: "Frontier API, structured outputs",
  },
  {
    id: "claude",
    model: "claude-haiku-4-5",
    label: "Claude Haiku 4.5",
    detail: "Cheaper frontier API",
  },
  {
    id: "local",
    model: "qwen3.5-2b-lora",
    label: "Qwen3.5-2B + LoRA",
    detail: "Fine-tuned local model, JSON-schema constrained",
  },
  {
    id: "cascade",
    model: "cascade",
    label: "Cascade",
    detail: "Local model, escalating low-confidence documents",
  },
];

export const NODE_WIDTH = 190;
export const NODE_HEIGHT = 120;

export const SAMPLE_PASSAGE = `If morality is to have any genuine authority over us, its commands must hold unconditionally, not as counsels of prudence. A hypothetical imperative says: "If you want X, do Y." But such rules bind us only insofar as we happen to desire X; change the desire and the rule evaporates. A moral law that could be dismissed by simply wanting something else would be no law at all, only advice.

Therefore the moral law must take the form of a categorical imperative, binding on every rational agent regardless of inclination. And since a law that applied only to me would not be a law but a preference, the principle of any action I take must be one I can will to be universal. This is why lying for personal gain is wrong: if everyone lied when convenient, the very practice of promising would collapse, and with it the possibility of the lie itself.

One might object that consequences, not principles, are what matter morally — that a lie which saves a life is obviously permissible. But this conflates two questions: what produces good outcomes, and what reason demands. Reason, unlike outcome, is not a matter of luck. To ground morality in consequences is to make it hostage to circumstance, which is precisely what a moral law cannot be.`;
