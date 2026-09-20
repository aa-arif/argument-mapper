"""Turn gold documents into supervised fine-tuning examples.

The local model is trained **zero-shot**: system prompt, the document, and the
gold graph as the answer. No worked examples.

That is the point of fine-tuning rather than a convenience. The Claude route
needs three exemplars in its prefix -- about 4,900 tokens carried on every
single request -- to be told what the task is. A fine-tuned model has the task
in its weights, so its prompt is the document alone. The saving shows up twice
over: in tokens per request, and in latency, both of which the cost comparison
in milestone 5 turns on.

The system prompt is deliberately the *same* text the Claude route uses. If the
two routes were given different task framings, any quality difference between
them would confound "fine-tuned 2B vs frontier API" with "prompt A vs prompt
B", and the question this repository asks would go unanswered.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from argmap.prompts import SYSTEM_PROMPT, build_user_prompt, document_to_graph
from argmap.schema import Document


def to_messages(doc: Document, *, include_answer: bool = True) -> list[dict[str, str]]:
    """Render one document as a chat-format training example."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(doc.text)},
    ]
    if include_answer:
        # `separators` without spaces: the target is machine-read JSON, and
        # every space is a token the model has to spend learning to emit.
        answer = json.dumps(
            document_to_graph(doc).model_dump(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages.append({"role": "assistant", "content": answer})
    return messages


def to_sft_records(docs: Iterable[Document]) -> list[dict[str, object]]:
    """Build chat-format records for TRL's `SFTTrainer`."""
    return [{"doc_id": d.doc_id, "messages": to_messages(d)} for d in docs]


def write_sft_jsonl(docs: Sequence[Document], path: Path) -> int:
    """Write training records. Gitignored -- these embed corpus text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = to_sft_records(docs)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    return len(records)


def answer_token_estimate(doc: Document) -> int:
    """Rough token count of the target answer, for choosing a sequence length.

    Deliberately crude: it informs a max-length choice, and a max length set
    from a crude estimate plus headroom is fine, whereas one set from a real
    tokenizer on a machine with no GPU is a dependency for no benefit.
    """
    answer = json.dumps(document_to_graph(doc).model_dump(), separators=(",", ":"))
    return len(answer) // 3
