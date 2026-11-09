#!/usr/bin/env python3
"""Build a LongBench-E prompt file for run_best_compare's ``longbench_e`` scenario.

Reads only the needed ``*_e.jsonl`` members of LongBench's data.zip through HTTP range requests
(``remotezip``), so the whole archive is never downloaded. Each task's official prompt template is
applied, and a prompt is kept when its length WITH the chat template is within the range. Tasks are
written one after another in the dataset's own order, so related prompts stay as close together as
the dataset puts them. Shuffling would spread them out and hide any match, so it is opt-in.

Each line is ``{"prompt", "reference", "answers", "task", "id", "tokens"}``. ``reference`` is the
first answer, because f1_main scores against one string; ``answers`` keeps them all.

Fixes against the first draft of this script:

* the count tokenizes the rendered template text. ``apply_chat_template(tokenize=True)`` returns a
  dict on newer transformers, and ``len()`` of that dict is 2, so every prompt looked 2 tokens long;
* ``reference`` is a string, which is what f1_main's F1 expects (``answers`` is a list).

    pip install remotezip transformers jinja2 sentencepiece tiktoken
    python -m f1_benchmark.make_longbench_e --tokenizer zai-org/GLM-5.1 --out longbench_e_2k4k.jsonl
"""
import argparse
import io
import json
import os

ZIP_URL = "https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip"

# LongBench-E tasks and their official prompt templates (LongBench's dataset2prompt.json).
TASKS = {
    "qasper_e": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en_e": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa_e": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa_e": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report_e": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "multi_news_e": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec_e": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa_e": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum_e": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lcc_e": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p_e": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}
# LongBench's own "length" is a word count; for the code tasks it is not comparable, so no pre-filter.
WORD_RANGE = (800, 4500)
CODE_TASKS = ("lcc_e", "repobench-p_e")


def make_counter(tok):
    """Token count of a prompt as the server sees it: the chat template applied, thinking off."""
    def count(prompt: str) -> int:
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except Exception:  # no chat template: count the raw prompt
            text = prompt
        return len(tok(text, add_special_tokens=False).input_ids)
    return count


def select(task_lines, count, n: int, min_tok: int, max_tok: int) -> list:
    """Records for ``{task: iterable of parsed jsonl rows}``, up to ceil(n / tasks) per task.

    Reading a task stops as soon as it has enough, so only the start of each file is fetched."""
    per_task = -(-n // max(1, len(task_lines)))
    out = []
    for task, rows in task_lines.items():
        kept = 0
        for ex in rows:
            if task not in CODE_TASKS and not WORD_RANGE[0] <= ex.get("length", 0) <= WORD_RANGE[1]:
                continue
            prompt = TASKS[task].format(context=ex["context"], input=ex.get("input", ""))
            k = count(prompt)
            if not min_tok <= k <= max_tok:
                continue
            answers = ex.get("answers") or [""]
            if isinstance(answers, str):
                answers = [answers]
            out.append({"prompt": prompt, "reference": str(answers[0]), "answers": answers,
                        "task": task, "id": ex.get("_id"), "tokens": k})
            kept += 1
            if kept >= per_task:
                break
        print(f"  {task:20s} kept {kept}")
    return out[:n]


def _rows(z, member):
    with z.open(member) as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            yield json.loads(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", default=os.environ.get("MODEL", "zai-org/GLM-5.1"),
                    help="GLM-5.1 tokenizer: a Hugging Face id or the model folder (only tokenizer files are read)")
    ap.add_argument("--n", type=int, default=300, help="prompts wanted, spread evenly over the tasks")
    ap.add_argument("--min-tok", type=int, default=2000)
    ap.add_argument("--max-tok", type=int, default=4000)
    ap.add_argument("--tasks", default=",".join(TASKS), help="comma-separated subset of the tasks")
    ap.add_argument("--zip", default=ZIP_URL, help="data.zip URL, or a local path to a downloaded copy")
    ap.add_argument("--shuffle", type=int, default=None, metavar="SEED",
                    help="shuffle with this seed (default: keep the dataset's order)")
    ap.add_argument("--out", default="longbench_e_2k4k.jsonl")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    count = make_counter(AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True))
    tasks = [t for t in args.tasks.split(",") if t]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        ap.error(f"unknown task(s) {unknown}; known: {', '.join(TASKS)}")

    if os.path.isfile(args.zip):
        import zipfile
        z = zipfile.ZipFile(args.zip)
    else:
        from remotezip import RemoteZip
        z = RemoteZip(args.zip)
    with z:
        names = {m.split("/")[-1].removesuffix(".jsonl"): m for m in z.namelist()}
        missing = [t for t in tasks if t not in names]
        for t in missing:
            print(f"  [skip] {t} not in the zip")
        results = select({t: _rows(z, names[t]) for t in tasks if t in names}, count,
                         args.n, args.min_tok, args.max_tok)

    if args.shuffle is not None:
        import random
        random.Random(args.shuffle).shuffle(results)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    toks = [r["tokens"] for r in results]
    if not toks:
        print("No prompt in range; nothing written.")
        return 1
    print(f"Wrote {len(results)} prompts to {args.out} | tokens min {min(toks)} "
          f"mean {sum(toks) // len(toks)} max {max(toks)}")
    print(f"Run it with: SCENARIO=longbench_e NUM_PROMPTS={len(results)} "
          f"F1_DATASET=<path on the box>/{os.path.basename(args.out)} RUNS=\"bff:48\" ./run_best_compare.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
