# Third-party notices

This repository contains no ported third-party code. It redistributes third-party data:

## GSM8K

- **What:** `results/2026-11-12/gate/items/items.jsonl`, the registered item set of the published gate: the 1,319
  GSM8K test questions with their final answers (`{item, question, answer}`, `item` the test index). The gate's
  NeMo Evaluator configurations under `results/2026-11-12/gate/nel/` name the benchmark only.
- **Source:** GSM8K, <https://github.com/openai/grade-school-math> at commit `3101c7d5072418e28b9008a6636bde82a006892c`
  (Hugging Face `openai/gsm8k`, config `main`, revision `740312add88f781978c0658806c59bc2815b9866`).
- **License:** MIT. The license text, fetched from the repository at that commit:

```
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

