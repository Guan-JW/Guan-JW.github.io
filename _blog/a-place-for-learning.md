---
title: "A place for learning, experiments, and small discoveries"
date: 2026-09-20
excerpt: "A sample notebook entry to preview research notes, code, equations, and personal reflections."
tags: [Things I Learned, Personal]
toc: true
published: false
---

> This is a sample draft for checking the blog design. Replace it with your own writing when you are ready.

A useful note does not need to be a finished paper. It can begin with a question, a surprising result, or an idea worth returning to.

## Start with a question

What did I learn today that I would like to remember a month from now?

This space can hold reading notes, small experiments, and interests outside research. A short entry might explain one concept; a longer one might connect a few papers or walk through an experiment.

## Make an idea concrete

For example, a compression ratio can be written as

$$
r = \frac{\text{original size}}{\text{compressed size}}.
$$

A small code snippet makes the same idea easy to reproduce:

```python
def compression_ratio(original_bytes, compressed_bytes):
    if compressed_bytes <= 0:
        raise ValueError("compressed_bytes must be positive")
    return original_bytes / compressed_bytes

print(compression_ratio(1024, 256))  # 4.0
```

### Record the context

A result is more useful when it includes the assumptions and the next question.

| Observation | Follow-up question |
| --- | --- |
| The representation takes less space. | How much time does decoding add? |
| One input works well. | Does the result hold for other inputs? |

## Leave room for other interests

Not every entry has to be technical. A book, a walk, or an interesting conversation can be worth a few paragraphs too.

中文笔记也可以放在这里：记录一个新学到的概念、一次实验的发现，或者一个想继续探索的问题。

## Takeaways

- Keep each note focused on an idea.
- Include enough context to revisit it later.
- Finish with an open question when there is more to explore.
