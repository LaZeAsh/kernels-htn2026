"""Pure Python check of prompt proposals plus exact window verification.

Run with: python3 agent/window4_prompt_lookup_oracle.py
"""

import random


def next_token(prefix):
    state = 1469598103934665603
    for token in prefix:
        state = ((state ^ (token + 1)) * 1099511628211) & ((1 << 64) - 1)
    return state % 19


def lookups(prompt):
    tables = {4: {}, 3: {}}
    for width, table in tables.items():
        for start in range(len(prompt) - width - 2):
            table[tuple(prompt[start:start + width])] = tuple(
                prompt[start + width:start + width + 3]
            )
    return tables


def record(history, tables):
    for width, table in tables.items():
        start = len(history) - width - 3
        if start >= 0:
            table[tuple(history[start:start + width])] = tuple(
                history[start + width:start + width + 3]
            )


def sequential(prompt, count):
    history = prompt[:]
    output = []
    for _ in range(count):
        token = next_token(history)
        output.append(token)
        history.append(token)
    return output


def windowed(prompt, count, stats=None):
    if count <= 0:
        return []
    first = next_token(prompt)
    output = [first]
    if count == 1:
        return output
    tables = lookups(prompt)
    history = prompt + [first]
    record(history, tables)
    current = first
    guesses = [first] * 3
    remaining = count - 1
    while remaining:
        for width in (4, 3):
            proposal = tables[width].get(tuple(history[-width:]))
            if proposal is not None:
                if stats is not None:
                    stats[width] += 1
                guesses = list(proposal)
                break
        proposed = [current, *guesses]
        prefix_before_current = history[:-1]
        results = [
            next_token(prefix_before_current + proposed[:index + 1])
            for index in range(4)
        ]
        accepted = 1
        while accepted < 4 and guesses[accepted - 1] == results[accepted - 1]:
            accepted += 1
        emitted = min(accepted, remaining)
        for token in results[:emitted]:
            history.append(token)
            record(history, tables)
            output.append(token)
        remaining -= emitted
        if remaining:
            current = results[accepted - 1]
            guesses = results[accepted:] + [results[-1]] * (3 - len(results[accepted:]))
    return output


if __name__ == "__main__":
    rng = random.Random(7)
    for seed in range(100):
        prompt = [rng.randrange(19) for _ in range(9 + seed % 27)]
        for count in (1, 2, 3, 4, 5, 17, 32):
            actual = windowed(prompt, count)
            expected = sequential(prompt, count)
            assert len(actual) == count and actual == expected, (seed, count)
    next_token = lambda prefix: (len(prefix) % 4) + 1
    repeating_prompt = [1, 2, 3, 4] * 8
    stats = {4: 0, 3: 0}
    actual = windowed(repeating_prompt, 32, stats)
    assert actual == sequential(repeating_prompt, 32)
    assert stats[4] > 0, stats
    print("701 prompt/count oracle cases passed; repeating context exercised lookup")
