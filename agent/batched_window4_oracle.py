"""Pure Python scheduler oracle for batched four-token verification."""

import random
from collections import deque


def greedy(prefix, lane):
    if lane % 2 == 0:
        return lane  # high acceptance lanes create uneven queues
    value = 0xCBF29CE484222325
    for token in prefix:
        value = ((value ^ (token + 1)) * 0x100000001B3) & ((1 << 64) - 1)
    return (value + lane) % 29


def sequential(prompts, count):
    histories = [prompt[:] for prompt in prompts]
    outputs = []
    for _ in range(count):
        step = [greedy(history, lane) for lane, history in enumerate(histories)]
        outputs.append(step)
        for history, token in zip(histories, step):
            history.append(token)
    return outputs


def batched(prompts, count):
    batch = len(prompts)
    if count <= 0:
        return [], False, False, set()
    histories = [prompt[:] for prompt in prompts]
    current = [greedy(history, b) for b, history in enumerate(histories)]
    for b in range(batch):
        histories[b].append(current[b])
    emitted = [current[:]]
    if count == 1:
        return emitted, False, False, set()
    guesses = [[token] * 3 for token in current]
    positions = [len(prompt) for prompt in prompts]
    generated = [1] * batch
    queues = [deque() for _ in range(batch)]
    uneven = frozen = False
    accept_counts = set()
    while len(emitted) < count:
        if all(queues):
            emitted.append([queue.popleft() for queue in queues])
            continue
        before = [(current[b], guesses[b][:], positions[b]) for b in range(batch)]
        was_finished = [generated[b] == count for b in range(batch)]
        for b in range(batch):
            if generated[b] == count:
                continue
            proposed = [current[b], *guesses[b]]
            prefix = histories[b][:-1]
            outputs = [greedy(prefix + proposed[:i + 1], b) for i in range(4)]
            accepted = 1
            while accepted < 4 and guesses[b][accepted - 1] == outputs[accepted - 1]:
                accepted += 1
            accept_counts.add(accepted)
            take = min(accepted, count - generated[b])
            queues[b].extend(outputs[:take])
            histories[b].extend(outputs[:take])
            generated[b] += take
            if generated[b] < count:
                current[b] = outputs[accepted - 1]
                guesses[b] = outputs[accepted:] + [outputs[-1]] * (3 - len(outputs[accepted:]))
                positions[b] += accepted
        if len(set(map(len, queues))) > 1:
            uneven = True
        for b in range(batch):
            if was_finished[b]:
                assert (current[b], guesses[b], positions[b]) == before[b], "finished row changed"
                frozen = True
    return emitted, uneven, frozen, accept_counts


if __name__ == "__main__":
    rng = random.Random(18)
    exercised_uneven = exercised_frozen = False
    all_accept_counts = set()
    cases = 0
    for batch in range(1, 5):
        for seed in range(20):
            prompts = [[rng.randrange(29) for _ in range(8 + seed % 9)]
                       for _ in range(batch)]
            for count in (1, 2, 3, 5, 9, 17):
                actual, uneven, frozen, accepts = batched(prompts, count)
                assert actual == sequential(prompts, count), (batch, seed, count)
                assert len(actual) == count and all(len(step) == batch for step in actual)
                exercised_uneven |= uneven
                exercised_frozen |= frozen
                all_accept_counts.update(accepts)
                cases += 1
    assert exercised_uneven and exercised_frozen
    assert 1 in all_accept_counts and 4 in all_accept_counts, all_accept_counts
    print(f"{cases} batched window oracle cases passed; uneven queues and frozen rows exercised")
