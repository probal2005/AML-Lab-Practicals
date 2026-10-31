#!/usr/bin/env python3
"""AML Practical 8: Hidden Markov Model for Weather Sequence Prediction."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm

# ============================================================
# 1. Paths and experiment configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "experiment_output.txt"

states = ["Rainy", "Sunny"]
observations = ["Walk", "Shop", "Clean"]

# Observation codes:
# 0 = Walk, 1 = Shop, 2 = Clean
observation_sequence = np.array(
    [[0], [1], [2], [0], [0], [1], [2]],
    dtype=int,
)

# ============================================================
# 2. Define the Hidden Markov Model
# ============================================================

model = hmm.CategoricalHMM(
    n_components=len(states),
    n_features=len(observations),
    random_state=42,
)

# Initial probabilities: Rainy, Sunny
model.startprob_ = np.array([0.6, 0.4])

# Transition probability matrix:
# Rows = current state; columns = next state
model.transmat_ = np.array([
    [0.7, 0.3],
    [0.4, 0.6],
])

# Emission probability matrix:
# Columns = Walk, Shop, Clean
model.emissionprob_ = np.array([
    [0.1, 0.4, 0.5],  # Rainy
    [0.6, 0.3, 0.1],  # Sunny
])

# ============================================================
# 3. Predict the most likely hidden state sequence
# ============================================================

log_probability, hidden_state_sequence = model.decode(
    observation_sequence,
    algorithm="viterbi",
)

posterior_probabilities = model.predict_proba(observation_sequence)

# ============================================================
# 4. Generate a sample sequence
# ============================================================

generated_observations, generated_states = model.sample(10)

# ============================================================
# 5. Prepare and display the experiment output
# ============================================================

lines = [
    "=" * 66,
    "AML PRACTICAL 8: HIDDEN MARKOV MODEL",
    "Weather Sequence Prediction Using the Viterbi Algorithm",
    "=" * 66,
    "",
    "HIDDEN STATES",
    "0 = Rainy",
    "1 = Sunny",
    "",
    "OBSERVATION SYMBOLS",
    "0 = Walk",
    "1 = Shop",
    "2 = Clean",
    "",
    "START PROBABILITIES",
    str(model.startprob_),
    "",
    "TRANSITION PROBABILITY MATRIX",
    "Rows: current state; columns: next state",
    str(model.transmat_),
    "",
    "EMISSION PROBABILITY MATRIX",
    "Columns: Walk, Shop, Clean",
    str(model.emissionprob_),
    "",
    "OBSERVATION SEQUENCE",
    " -> ".join(observations[row[0]] for row in observation_sequence),
    "",
    "PREDICTED HIDDEN STATE SEQUENCE (VITERBI)",
    " -> ".join(states[int(s)] for s in hidden_state_sequence),
    "",
    f"Log Probability of Observation Sequence: {log_probability:.6f}",
    "",
    "OBSERVATION AND PREDICTED WEATHER TABLE",
    "-" * 66,
    f"{'Time':<8}{'Observation':<16}{'Predicted State':<20}",
    "-" * 66,
]

for i, (obs, state) in enumerate(
    zip(observation_sequence, hidden_state_sequence), start=1
):
    lines.append(
        f"{i:<8}{observations[int(obs[0])]:<16}{states[int(state)]:<20}"
    )

lines.extend([
    "",
    "POSTERIOR STATE PROBABILITIES",
    "These are probabilities for each hidden state given the full",
    "observation sequence, calculated by the forward-backward method.",
])

for i, (obs, probabilities) in enumerate(
    zip(observation_sequence, posterior_probabilities), start=1
):
    lines.append(
        f"Time {i}: {observations[int(obs[0])]:<6} | "
        f"Rainy = {probabilities[0]:.4f}, "
        f"Sunny = {probabilities[1]:.4f}"
    )

lines.extend([
    "",
    "GENERATED SAMPLE OBSERVATIONS",
    " -> ".join(observations[int(row[0])] for row in generated_observations),
    "",
    "GENERATED SAMPLE HIDDEN STATES",
    " -> ".join(states[int(s)] for s in generated_states),
    "",
    "ANALYSIS",
    "- The Viterbi algorithm finds the single most likely hidden-state path.",
    "- Transition probabilities describe movement between weather states.",
    "- Emission probabilities describe activities associated with each state.",
    "- Posterior probabilities quantify uncertainty at each time step.",
    "- The generated sample is stochastic; its sequence may vary by configuration.",
    "",
    "EXPERIMENT STATUS: COMPLETED",
    "=" * 66,
])

report = "\n".join(lines)

print(report)
OUTPUT_FILE.write_text(report + "\n", encoding="utf-8")

# ============================================================
# 6. Graph 1: Predicted weather sequence
# ============================================================

time_steps = np.arange(1, len(observation_sequence) + 1)
observation_labels = [
    observations[int(row[0])] for row in observation_sequence
]
predicted_labels = [
    states[int(state)] for state in hidden_state_sequence
]

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.step(
    time_steps,
    hidden_state_sequence,
    where="mid",
    linewidth=2,
    marker="o",
    markersize=9,
    label="Predicted hidden state",
)
ax.set_yticks([0, 1], states)
ax.set_xticks(time_steps, observation_labels)
ax.set_ylim(-0.35, 1.35)
ax.set_xlabel("Observed activity")
ax.set_ylabel("Predicted weather state")
ax.set_title("HMM Weather Prediction: Viterbi Hidden-State Sequence")
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend()
for x, state in zip(time_steps, predicted_labels):
    ax.annotate(
        state,
        (x, states.index(state)),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "predicted_weather_sequence.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 7. Graph 2: Posterior state probabilities
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(
    time_steps,
    posterior_probabilities[:, 0],
    marker="o",
    linewidth=2,
    label="Rainy",
)
ax.plot(
    time_steps,
    posterior_probabilities[:, 1],
    marker="s",
    linewidth=2,
    label="Sunny",
)
ax.set_xticks(time_steps, observation_labels)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Observed activity")
ax.set_ylabel("Posterior probability")
ax.set_title("Posterior Probability of Each Weather State")
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend()
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "state_probability_chart.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 8. Graph 3: Transition probability heatmap
# ============================================================

fig, ax = plt.subplots(figsize=(7.5, 5.5))
image = ax.imshow(model.transmat_, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(states)), states)
ax.set_yticks(range(len(states)), states)
ax.set_xlabel("Next state")
ax.set_ylabel("Current state")
ax.set_title("HMM Weather-State Transition Probability Matrix")

for row in range(len(states)):
    for col in range(len(states)):
        value = model.transmat_[row, col]
        ax.text(
            col,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=13,
            color="black",
        )

fig.colorbar(image, ax=ax, label="Transition probability")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "transition_matrix.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# 9. Final verification
# ============================================================

print("\nGRAPH FILES CREATED:")
for filename in (
    "predicted_weather_sequence.png",
    "state_probability_chart.png",
    "transition_matrix.png",
):
    path = OUTPUT_DIR / filename
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Graph was not created correctly: {path}")
    print(f"  OK: {path.name} ({path.stat().st_size:,} bytes)")

if not OUTPUT_FILE.is_file():
    raise RuntimeError("Experiment output report was not saved.")

print(f"\nText report saved to: {OUTPUT_FILE}")
print("Experiment completed successfully.")
