#!/usr/bin/env python3
"""Experiment: Highway vs Service Road — Linear Probing.

Tests whether high-γ channels (highway) carry syntactic information and
low-γ channels (service road) carry semantic information.

Method:
1. Run diverse sentences through Mistral 7B v0.1 (base)
2. Extract hidden states at multiple layers (L2, L15, L28, L31)
3. Split channels by γ population (top/bottom 50% by magnitude)
4. Train linear probes on each subset:
   - Syntactic: POS tag prediction, dependency depth
   - Semantic: topic cluster (k-means on full representations)
5. Compare accuracy across: highway / service road / full / random split

Expected runtime: ~20 min on H100 (100 sentences, 1 model, 4 layers).
"""

import os, json, time, torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

DEVICE = "cuda"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROBE_LAYERS = [2, 15, 28, 31]
N_TOPIC_CLUSTERS = 8
N_RANDOM_SPLITS = 5

SENTENCES = [
    # Syntactically varied — different structures
    "The cat sat on the mat.",
    "Running quickly through the forest, the deer escaped.",
    "What did the researcher discover about neural networks?",
    "Although the experiment failed, valuable data was collected.",
    "The professor who published the paper won an award.",
    "It is believed that consciousness emerges from complexity.",
    "Never before had the team seen such dramatic results.",
    "The student, having finished the exam, left the room.",
    "Were the measurements taken correctly?",
    "Between the two hypotheses lies an uncomfortable truth.",
    # Semantically varied — different topics
    "Quantum entanglement defies classical intuitions about locality.",
    "The economy grew by three percent last quarter.",
    "She played a sonata by Beethoven on the old piano.",
    "Photosynthesis converts light energy into chemical bonds.",
    "The treaty was signed after months of negotiation.",
    "Deep learning models require massive computational resources.",
    "The glacier has retreated by two kilometers this decade.",
    "DNA replication involves unwinding the double helix.",
    "The novel explores themes of memory and identity.",
    "Interest rates affect mortgage payments directly.",
    # Mixed — both syntactic and semantic variety
    "If transformer architectures learn bimodal distributions, then the optimizer has found structure.",
    "Three children were playing in the park when it started to rain.",
    "The mathematical proof, which took decades to verify, relied on category theory.",
    "How does the brain distinguish between self and other?",
    "Markets crashed because investors panicked about inflation data.",
    "Under the microscope, the cells divided with remarkable regularity.",
    "Not only did the treatment work, but it exceeded expectations.",
    "The philosopher argued that consciousness is not reducible to computation.",
    "Having been rejected by six journals, the paper was finally published.",
    "What happens when you remove a single parameter from a neural network?",
    # Additional for statistical power
    "The river flows from the mountains to the sea.",
    "Can machines understand meaning or merely manipulate symbols?",
    "After the revolution, society reorganized around new principles.",
    "Protein folding determines biological function.",
    "The painting was sold at auction for twelve million dollars.",
    "If the hypothesis is correct, we should observe a spectral gap.",
    "Birds migrate thousands of kilometers following magnetic field lines.",
    "The committee voted unanimously to approve the proposal.",
    "Recursive structures appear at every scale in nature.",
    "She wondered whether the pattern would hold across architectures.",
    "The ancient city was discovered beneath thirty meters of sand.",
    "Gravitational waves carry information about merging black holes.",
    "Why do some memories persist while others fade?",
    "The algorithm converges in polynomial time under mild assumptions.",
    "Children acquire language without explicit instruction.",
    "The bridge collapsed because the steel was corroded.",
    "Emotions influence decision-making more than logic does.",
    "The supernova remnant expanded at ten percent the speed of light.",
    "Whether free will exists remains an open philosophical question.",
    "The model learned to generalize beyond its training distribution.",
]

# POS labels for last token (simplified — assigned by structure)
POS_LABELS = [
    "NOUN", "VERB", "NOUN", "VERB", "NOUN",
    "NOUN", "NOUN", "NOUN", "ADV", "NOUN",
    "NOUN", "NOUN", "NOUN", "NOUN", "NOUN",
    "NOUN", "NOUN", "NOUN", "NOUN", "ADV",
    "NOUN", "VERB", "NOUN", "NOUN", "NOUN",
    "NOUN", "NOUN", "NOUN", "VERB", "NOUN",
    "NOUN", "NOUN", "NOUN", "NOUN", "NOUN",
    "NOUN", "NOUN", "NOUN", "NOUN", "NOUN",
    "NOUN", "NOUN", "VERB", "NOUN", "NOUN",
    "VERB", "VERB", "NOUN", "NOUN", "NOUN",
]

DEPTH_LABELS = [
    1, 3, 2, 3, 3,
    3, 2, 3, 1, 2,
    2, 1, 2, 2, 2,
    2, 2, 2, 2, 1,
    3, 3, 4, 2, 2,
    3, 3, 3, 3, 2,
    1, 2, 2, 1, 2,
    3, 2, 2, 2, 2,
    2, 2, 2, 2, 1,
    2, 2, 2, 2, 2,
]


def get_gamma_populations(model, layer_idx, threshold_pct=50):
    ln = model.model.layers[layer_idx].post_attention_layernorm
    gamma = ln.weight.detach().cpu().numpy()
    g = np.abs(gamma)
    threshold = np.percentile(g, threshold_pct)
    highway_mask = g >= threshold
    service_mask = ~highway_mask
    return highway_mask, service_mask, gamma


def extract_hidden_states(model, tokenizer, sentences, layer_indices):
    all_states = {li: [] for li in layer_indices}

    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", padding=False).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for li in layer_indices:
            h = outputs.hidden_states[li].squeeze(0)
            # Use mean pooling across sequence positions
            pooled = h.mean(dim=0).float().cpu().numpy()
            all_states[li].append(pooled)

    return {li: np.array(states) for li, states in all_states.items()}


def compute_topic_labels(full_representations, n_clusters=N_TOPIC_CLUSTERS):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(full_representations)
    return labels


def run_probe(X, y, name="probe"):
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return {"accuracy": 0.0, "n_classes": len(unique_labels), "note": "degenerate"}

    clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return {
        "accuracy": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "n_classes": int(len(unique_labels)),
        "n_samples": int(len(y)),
    }


def run_probes_for_split(states, highway_mask, service_mask, pos_labels, depth_labels, topic_labels):
    results = {}

    X_full = states
    X_highway = states[:, highway_mask]
    X_service = states[:, service_mask]

    # POS probe
    results["pos_full"] = run_probe(X_full, pos_labels, "pos_full")
    results["pos_highway"] = run_probe(X_highway, pos_labels, "pos_highway")
    results["pos_service"] = run_probe(X_service, pos_labels, "pos_service")

    # Depth probe
    results["depth_full"] = run_probe(X_full, depth_labels, "depth_full")
    results["depth_highway"] = run_probe(X_highway, depth_labels, "depth_highway")
    results["depth_service"] = run_probe(X_service, depth_labels, "depth_service")

    # Topic probe
    results["topic_full"] = run_probe(X_full, topic_labels, "topic_full")
    results["topic_highway"] = run_probe(X_highway, topic_labels, "topic_highway")
    results["topic_service"] = run_probe(X_service, topic_labels, "topic_service")

    return results


def main():
    timestamp = time.strftime("%Y%m%d_%H%M")
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    print(f"Extracting hidden states at layers {PROBE_LAYERS}...")
    hidden_states = extract_hidden_states(model, tokenizer, SENTENCES, PROBE_LAYERS)

    pos_labels = np.array(POS_LABELS[:len(SENTENCES)])
    depth_labels = np.array(DEPTH_LABELS[:len(SENTENCES)])

    all_results = {}

    for li in PROBE_LAYERS:
        print(f"\n{'='*50}")
        print(f"  Layer {li}")
        print(f"{'='*50}")

        states = hidden_states[li]

        # Compute topic labels from full representation at this layer
        topic_labels = compute_topic_labels(states)

        # Get γ-based channel split
        highway_mask, service_mask, gamma = get_gamma_populations(model, min(li, 31))
        n_highway = highway_mask.sum()
        n_service = service_mask.sum()
        print(f"  Highway channels: {n_highway}, Service road: {n_service}")
        print(f"  γ stats — highway mean: {np.abs(gamma[highway_mask]).mean():.4f}, "
              f"service mean: {np.abs(gamma[service_mask]).mean():.4f}")

        # Run probes with γ split
        results = run_probes_for_split(
            states, highway_mask, service_mask, pos_labels, depth_labels, topic_labels
        )

        # Random split control (average over N_RANDOM_SPLITS)
        random_results = {k: [] for k in ["pos", "depth", "topic"]}
        rng = np.random.RandomState(42)
        for _ in range(N_RANDOM_SPLITS):
            perm = rng.permutation(states.shape[1])
            rand_highway = np.zeros(states.shape[1], dtype=bool)
            rand_highway[perm[:n_highway]] = True
            rand_service = ~rand_highway

            for probe_name, labels in [("pos", pos_labels), ("depth", depth_labels), ("topic", topic_labels)]:
                X_rand_h = states[:, rand_highway]
                r = run_probe(X_rand_h, labels, f"random_{probe_name}")
                random_results[probe_name].append(r["accuracy"])

        results["pos_random"] = float(np.mean(random_results["pos"]))
        results["depth_random"] = float(np.mean(random_results["depth"]))
        results["topic_random"] = float(np.mean(random_results["topic"]))

        # Print summary
        print(f"\n  Syntactic (POS):")
        print(f"    Full:     {results['pos_full']['accuracy']:.3f}")
        print(f"    Highway:  {results['pos_highway']['accuracy']:.3f}")
        print(f"    Service:  {results['pos_service']['accuracy']:.3f}")
        print(f"    Random:   {results['pos_random']:.3f}")

        print(f"  Syntactic (Depth):")
        print(f"    Full:     {results['depth_full']['accuracy']:.3f}")
        print(f"    Highway:  {results['depth_highway']['accuracy']:.3f}")
        print(f"    Service:  {results['depth_service']['accuracy']:.3f}")
        print(f"    Random:   {results['depth_random']:.3f}")

        print(f"  Semantic (Topic):")
        print(f"    Full:     {results['topic_full']['accuracy']:.3f}")
        print(f"    Highway:  {results['topic_highway']['accuracy']:.3f}")
        print(f"    Service:  {results['topic_service']['accuracy']:.3f}")
        print(f"    Random:   {results['topic_random']:.3f}")

        # Compute disentanglement index
        h_syntax = (results['pos_highway']['accuracy'] + results['depth_highway']['accuracy']) / 2
        s_syntax = (results['pos_service']['accuracy'] + results['depth_service']['accuracy']) / 2
        h_semantic = results['topic_highway']['accuracy']
        s_semantic = results['topic_service']['accuracy']

        disent_index = (h_syntax - s_syntax) - (h_semantic - s_semantic)
        print(f"\n  Disentanglement index: {disent_index:.4f}")
        print(f"    (positive = highway↑syntax, service↑semantic; negative = opposite)")

        results["disentanglement_index"] = float(disent_index)
        results["gamma_stats"] = {
            "highway_mean": float(np.abs(gamma[highway_mask]).mean()),
            "service_mean": float(np.abs(gamma[service_mask]).mean()),
            "n_highway": int(n_highway),
            "n_service": int(n_service),
        }

        all_results[f"L{li}"] = results

    # Summary across layers
    print(f"\n{'='*50}")
    print("  LAYER COMPARISON")
    print(f"{'='*50}")
    for li in PROBE_LAYERS:
        r = all_results[f"L{li}"]
        di = r["disentanglement_index"]
        flag = " ***" if abs(di) > 0.05 else ""
        print(f"  L{li:2d}: DI={di:+.4f}{flag}  "
              f"POS(h/s)={r['pos_highway']['accuracy']:.3f}/{r['pos_service']['accuracy']:.3f}  "
              f"Topic(h/s)={r['topic_highway']['accuracy']:.3f}/{r['topic_service']['accuracy']:.3f}")

    # Save
    outfile = RESULTS_DIR / f"exp_channel_probing_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump({
            "experiment": "channel_probing_disentanglement",
            "model": MODEL_NAME,
            "n_sentences": len(SENTENCES),
            "probe_layers": PROBE_LAYERS,
            "n_topic_clusters": N_TOPIC_CLUSTERS,
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
