"""
data.pipelines.math_pipeline
=============================
Math training data pipeline.

Generates algebra, calculus, probability, and linear algebra problems
with full step-by-step solutions at configurable scale and difficulty.

Usage:
    cell-ai data --pipeline math --count 100000
    cell-ai data --pipeline math --count 1000000 --difficulty-dist 0.3,0.4,0.2,0.1
    cell-ai data --pipeline math --stats
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DOMAINS = ["algebra", "calculus", "probability", "linear_algebra", "number_theory"]
DIFFICULTIES = ["easy", "medium", "hard", "competition"]


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------

def _gen_algebra(difficulty: str, rng: random.Random) -> Dict:
    if difficulty == "easy":
        a = rng.randint(1, 10)
        b = rng.randint(1, 20)
        c = a * rng.randint(1, 10) + b
        x = (c - b) / a
        return {
            "problem": f"Solve for x: {a}x + {b} = {c}",
            "solution": f"x = ({c} - {b}) / {a} = {x:.4g}",
            "steps": [
                f"Subtract {b} from both sides: {a}x = {c - b}",
                f"Divide both sides by {a}: x = {x:.4g}",
            ],
        }
    elif difficulty == "medium":
        a = rng.randint(1, 5)
        b = rng.randint(-5, 5)
        c = rng.randint(-10, 10)
        disc = b * b - 4 * a * c
        problem = f"Solve: {a}x² + {b}x + {c} = 0"
        if disc < 0:
            sol = "No real solutions (discriminant < 0)"
            steps = [f"Discriminant = {b}² - 4·{a}·{c} = {disc} < 0"]
        elif disc == 0:
            x = -b / (2 * a)
            sol = f"x = {x:.4g} (double root)"
            steps = [f"x = -{b} / (2·{a}) = {x:.4g}"]
        else:
            x1 = (-b + math.sqrt(disc)) / (2 * a)
            x2 = (-b - math.sqrt(disc)) / (2 * a)
            sol = f"x₁ = {x1:.4g}, x₂ = {x2:.4g}"
            steps = [f"Discriminant = {disc}", f"x = (-{b} ± √{disc}) / (2·{a})", f"x₁ = {x1:.4g}, x₂ = {x2:.4g}"]
        return {"problem": problem, "solution": sol, "steps": steps}
    else:
        a = rng.randint(1, 3)
        b = rng.randint(-3, 3)
        c = rng.randint(-5, 5)
        d = rng.randint(-3, 3)
        return {
            "problem": f"Factor: {a}x³ + {b}x² + {c}x + {d}",
            "solution": f"Use rational root theorem and polynomial division.",
            "steps": [
                f"Possible rational roots: ±factors({abs(d)})/factors({a})",
                "Test candidates by substitution",
                "Apply polynomial long division once root found",
            ],
        }


def _gen_calculus(difficulty: str, rng: random.Random) -> Dict:
    funcs = [
        ("x²",        "2x",              "∫x²dx = x³/3 + C"),
        ("sin(x)",    "cos(x)",           "∫sin(x)dx = -cos(x) + C"),
        ("eˣ",        "eˣ",              "∫eˣdx = eˣ + C"),
        ("ln(x)",     "1/x",             "∫ln(x)dx = x·ln(x) - x + C"),
        ("x³ + 2x",   "3x² + 2",         "∫(x³+2x)dx = x⁴/4 + x² + C"),
        ("cos(x)",    "-sin(x) (d/dx),  sin(x) + C (∫)", "sin(x) + C"),
    ]
    func, deriv, integral = rng.choice(funcs)
    if difficulty in ("easy", "medium"):
        op = rng.choice(["differentiate", "integrate"])
        if op == "differentiate":
            return {
                "problem": f"Find d/dx [{func}]",
                "solution": deriv,
                "steps": [f"Apply standard differentiation rule to {func}"],
                "domain": "calculus",
            }
        else:
            return {
                "problem": f"Evaluate ∫{func} dx",
                "solution": integral,
                "steps": [f"Apply standard integration rule to {func}"],
                "domain": "calculus",
            }
    else:
        a = rng.randint(0, 3)
        b = rng.randint(a + 1, a + 5)
        return {
            "problem": f"Evaluate ∫₍{a}₎^{b} {func} dx",
            "solution": f"Apply FTC: F({b}) - F({a}) where F is the antiderivative",
            "steps": ["Find antiderivative F(x)", f"Evaluate F({b}) - F({a})"],
        }


def _gen_probability(difficulty: str, rng: random.Random) -> Dict:
    if difficulty == "easy":
        n = rng.randint(2, 6)
        k = rng.randint(1, n)
        p = round(k / n, 4)
        return {
            "problem": f"A fair {n}-sided die is rolled. P(result ≤ {k}) = ?",
            "solution": f"{k}/{n} = {p}",
            "steps": [f"Favorable outcomes: {k}", f"Total outcomes: {n}", f"P = {k}/{n}"],
        }
    else:
        n = rng.randint(5, 20)
        k = rng.randint(1, min(n, 5))
        p = round(rng.uniform(0.1, 0.9), 2)
        binom = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        return {
            "problem": f"X ~ Binomial(n={n}, p={p}). P(X={k}) = ?",
            "solution": f"C({n},{k}) · {p}^{k} · {1-p:.2f}^{n-k} ≈ {binom:.6f}",
            "steps": [
                f"C({n},{k}) = {math.comb(n, k)}",
                f"{p}^{k} = {p**k:.6f}",
                f"{1-p:.2f}^{n-k} = {(1-p)**(n-k):.6f}",
                f"Product = {binom:.6f}",
            ],
        }


def _gen_linear_algebra(difficulty: str, rng: random.Random) -> Dict:
    if difficulty in ("easy", "medium"):
        a = [[rng.randint(-3, 3) for _ in range(2)] for _ in range(2)]
        det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        mat_str = f"[[{a[0][0]}, {a[0][1]}], [{a[1][0]}, {a[1][1]}]]"
        return {
            "problem": f"Find the determinant of matrix {mat_str}",
            "solution": f"det = {a[0][0]}·{a[1][1]} - {a[0][1]}·{a[1][0]} = {det}",
            "steps": [
                "Apply 2×2 determinant formula: ad - bc",
                f"= {a[0][0]}×{a[1][1]} - {a[0][1]}×{a[1][0]} = {det}",
            ],
        }
    else:
        n = 3
        a = [[rng.randint(-2, 2) for _ in range(n)] for _ in range(n)]
        mat_str = str(a)
        return {
            "problem": f"Find the eigenvalues of {mat_str}",
            "solution": "Solve det(A - λI) = 0 using characteristic polynomial.",
            "steps": [
                "Form A - λI",
                "Compute det(A - λI) = 0",
                "Solve the resulting cubic/quadratic for λ",
            ],
        }


GENERATORS = {
    "algebra":       _gen_algebra,
    "calculus":      _gen_calculus,
    "probability":   _gen_probability,
    "linear_algebra": _gen_linear_algebra,
}


def generate(
    count: int = 100_000,
    output_dir: Optional[Path] = None,
    difficulty_dist: Optional[List[float]] = None,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Generate `count` math problems and write to output_dir.

    Args:
        count:           total number of problems to generate
        output_dir:      directory (default: DATA_ROOT/math/generated)
        difficulty_dist: [easy, medium, hard, competition] proportions (sum=1)
        seed:            random seed for reproducibility

    Returns:
        dict of counts per difficulty and domain
    """
    from data.config import MATH_DIR, ensure_dirs
    ensure_dirs()
    output_dir = output_dir or MATH_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    difficulty_dist = difficulty_dist or [0.35, 0.40, 0.20, 0.05]
    assert abs(sum(difficulty_dist) - 1.0) < 1e-6, "difficulty_dist must sum to 1"

    rng = random.Random(seed)
    counts: Dict[str, int] = {d: 0 for d in DIFFICULTIES}

    train_out = open(output_dir / "train.jsonl", "w", encoding="utf-8")
    val_out   = open(output_dir / "val.jsonl",   "w", encoding="utf-8")
    test_out  = open(output_dir / "test.jsonl",  "w", encoding="utf-8")

    try:
        for i in range(count):
            # Sample difficulty
            r = rng.random()
            cumulative = 0.0
            difficulty = DIFFICULTIES[-1]
            for d, prob in zip(DIFFICULTIES, difficulty_dist):
                cumulative += prob
                if r < cumulative:
                    difficulty = d
                    break

            domain = rng.choice(list(GENERATORS.keys()))
            try:
                record = GENERATORS[domain](difficulty, rng)
            except Exception:
                continue

            record["domain"] = domain
            record["difficulty"] = difficulty
            record["id"] = i

            line = json.dumps(record, ensure_ascii=False) + "\n"
            split_r = rng.random()
            if split_r < 0.95:
                train_out.write(line)
            elif split_r < 0.975:
                val_out.write(line)
            else:
                test_out.write(line)

            counts[difficulty] += 1

            if (i + 1) % 50_000 == 0:
                logger.info(f"  Generated {i+1:,}/{count:,} problems ...")
    finally:
        train_out.close()
        val_out.close()
        test_out.close()

    logger.info(f"Generated {count:,} problems in {output_dir}")
    for d, n in counts.items():
        logger.info(f"  {d}: {n:,}")
    return counts


def stats(output_dir: Optional[Path] = None) -> Dict:
    from data.config import MATH_DIR
    output_dir = output_dir or MATH_DIR

    def count_lines(p: Path) -> int:
        try:
            return sum(1 for _ in open(p, encoding="utf-8"))
        except Exception:
            return 0

    return {f.name: count_lines(f) for f in output_dir.glob("*.jsonl")} if output_dir.exists() else {}
