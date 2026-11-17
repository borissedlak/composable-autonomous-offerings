# B.1–C.1 with 10 seeds instead of 3 (2026-11-17)

Change: `seeds = list(range(10))` in `notebooks/tsc/pegasus/evaluation_setup.py` (was `range(3)`). A seed sets the
training draw (which 50 benchmark runs per service are drawn), the data split, and the GP seed (`35 + s`).
Every result below is a median over the 10 training draws; "holds" still means coverage ≥ 95% under *every* seed, so
more seeds make "holds" strictly harder to reach.

Rerun: 2_B1, 2_B3, 2_B4, 2_C1 (recomputed from scratch, since editing `evaluation_setup.py` invalidates the cache), plus
`gamma_calibration.py` (it also loops over `seeds`; it gives the Cov. column of Table I and the 0.92 / 0.91 numbers).
In `6_evaluation.tex`, each changed number is marked with `\hl{}`. "Paper" below is the value that was in the tex.
The tex did not always match the last 3-seed run, so both are listed.

## What changed

| Where | Metric | Paper | 3 seeds (last run) | **10 seeds** |
|---|---|---|---|---|
| B.1 | Pegasus: chains held / median tightness | 15 / 1.81× | 15 / 1.94× | **15 / 1.97×** |
| B.1 | Quantile Union: chains held / tightness | 13 / 1.34× | 11 / 1.46× | **5 / 1.46×** |
| B.1 | GP Queue: chains held / tightness | 7 / 1.15× | 4 / 1.17× | **3 / 1.17×** |
| B.1 | Conformal: tightness (holds on all 15) | 3.67× | 4.31× | **4.94×** |
| B.1 | γ coverage, services inside compositions | 0.91 | 0.91 | **0.92** (benchmark runs stay at 0.92) |
| Table I | Cov. A3 / L1 / L3 / L4 / L5 / V1 / V4 | .95/.95/.77/.78/.91/.93/.90 | same | **.92/.94/.80/.84/.88/.94/.92** |
| C.1 | Pegasus (pretrained), n = 0 → n = 100 | 1.78× → 1.34× | same | **1.83× → 1.37×** |
| C.1 | Pegasus (blank): fails at / n = 100 | one n / 1.22× | n = 80 / 1.22× | **n = 20 and 80 / 1.29×** |
| B.4a | Pegasus, u ≤ 0.7 / u = 0.9 (share ∞) | 1.53–1.90× / 4.09× (35%) | 1.55–1.88× / 4.14× (38%) | **1.62–1.91× / 3.89× (38%)** |
| B.4b | Pegasus, C² = 0 → C² = 4 | 1.15× → 2.13× | 1.14× → 2.14× | **1.13× → 2.18×** |
| B.4b | Quantile Union holds up to | C² = 1 | C² = 1 | **C² = 0.3** |
| B.3 | Conformal at h = 6 | holds | holds (1.60×) | **fails (1.56×)**; Pegasus is the only method that holds at h = 6 |
| B.3 | Pegasus inference time, h = 6 | "≈ 300 ms" | 790 ms | **570 ms** (wall clock; noisy, since 4 notebooks ran in parallel) |

What the changes mean:
- **Pegasus's conclusions all hold:** it still holds on 15/15 chains, at every load, and at every tree height. Its
  tightness moves by a few hundredths.
- **The baselines lose "holds" status.** With 10 draws there are more chances for one bad draw, and a method holds
  only if it covers under every draw. Quantile Union drops from 11 to 5 chains in B.1 and now holds in B.4b only up to C² = 0.3.
  Conformal now fails at h = 6 in B.3. This strengthens the paper's argument but changes several sentences.
- **Pegasus (blank) in C.1** now also fails at n = 20, so "all n but one" becomes "all n but two".

## Unchanged
- **B.2:** it already used 10 seeds (`b2_seeds`), and its outputs are identical.
- **B.3:** Pegasus, GP Sum, GP Queue, and Oracle tightness change by at most 0.03×; training times and the slowdown table are unchanged.
- **B.4:** GP Queue never holds, and Conformal still holds up to C² = 2.
- **C.2:** not rerun (it uses one fixed seed). Its cache key includes `evaluation_setup.py`, so its next run recomputes, with identical results.

## Not changed in the tex, but worth checking (independent of the seed count)
- B.4b: "Pegasus holds for every $C_a^2$". At C² = 0 Pegasus does **not** hold (○), both with 3 and with 10 seeds.
- B.3: "training the 61 GPs at h = 6 takes less than 10 s". The notebook reports 15 s, both before and after.
- Fig. 3 caption: "on average 7× more accurate" than Conformal. I can't derive the 7× from the B.1 output
  (the tightness ratio is 4.94 / 1.97 ≈ 2.5).
