# Midterm Defense Slide Deck — Source Document

**Instruction to Claude (or any deck generator):** Turn this file into a slide deck (PowerPoint / Google Slides / Keynote). Each `## Slide N` heading is one slide. Use the **Title**, **Body** and **Visual** fields as the slide content; put **Speaker notes** in the notes pane, not on the slide. Do not add slides that are not here. Do not compress two slides into one.

**Design direction:** clean academic, high contrast, one idea per slide. Large numbers should be genuinely large. Use a single accent color for "our model" and a neutral gray for "baseline" consistently across every chart and table. Slides marked `[RESULT]` carry a number the panel will write down — give those the most visual weight.

**Presenter:** L.U.L. Ramoy
**Thesis:** *HR-ETE-GNN: Robust Volatility Forecasting via Hurst-Regime-Adaptive Rényi Effective Transfer Entropy Graph Neural Networks*
**Occasion:** Midterm defense — experiment design + preliminary results
**Audience assumption:** intelligent panelists who are **not** specialists in volatility forecasting, information theory, or graph neural networks. Every technical term is defined in plain English *before* it is used.

**Image assets available in `preliminary_results/`:**
`F1_sample_and_regimes.png`, `F2_adjacency_heatmaps.png`, `F3_transmitters.png`, `F4_forecasts.png`, `F5_dashboard.png`

**Tone rule for the whole deck:** explain it the way you would to a smart friend over coffee. Analogy first, then the technical name, then the number. Never the reverse.

---

# ACT 0 — Framing

## Slide 1 — Title

**Title:** Predicting How Wild Tomorrow Will Be
**Subtitle:** Forecasting stock market volatility by measuring how panic travels between countries
**Body:**
- L.U.L. Ramoy
- HR-ETE-GNN — Hurst-Regime-Adaptive Rényi Effective Transfer Entropy Graph Neural Network
- Midterm Defense — Experiment Design & Preliminary Results

**Speaker notes:** The title of the thesis is a mouthful. Don't defend the acronym yet — by slide 20 every letter in it will have been explained one at a time. Open by saying exactly that: "I'm going to unpack that title one piece at a time, and I'm going to start from zero."

---

## Slide 2 — The one-sentence version

**Title:** The Whole Thesis in One Sentence
**Body:**
> I build a system that predicts **how wildly a country's stock market will swing tomorrow**, by watching **how nervousness spreads between countries** — the way a weather service predicts tomorrow's storm by watching the pressure systems moving in from next door.

Three ideas hidden in that sentence:

| Idea | Everyday version | Where I explain it |
|---|---|---|
| "how wildly a market swings" | How bumpy is the ride? | Act 1 |
| "predicting it" | Bumpiness is streaky — yesterday hints at today | Act 1 |
| "nervousness spreads between countries" | Panic is contagious, and it travels in a *direction* | Act 3 |

**Visual:** Weather-map metaphor illustration, or a simple 3-icon row.

**Speaker notes:** This is the anchor slide. If a panelist gets lost later, come back to this sentence. Note the weather analogy is load-bearing all the way through: we're not predicting whether it rains up or down, we're predicting how strong the wind will be.

---

## Slide 3 — Roadmap

**Title:** Where We're Going
**Body:**
1. **What am I predicting?** — Realized Volatility, built from scratch
2. **Research Question 1** — Does my model forecast better than the existing one? (H1a, H1b)
3. **Research Question 2** — Do the two ways of measuring produce *different maps* of the world? (H2a, H2b)
4. **Preliminary results** — what the pilot run actually says
5. **What changes before the final run**

**Speaker notes:** Flag up front that RQ1 and RQ2 are genuinely different questions — RQ1 is "does it predict better", RQ2 is "does it *see* differently". RQ2 can succeed even if RQ1 fails. Say this early; it buys goodwill when the preliminary RQ1 numbers come up.

---

# ACT 1 — What is Realized Volatility?

## Slide 4 — Prices are the wrong thing to look at

**Title:** Step 1: Stop Looking at Prices
**Body:**
- A German share costs ~$30. A Japanese share costs ~$60.
- A "$1 move" means something completely different in each.
- So the first thing anyone does is convert to a **return** — the *percentage* change from yesterday to today.
- Returns are comparable across countries, across time, across price levels.

**Visual:** Left panel = raw price line of the German ETF (EWG) over six years. Right panel = the same series as daily % returns.

**Speaker notes:** Keep this fast. It's a warm-up. Point at the return chart and say: "notice it hovers around zero and looks like noise — which way it goes tomorrow is basically unpredictable. But notice the *width of the band* changes. That width is what this thesis is about."

---

## Slide 5 — Volatility is about the size of the bumps, not the direction

**Title:** Two Investments. Same Start. Same Finish. Not the Same.
**Body:**
- Investment A: climbs smoothly to $120.
- Investment B: ends at $120 too — after crashing to $60 and back twice.
- A return-based scorecard calls these **identical**: same start, same end, same profit.
- Nobody would call them the same. B could have forced you to sell at the worst moment, or your broker could have closed you out at the bottom.

> **Volatility is the number that tells these two apart.** In finance it *is* the working definition of risk.

**Visual:** Two line charts overlaid, same endpoints, wildly different paths.

**Speaker notes:** This is the emotional hook of the whole first act. Let it land. Ask the panel rhetorically: "would you rather hold A or B?" — everyone has an intuition here, and that intuition IS volatility.

---

## Slide 6 — Stress-testing the idea: four made-up scenarios

**Title:** Does Our Ruler Actually Measure the Right Thing?
**Body:**
Before touching real data, test the *concept* against four deliberately extreme invented price paths. A good risk measure should react sensibly to all four.

| # | Scenario | What the price does |
|---|---|---|
| 1 | **Steadily rising** | The same small +0.30% every single day |
| 2 | **Oscillating +100 / −100** | Slams between the two extremes every step |
| 3 | **Very volatile** | Big, erratic, patternless jumps |
| 4 | **Flat / unchanged** | Nothing moves at all |

**Visual:** 2×2 grid of four price-path sparklines, unlabeled numbers, just the shapes.

**Speaker notes:** Set this up as a test of the *measure*, not of the market. "If my ruler is a good ruler, here's what it should say about each of these." Don't reveal the answers yet — the next slide is the reveal.

---

## Slide 7 — What the measure says about each scenario `[KEY CONCEPT]`

**Title:** The Four Scenarios, Scored
**Body:**

| Scenario | Realized Volatility | Why |
|---|---|---|
| **Steadily rising** | **Low, flat** | A smooth predictable climb is not risky, no matter how far it travels |
| **Oscillating ±100** | **Enormous, pinned at a ceiling** | Maximum possible move, over and over |
| **Very volatile** | **High and moving** | Unevenly sized jumps — closest to a real turbulent market |
| **Flat / unchanged** | **Exactly zero** | No movement, correctly reported as no risk |

> **The takeaway:** Realized Volatility depends **only on the size of step-to-step changes** — never on direction, never on the overall trend.

**Visual:** Same 2×2 grid as Slide 6, now with the RV curve drawn underneath each path.

**Speaker notes:** The two sharpest points to make out loud:
1. The steady climber and the flat line score almost the *same* — because each individual step is tiny either way. A trend is not risk.
2. The oscillator scores maximum even though it is perfectly *regular* and in one sense totally predictable. RV does not care about predictability. It only measures size.
This is the slide where a panelist's intuition either clicks or doesn't. Don't rush it.

---

## Slide 8 — Building the measure, four steps

**Title:** Realized Volatility, From Scratch
**Body:**

| Step | What you do | Why |
|---|---|---|
| 1 | Take the last **20** daily returns | One trading month of recent history |
| 2 | **Square** each one | Up and down both now count as "movement" |
| 3 | **Average** the squares | One number for the typical squared move |
| 4 | Take the **square root** | Puts the answer back into readable "% per day" |

$$RV_t = \sqrt{\frac{1}{20}\sum_{i=1}^{20}(100\,r_{t-i})^2}$$

**Two questions people always ask:**
- *Why square instead of just dropping the minus sign?* Because a −6% day is not merely twice as alarming as a −3% day. Squaring counts it **four** times as much.
- *Why the square root at the end?* Without it the number is in unreadable "percent squared" units.

**Speaker notes:** Show the formula only *after* the table. Say explicitly: "that formula looks intimidating written down and is completely mundane once you see it as four steps you already agreed with." The window slides forward one day at a time — a 20-day-wide moving spotlight.

---

## Slide 9 — Risk is not a constant

**Title:** A 17× Difference Between the Calmest and Wildest Periods
**Body:**
- Applied to all ten country ETFs, every day, 2018–2024.
- Sanity check passes: **emerging markets on top** (Brazil 2.06, South Africa 1.91, South Korea 1.51), **developed markets at the bottom** (Japan 1.01, Canada 1.09, UK 1.09).
- Every series is heavily right-skewed with fat tails (Brazil: skew 5.4, kurtosis 36).

> Risk is not a fixed property of a market. It moves, **enormously**, over time. Any system that treats it as a constant is catastrophically wrong roughly once a decade.

**Visual:** `preliminary_results/F1_sample_and_regimes.png` — RV over time for all ten countries, with COVID visibly exploding.

**Speaker notes:** The ranking matching finance-textbook expectations is your credibility moment — the from-scratch calculation is measuring the real thing. The fat-tail numbers (kurtosis of 36 against a normal distribution's 3) are also the first hint of *why* a tail-sensitive method might be worth trying.

---

## Slide 10 — Why forecasting volatility is even possible

**Title:** Direction Is Unpredictable. Size Is Not.
**Body:**
- Yesterday's **direction** → tells you essentially nothing about today's direction. *(autocorrelation ≈ 0)*
- Yesterday's **volatility** → tells you a great deal about today's volatility. *(autocorrelation ≈ 0.9, still positive two months out)*

> **Volatility clustering** (Mandelbrot, 1963): *"Large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes."*

Calm days cluster with calm days. Storms have a duration.

**Visual:** Bar chart — gray bars (returns, flat at zero) vs. accent bars (volatility, tall and slowly decaying) across lags 1–40.

**Speaker notes:** This gap is the licence for the entire thesis. We are not trying to beat the market on direction, which is close to impossible and would make me rich instead of a student. We are forecasting the *size* of moves, which is genuinely forecastable — and which is what risk managers, option traders and regulators actually need.

---

## Slide 11 — The task, stated precisely

**Title:** What Exactly Am I Predicting?
**Body:**
> Given everything known up to and including today (day *t*), predict **RV for tomorrow (day *t*+1)** — for each of the 10 country ETFs.

- **One number per country. One day ahead.** That is the whole task.
- 10 country ETFs: Australia, Canada, Germany, Japan, Taiwan, UK, Mexico, South Korea, Brazil, South Africa.

**Visual:** Simple diagram: [everything up to today] → [model] → [10 numbers for tomorrow].

**Speaker notes:** Anchor it in stakes with the Value-at-Risk example if there's time: a $10M German fund must state daily, by law, "with 99% confidence we will not lose more than X tomorrow", and X ≈ 2.33 × volatility × portfolio value. Everything hinges on which volatility you plug in. Getting it wrong *as a crisis starts* is how institutions blow up.

---

## Slide 12 — Countries do not move alone

**Title:** When Wall Street Has a Bad Night, Tokyo Opens Lower
**Body:**
- In March 2020, every market on earth moved together.
- **Correlation** tells us Germany and Brazil get nervous at the same time.
- Correlation **cannot** tell us who goes first — it is symmetric by construction. Germany-with-Brazil is identical to Brazil-with-Germany.

> For forecasting, **direction is everything.** "Germany moves first, Brazil follows" is usable. "They wobble together" is not.

**Visual:** March 2020 — ten normalized ETF lines all collapsing in lockstep.

**Speaker notes:** This is the bridge from Act 1 into the two research questions. The tool that *does* capture direction is transfer entropy — and that's where RQ1 and RQ2 both live.

---

# ACT 2 — Research Question 1

## Slide 13 — Transfer entropy, without the mathematics

**Title:** "Did Knowing About Germany Help Me Predict Brazil?"
**Body:**
A three-step thought experiment:

1. Guess tomorrow's Brazil using **only Brazil's own history**. You'll be wrong by some amount → call it **Error A**.
2. Guess again using **Brazil's history *and* Germany's history** → **Error B**.
3. If Error B < Error A, Germany's past contained genuinely useful information about Brazil's future that was *not already* in Brazil's own past.

> **Transfer entropy is the size of that improvement.** That's it.

| Property | Why it matters here |
|---|---|
| **Directional** | Germany→Brazil and Brazil→Germany are separate numbers. Correlation can't do this. |
| Catches **non-straight-line** relationships | Correlation only sees straight lines. Crises are not straight lines. |
| Measures **new** information | Automatically ignores whatever Brazil's own past already told us. |

**Speaker notes:** Resist all temptation to show the entropy formula. The thought experiment IS the definition for this audience. If a panelist asks for the math, it's in Chapter 3 — offer that.

---

## Slide 14 — From pairs to a network

**Title:** 90 Pairs Become a Map
**Body:**
- Compute that number for **every ordered pair** of the 10 countries — 90 pairs in total.
- Keep only the pairs where the improvement is big enough to be real rather than luck.
- What's left is a **network (graph)**: dots = countries, arrows = "this country's past helps predict that one's future", thicker arrow = more information flowing.
- That network is fed to a **Graph Neural Network**: the model forecasts each country's volatility while *listening only to the countries the arrows say are worth listening to*.

> That's what makes it a *graph* neural network — the structure isn't guessed, it's **measured from the data**.

**Visual:** A 10-node directed graph with weighted arrows.

**Speaker notes:** This explains the "ETE" and "GNN" in the title. Two letters left: the R (Rényi) and the H (Hurst). Those are the next two slides, and they're where the contribution lives.

---

## Slide 15 — The α dial: the actual contribution `[KEY CONCEPT]`

**Title:** Ordinary Transfer Entropy Treats a Crash Like a Sleepy Tuesday
**Body:**
Standard transfer entropy weights every day equally when it takes its average. **That is a strange choice for a thesis about crises.**

**Rényi** transfer entropy adds a tuning dial, **α**:

| Setting | Effect | Best for |
|---|---|---|
| **α < 1** | Amplifies **rare, extreme** days | Crashes and panics |
| **α = 1** | Everything counts equally — *this is ordinary transfer entropy* | The neutral default |
| **α > 1** | Amplifies **ordinary, typical** days | Calm markets |

> Think of it as a **camera exposure setting**. Turn it one way and the bright ordinary scene is properly exposed. Turn it the other and you can finally see detail in the dark corners. The rare crash days *are* the dark corners — and they're exactly what a risk manager cares about.

**Visual:** Camera exposure metaphor — same photo at three exposures, labeled α<1, α=1, α>1.

**Speaker notes:** Emphasize: **α = 1 recovers the existing method exactly.** My model doesn't replace the baseline; it *contains* it as a special case. That's what makes the comparison clean — and it's also a statistical complication I'll flag later (nested models).

---

## Slide 16 — The Hurst exponent: a calm-or-turbulent detector

**Title:** How Do We Know a Crisis Is Happening, Without Being Told?
**Body:**
The **Hurst exponent (H)** is one number describing a market's character:

| Value | Name | Day to day |
|---|---|---|
| **H > 0.5** | persistent | Trends continue. Up days follow up days. Calm, orderly. |
| **H ≈ 0.5** | random walk | Yesterday says nothing about today. |
| **H < 0.5** | anti-persistent | Violent reversals. Up 3%, down 4%, up 2%. **Panic.** |

- Computed on a rolling basis as an automatic regime switch.
- When H drops below 0.5, the model dials α toward the tail-sensitive setting.
- That is the **"Hurst-Regime-Adaptive"** in the title.

**Visual:** `preliminary_results/F1_sample_and_regimes.png` — shaded turbulent bands lining up with elevated volatility.

**Speaker notes:** The detector is detecting something real — the shaded bands line up with the periods when volatility across all ten countries was genuinely elevated. **But flag the difficulty now, before the panel finds it:** turbulent days are a small minority of the sample. The crisis periods this thesis cares most about are also the periods with the least data to learn from. That is the central difficulty of the whole exercise, and it comes back to bite in the preliminary results.

---

## Slide 17 — Research Question 1, stated

**Title:** Research Question 1
**Body:**
> **Does replacing Shannon-based Effective Transfer Entropy with Effective *Rényi* Transfer Entropy as the graph-construction metric produce statistically significant forecasting gains?**

In plain English:

> **If I build the map of "who influences whom" using a method that pays extra attention to crash days, do I get better forecasts?**

- Everything else — the neural network, the data, the training, the seeds — is held **identical**.
- The *only* thing that changes is how the map is drawn.

**Speaker notes:** Stress the controlled-experiment framing. This is one knob, turned, with everything else pinned down. That's why the comparison is meaningful.

---

## Slide 18 — Before H1a: demystifying QLIKE `[KEY CONCEPT]`

**Title:** First — How Do We Score a Forecast? Meet QLIKE
**Body:**
Every hypothesis below rests on a **loss function**: a number saying how badly a forecast missed. Lower = better.

**The obvious choice, squared error, has a problem here.** It punishes a miss of 0.5 the same whether true volatility was 1.0 or 20.0. But missing by 0.5 on a calm day is a disaster; missing by 0.5 in a crisis is a rounding error.

**QLIKE fixes this by scoring the *ratio*, not the difference:**

$$\text{QLIKE} = \frac{RV}{\widehat{RV}} - \log\!\left(\frac{RV}{\widehat{RV}}\right) - 1$$

| If your forecast is... | QLIKE says |
|---|---|
| Exactly right (ratio = 1) | **0** — a perfect score |
| Too low (you missed an incoming storm) | **Heavily punished** |
| Too high (you were over-cautious) | Punished, but **more gently** |

**Two reasons it's the right ruler here:**
1. **It's asymmetric on purpose.** Under-forecasting risk is the mistake that blows up institutions. QLIKE punishes it harder.
2. **It's proxy-robust** (Patton, 2011). We never observe "true" volatility — only a noisy 20-day estimate of it. QLIKE ranks models correctly *despite* that noise. Squared error and MAPE do not.

**Speaker notes:** Deliver the analogy out loud: "QLIKE is a grading scheme for a weather forecaster that punishes 'sunny, no umbrella needed' before a hurricane much harder than 'bring a coat' on a nice day." That's the whole intuition. Then the formula is just bookkeeping.

**Honest caveat to state on this slide if asked:** QLIKE is undefined for forecasts ≤ 0, and a neural net trained under squared error can emit negative volatility. The final run handles this explicitly (softplus output / log-RV target), and I report how many forecasts needed intervention. Also: I *train* under squared error but *evaluate* under QLIKE. That's a deliberate choice — QLIKE is proxy-robust for scoring, squared error is what the architecture inherits from Lee & Cho — and it's justified in §3.8.

---

## Slide 19 — The other two words: Giacomini–White and "loss differential"

**Title:** Two More Terms, Then No More Jargon
**Body:**

**Loss differential (d_t):**
> On each test day, score both models with QLIKE, then subtract:
> **d_t = (baseline's loss) − (my model's loss)**
> **d_t > 0 means my model lost less — my model won that day.**
> Computed as the average across the ten countries, once per day.

**Giacomini–White test:**
> "Is the average of those daily wins reliably above zero — or just a lucky streak?"
> It's a t-test with two adjustments: **HAC standard errors** (because a good day tends to be followed by another good day, so the days aren't independent), and it stays valid when one model *contains* the other as a special case — which is exactly our situation at α = 1.

**Speaker notes:** If a panelist asks why not plain Diebold–Mariano: because α = 1 makes the baseline a limiting special case of my model — the models are **nested** — and standard DM asymptotics assume they aren't. I report DM anyway, as a familiar descriptive robustness check with the nesting caveat noted. Clark–West is the standard nested alternative if pushed.

---

## Slide 20 — Hypothesis 1a

**Title:** H1a — "Does It Work At All?"
**Body:**

| | |
|---|---|
| **H₀** | Across the test period, my model and the baseline are equally accurate. Any difference is sampling noise. *(average d_t = 0)* |
| **H₁** | My model has strictly lower expected loss. *(average d_t > 0)* |

**Test:** Giacomini–White, HAC-robust, one-sided. Reported alongside HLN-corrected Diebold–Mariano, plus RMSE and MAE as secondary losses.

> **Why this hypothesis exists:** it's the basic sanity check. Take every test day, see how badly each model missed, subtract, and ask whether the average difference is reliably above zero. It answers *"is it better on average?"* — **nothing about when, and nothing about why.**

**Speaker notes:** Be honest that H1a alone is a weak claim. A model that is uniformly 1% better everywhere would pass H1a and still not support the thesis's actual argument. That's what H1b is for — and that's the natural setup for the next slide.

---

## Slide 21 — Hypothesis 1b `[THE CORE CLAIM]`

**Title:** H1b — The Hypothesis the Thesis Actually Rests On
**Body:**

| | |
|---|---|
| **H₀** | Whatever advantage exists, it is **the same** in turbulent and calm markets. The regime indicator carries no information about when the Rényi model wins. *(β₁ = 0)* |
| **H₁** | The advantage is **concentrated in turbulence** — the gain is systematically larger on days following an anti-persistent (H < 0.5) signal. *(β₁ > 0)* |

**The regression:**
$$d_t = \beta_0 + \beta_1 \cdot \text{Turbulent}_{t-1} + \varepsilon_t$$
- **β₀** = my advantage in calm markets
- **β₁** = the *extra* advantage in turbulence ← **the number the thesis rests on**

**Why this is the real claim:** α < 1 up-weights the tails of the return distribution. Tails only matter when returns are non-Gaussian. That's what crises *are*. So the mechanism I claim predicts something specific and falsifiable: **not "is the gain positive" but "is the gain bigger when the market is turbulent."**

**Speaker notes:** Say the failure mode out loud — it's the most credible thing you can do at a midterm:
> "If H1a passes and H1b fails, I have a model that works but **not for the reason I claimed**, and the honest write-up says exactly that."
The ideal outcome is β₀ ≈ 0 with β₁ > 0: no cost in calm markets, real gain in crises. That's precisely the "improve robustness under extreme regimes while preserving calm-period performance" objective.

---

## Slide 21b — `[KEY CONCEPT]` H1b worked end to end

**Title:** Where the H1b Numbers Come From
**Body:**
Four steps, on six real test days: **score → subtract → sort → decide.**

**Speaker notes:** This slide exists because "β₁ is the regime coefficient in a Giacomini–White instrument regression" means nothing to a non-specialist, and they are entitled to see where the number came from. Walk it left to right and do not rush:

1. **Score.** Every forecast gets a QLIKE score. Say "golf score — zero is perfect, higher is worse" before you say QLIKE. Point at 2023-09-29: Shannon forecast 0.886 against a true 0.830 and scored 0.00212; my model forecast 0.843 and scored 0.00012.
2. **Subtract.** Average the ten markets, then take the baseline's loss minus mine. That difference is the *margin*. Positive margin = I won that day. Say the sign rule out loud — "a lower loss for me makes the margin larger" — because this is where people get turned around.
3. **Sort.** Put each day in the calm pile or the turbulent pile and average each. The calm average is β₀. The turbulent average is β₀+β₁. **β₁ is just the difference between the two averages** — the regression is bookkeeping, not magic.
4. **Decide.** β₁ > 0 means the margin widens in turbulence, which is exactly what tail-sensitivity is supposed to buy.

**Be explicit that panel 1–3 numbers are an illustration.** The dates, the RV targets and the regime flags are real; the two forecast columns are invented because the pilot did not store per-day forecasts. Say that before a panelist asks — then point at the bottom strip, which is the real T11 output.

The bottom strip is the honest ending: β̂₁ = −0.002039 with a one-sided p of 0.978, computed from four days whose Hurst values were 0.4999, 0.4970, 0.4957 and 0.4987. That last detail is the one to land on. Those are not crises; one of them cleared the threshold by a ten-thousandth. The test did not fail — it never ran.

---

## Slide 22 — How the experiment is run

**Title:** The Experimental Setup
**Body:**

| | |
|---|---|
| **Data** | 10 country ETFs, daily. Pilot: 2018–2024. Final: from 2003. |
| **Split** | Train 942 days (to 2021-11-24) → Validate 314 → **Test 314 (2023-02-28 to 2024-05-28)** |
| **Controlled** | Identical architecture, seeds, epochs across arms. Only the graph changes. |
| **No leakage** | The graph is estimated on **training data only**, ending 461 calendar days before the test block opens. |
| **Arms compared** | HR-ETE-GNN (α=0.5), HR-ETE-GNN (α=1.5), H-ETE-GNN (Shannon, α=1.0), random-graph control, no-graph ablation, HAR-RV, random walk |

**Speaker notes:** The two controls are worth naming out loud because a good panelist will ask for them: a **random graph** (same number of edges, arbitrary structure) and a **no-graph ablation** (each country alone). Together they answer "is the graph doing anything at all, or is any graph as good as your graph?"

---

## Slide 23 — `[RESULT]` H1a preliminary result

**Title:** H1a — Preliminary Result: **Not Supported**
**Body:**

| Quantity | Value |
|---|---|
| Test days (T) | 314 |
| Mean loss differential (d̄) | **−0.000699** |
| Relative change vs. baseline | **−5.14%** *(a deterioration)* |
| HAC standard error | 0.000363 |
| **GW t-statistic** | **−1.928** |
| One-sided p-value | **0.973** |
| Decision at 5% | **Fail to reject H₀** |

**Mean QLIKE across all arms:**

| Model | QLIKE | vs. baseline |
|---|---|---|
| Random walk | 0.00601 | +55.8% |
| HAR-RV (Corsi 2009) | 0.00619 | +54.5% |
| Random graph | 0.01260 | +7.3% |
| No-graph ablation | 0.01351 | +0.6% |
| **H-ETE-GNN (Shannon, α=1.0)** | **0.01360** | — |
| **HR-ETE-GNN (α=0.5)** | **0.01430** | **−5.1%** |
| HR-ETE-GNN (α=1.5) | 0.01791 | −31.7% |

**Speaker notes:** Own this directly, in the first ten seconds: **in the pilot, the Rényi graph did not beat the Shannon graph.** Then make the three points that matter, in this order:
1. The sign is negative and the t-stat is −1.93 — this isn't "inconclusive", it's pointing the wrong way in this configuration.
2. **The simple baselines beat every GNN arm.** HAR-RV and the random walk win by >50%. That's a real finding about the pilot configuration, not a bug I'm hiding.
3. **The random graph beat the measured graph.** That's the most diagnostic number on the slide — it says the pilot's graph estimation is not yet adding signal. With 100 surrogates and 3 seeds, that is exactly what an under-powered pipeline looks like.

---

## Slide 24 — `[RESULT]` H1b preliminary result

**Title:** H1b — Preliminary Result: **Not Evaluable**
**Body:**

| Instrument | Turbulent days in test block | β̂₁ | t | one-sided p |
|---|---|---|---|---|
| Lagged Hurst (H < 0.5, URTH) | **4 of 314 (1.3%)** | −0.00204 | −2.02 | 0.978 |
| Relative high-vol (>80th pct) | **11 of 314 (3.5%)** | −0.00062 | −0.69 | 0.754 |
| Absolute high-vol (train-fixed) | **0 of 314 (0.0%)** | — | — | — |

> **The honest statement:** the test block (Feb 2023 – May 2024) was a calm market. With **4 turbulent days**, H1b has essentially no statistical power. This is **not evidence against the hypothesis** — it is an absence of evidence either way.

**Power audit:** minimum detectable effect at 80% power = **0.140** standardized. Anything smaller is invisible regardless of whether it's true.

**Speaker notes:** This is the slide where honesty is the whole argument. Say it plainly: "I cannot test my central claim on this sample, because my sample contains almost no crisis." Then immediately pivot to the fix — it's concrete and it's the natural next question:
- The final run starts from **2003**, not 2018. That pulls in the **2008 Global Financial Crisis, 2020 COVID, and 2022** — the sample already shows 130 turbulent days across 1,361, so the base rate exists; the *test block* just didn't happen to contain them.
- I will also run H1b a second way with an explicit crisis-window dummy, so the result doesn't hinge on the "H < 0.5 = turbulent" mapping alone.

---

## Slide 25 — Why the pilot is not the thesis

**Title:** The Pilot Ran at 1/180th of the Final Configuration
**Body:**

| | Pilot (these results) | Thesis run |
|---|---|---|
| Seeds | 3 | **20** |
| Epochs | 200 *(hit the cap — not converged)* | **3,000** |
| Surrogates per pair | 100 | **1,000** |
| α grid | 0.5, 1.0, 1.5 | 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5 |
| Sample start | 2018 | **2003** |
| Wall clock | 5 minutes | **49–74 hours** |

- **Every arm hit the 200-epoch cap** — none of them finished training.
- Training work ratio: **180×**. ER-TE work ratio: **23×**.

> The pilot's job was to prove the **pipeline runs end to end and produces the right table shapes** — not to produce the thesis's answer. It did that.

**Speaker notes:** This is the defensive slide, so deliver it as fact rather than excuse. The single most persuasive line: "every arm hit the epoch cap, which means not one of these models finished training. Comparing two half-trained models tells you about the training budget, not about the method." Also note the 100-surrogate p-value floor of 1/101 ≈ 0.0099 — visible in the H2a table as "pairs at the p-floor", and directly limiting how many edges can survive FDR correction.

---

# ACT 3 — Research Question 2

## Slide 26 — Research Question 2, stated

**Title:** Research Question 2 — A Genuinely Different Question
**Body:**
> **When the network is built from Rényi ETE versus Shannon ETE on the same crisis and non-crisis windows, does the resulting topology differ — do the top volatility-information transmitters change identity under tail-sensitive (α < 1) measurement during crisis periods?**

In plain English:

> **Forget forecasting for a moment. Do the two measurement tools draw a *different map of the world*?**

> **Why this matters independently:** if Shannon says Germany is the dominant crisis transmitter and Rényi says Brazil is, **that is a finding about measurement itself** — true regardless of which model forecasts better. RQ2 is a structural contribution, not a predictive one.

**Speaker notes:** Make the independence explicit, because it's the strongest card you hold at this midterm: "RQ1 asks *does it predict better*. RQ2 asks *does it see differently*. RQ2 can produce a real contribution even in a world where RQ1's answer is no." Given the RQ1 preliminary results, this framing matters.

---

## Slide 27 — Hypothesis 2a

**Title:** H2a — Is This Arrow Even Real?
**Body:**

| | |
|---|---|
| **H₀** | Country *i*'s past carries **no** information about country *j*'s future beyond what *j*'s own past already provides. The measured flow is indistinguishable from noise. |
| **H₁** | A genuine directional flow exists from *i* to *j*. |

**The problem this solves:** transfer entropy is always ≥ 0 by construction, and a finite-sample estimate is **essentially never exactly zero**. A raw ER-TE number tells you nothing on its own.

**The solution — circular-shift surrogates:**
1. Take Germany's series and **rotate it in time** — the rotation destroys the timing relationship with Brazil while preserving Germany's own internal rhythm.
2. Recompute the flow. Repeat 100× (pilot) / 1,000× (final).
3. That gives a null distribution: *"what does this number look like when there's genuinely no flow?"*
4. **p-value = the fraction of surrogates that beat the real value.**

**Multiple testing:** 90 ordered pairs tested at once → Benjamini–Hochberg FDR control at q = 0.10. Run separately for each α and for crisis vs. non-crisis windows.

**Speaker notes:** The rotation analogy that lands: "imagine shifting one country's calendar by six months. Germany still behaves like Germany — same rhythms, same internal streakiness — but any real timing link to Brazil is now scrambled. Do that a thousand times and you learn what pure coincidence looks like." Then: with 90 simultaneous tests at 5%, you'd expect ~4.5 false positives by chance. BH controls the expected proportion of your significant findings that are false, at 10%.

---

## Slide 28 — `[RESULT]` H2a preliminary result

**Title:** H2a — Preliminary Result: **The Two Measures See Different Amounts**
**Body:**

| α | Interpretation | Edges surviving BH-FDR | Density | Mean \|ER-TE\| |
|---|---|---|---|---|
| **0.5** | **Rényi, tail-emphasizing** | **36** | **40.0%** | 0.0377 |
| 1.0 | **Shannon — the baseline** | **0** | 0.0% | 0.0267 |
| 1.5 | Rényi, bulk-emphasizing | 10 | 11.1% | 0.0481 |

> **On the same data, same windows, same test:** the tail-sensitive setting finds **36 significant directed links**. The Shannon baseline finds **zero**.

**Speaker notes:** This is the strongest preliminary result in the deck. Lead with it after the RQ1 slides — the arc is "forecasting didn't move in the pilot, but the *measurement* result is clean and it is exactly the direction the mechanism predicts."

Then pre-empt the two objections a sharp panelist will raise:
1. *"36 of 90 is a 40% density — isn't that implausibly many?"* Yes, possibly. With only 100 surrogates the p-value floor is 0.0099 and 21 pairs sit *at* that floor, which inflates the count. 1,000 surrogates in the final run resolves this in both directions.
2. *"Zero edges for Shannon looks broken."* It's consistent — Shannon's mean flow magnitude is the lowest of the three (0.0267), so it's the arm most likely to be wiped out by FDR correction on a short sample. It's a power problem, not a bug.

---

## Slide 29 — Hypothesis 2b

**Title:** H2b — Does the *Cast of Characters* Change?
**Body:**

| | |
|---|---|
| **H₀** | During crisis windows, the set of dominant transmitters — the top-ranked source countries by significant out-degree — is **the same** under Shannon and under tail-sensitive Rényi. |
| **H₁** | **At least one country's status as a dominant transmitter changes** between the two networks. The edge sets and out-degree rankings diverge. |

**"Out-degree" in one line:** how many arrows point *out* of a country — i.e. how many other markets it demonstrably leads. High out-degree = a **transmitter** of nervousness. Low out-degree, high in-degree = a **receiver**.

**How it's tested formally** (upgraded from descriptive, at panel-anticipation):
1. **Hypergeometric test** on the top-K edge overlap — is the agreement between the two networks more or less than chance?
2. **Kendall's τ** rank concordance between the two out-degree rankings, with a stationary-bootstrap 95% CI.

**Speaker notes:** Be upfront that this started as a descriptive comparison — edge-overlap tables and heatmaps with no p-value — and that I formalized it precisely so it carries the same evidentiary weight as H2a. Saying "I noticed this weakness and fixed it" is worth more at a midterm than pretending it was never there.

---

## Slide 30 — `[RESULT]` H2b preliminary result

**Title:** H2b — Preliminary Result: **Mixed, and Interesting**
**Body:**

**Out-degree by country, per α:**

| Country | α=0.5 (Rényi) | α=1.0 (Shannon) | Rank under Shannon | Rank under Rényi | **Rank change** |
|---|---|---|---|---|---|
| Germany | 5 | 6 | 1 | 1 | 0 |
| **Australia** | 2 | 0 | 9 | 4 | **+5** |
| **Canada** | 3 | 1 | 5 | 2 | **+3** |
| UK | 3 | 3 | 3 | 2 | +1 |
| South Korea | 2 | 1 | 5 | 4 | +1 |
| Brazil | 0 | 0 | 9 | 10 | −1 |
| Japan | 1 | 1 | 5 | 7 | −2 |
| Taiwan | 2 | 4 | 2 | 4 | −2 |
| Mexico | 1 | 1 | 5 | 7 | −2 |
| **South Africa** | 1 | 3 | 3 | 7 | **−4** |

**The formal tests:**

| Quantity | Value | Reading |
|---|---|---|
| Top-K overlap (K=20) | **12 of 20** | Edges both α agree on |
| Expected if independent | 4.44 | |
| Hypergeometric p | **< 0.0001** | The two α agree **more** than chance |
| Kendall τ (out-degree ranks) | **0.453** | 1 = identical ranking |
| τ bootstrap 95% CI | **[0.270, 0.919]** | Does not contain 0 |

**Speaker notes:** Give the honest two-sided reading — this is a "yes, and" result:
- **Not a different world.** The two networks overlap far more than chance (12 vs 4.4 expected), and Germany is the #1 transmitter under both. The Rényi network is not noise, and it isn't a wholesale replacement for the Shannon picture.
- **But not the same world either.** τ = 0.453 is a moderate correlation, not agreement — roughly half the ranking information is *not* shared. And individual countries move a lot: Australia jumps 5 places under tail-sensitive measurement, South Africa drops 4.
- **So the preliminary read on H2b:** directionally supportive — the cast of characters does shift — but the pilot cannot yet distinguish "a real reordering during crises" from "sampling noise on a short, calm sample." The final run adds crisis vs. non-crisis window splits, which is where the hypothesis is actually pointed.

**Visual:** `preliminary_results/F2_adjacency_heatmaps.png` (side-by-side α=0.5 vs α=1.0 heatmaps) and `preliminary_results/F3_transmitters.png`.

---

# ACT 4 — Closing

## Slide 31 — Where things stand

**Title:** Honest Scorecard
**Body:**

| Hypothesis | Plain-English claim | Preliminary verdict |
|---|---|---|
| **H1a** | The Rényi graph forecasts better on average | ❌ **Not supported** — −5.1%, t = −1.93 |
| **H1b** | The gain concentrates in turbulent markets | ⚠️ **Not evaluable** — only 4 turbulent days in the test block |
| **H2a** | The measured links are real, not noise | ✅ **Supported** — 36 edges at α=0.5 vs **0** at α=1.0 |
| **H2b** | The cast of top transmitters changes | ⚠️ **Mixed** — τ = 0.45; big individual moves; overlap beats chance |

> **Overall:** the pipeline runs end to end and produces every table the thesis needs. The **structural** result (RQ2) is the strongest signal so far. The **forecasting** result (RQ1) cannot be judged from a 5-minute, 3-seed, 200-epoch pilot on a calm test period.

**Speaker notes:** Do not oversell. The panel will trust the final numbers precisely in proportion to how straight you are about these ones. If asked directly "so is your thesis working?" — the answer is: "the measurement half is showing exactly what I predicted. The forecasting half is untested, because the pilot was too small and the test window had no crisis in it. Both are fixable, and I know how."

---

## Slide 32 — What changes before the final run

**Title:** The Fix List
**Body:**

**Scale:**
1. Sample from **2003** → captures 2008, 2020, 2022 crises (the pilot's 2018–2024 test block had 4 turbulent days)
2. **20 seeds, 3,000 epochs** → every pilot arm hit the 200-epoch cap without converging
3. **1,000 surrogates** → removes the p = 0.0099 floor that is distorting the H2a edge counts

**Methodology hardening:**
4. Make **QLIKE the stated primary metric in §3.8** — currently the manuscript lists RMSE/MAE/MAPE and never names QLIKE. *(Internal consistency — the single most likely thing to get caught.)*
5. **Guarantee positive forecasts** (softplus output or log-RV target) so QLIKE is always defined; report any interventions.
6. **Confirm a fixed-length rolling window**, which Giacomini–White requires; demote DM to descriptive with the nesting caveat stated.
7. **Select α on the validation split only** — never on test — or report all α as separate models and let the MCS handle the multiplicity.
8. **Run H1b twice**: Hurst-based regime *and* an explicit crisis-window dummy. Agreement is a strong result; disagreement is something I need to know before the defense, not during it.

**Speaker notes:** Item 4 is the one to raise yourself rather than let the panel find. Items 6 and 7 are the ones a methodologically sharp panelist will go for — having the answer ready signals you know where the statistical soft spots are.

---

## Slide 33 — Close

**Title:** Thank You
**Body:**
> **RQ1:** Does paying extra attention to crash days produce better forecasts? — *Pilot says not yet; the real test hasn't been run.*
>
> **RQ2:** Does paying extra attention to crash days produce a different map of the world? — *Pilot says yes: 36 significant links vs. 0.*

**Questions.**

**Speaker notes:** Close on RQ2. It's the result you can defend hardest, and it's the one that's independent of whether the forecasting half comes good.

---

# Appendix Slides (hold in reserve — show only if asked)

## Slide A1 — H1c, per-ETF breakdown

**Title:** Appendix — Is the Result Driven by One Country?
**Body:** Diebold–Mariano per ETF, Benjamini–Hochberg at q = 0.10. **0 of 10 significant after correction.** Positive mean differential for 3 of 10 (Japan +3.4%, UK +0.9%, Germany +0.5%); worst are Brazil (−36.7%) and South Africa (−16.2%). No single market is carrying the pooled result in either direction.

## Slide A2 — H1d, Model Confidence Set

**Title:** Appendix — The Leaderboard
**Body:** The 90% Model Confidence Set contains **1 of 7** candidates: the **random walk**. Elimination order: HR-ETE-GNN(α=0.5) → no-graph → random graph → HR-ETE-GNN(α=1.5) → H-ETE-GNN → HAR-RV. In the pilot configuration every neural arm is eliminated. Expected for half-trained models; the informative version of this table comes from the full run.

## Slide A3 — Power audit

**Title:** Appendix — What Could This Sample Even Detect?
**Body:** T = 314. Minimum detectable standardized effect at 5% significance = **0.093**; at 80% power = **0.140**. Turbulent days: 130/1,361 whole sample (9.6%), but only **4/314 in the test block (1.3%)**. A null H1b result on this sample is uninformative rather than evidence of no effect.

## Slide A4 — Seed stability

**Title:** Appendix — Are These Numbers Stable Across Seeds?
**Body:** Per-seed RMSE standard deviation: α=0.5 → 0.0040; Shannon → 0.0062; α=1.5 → 0.0143. All arms trained with identical seed lists over 3 seeds, evaluated as seed ensembles. **All arms hit the 200-epoch cap** — none converged.

## Slide A5 — Runtime

**Title:** Appendix — Why the Pilot Was Small
**Body:** Pilot wall clock: **5 minutes**. Thesis run, serial estimate: **49–74 hours**, dominated by training. Training work ratio 180×; ER-TE work ratio 23×; sample-length factor 3.3×. The α arms parallelize perfectly.

## Slide A6 — Data & reproducibility

**Title:** Appendix — Data and Splits
**Body:** 10 country ETFs (EWA, EWC, EWG, EWJ, EWT, EWU, EWW, EWY, EWZ, EZA). Train 942 samples (2018-03-02 → 2021-11-24); validation 314 (2021-11-26 → 2023-02-27); test 314 (2023-02-28 → 2024-05-28). Adjacency estimated on training data only, ending **461 calendar days** before the test block opens. Identical seed lists across all arms.
