# HR-ETE-GNN — Presentation Script

Companion to the slide deck: https://claude.ai/artifact/BUuvNxVRxSimTs8GPK2o4h

30 slides · about 3,571 spoken words · roughly 26 minutes at 140 words per minute.
The same text is in each slide's speaker notes, so Present mode shows it too.

**How the two fit together.** The slides carry the pictures, the numbers and at most one line of words.
Everything that explains a picture is here. Every data slide uses the preliminary run: the 10 ETFs from
January 2018 to May 2024, at reduced scope (3 seeds, 100 surrogates, 200 epochs). The forecasting numbers are
the ones in Chapter IV (Tables 4.1.1–4.1.3). Say once, on slide 2, that this run shows the format, not the
finding, and read the prep notes at the end before presenting slides 22–25.

---

## 1. HR-ETE-GNN

*On screen:* Title, name, adviser.  ·  *about 34 s*

Good morning. My thesis is called HR-ETE-GNN. In one sentence: I take an existing volatility-forecasting model, change exactly one ingredient, which is how the network of influence between stock markets is measured, and test whether the forecasts get better. I'll show how both models are built, how they turn a network into a forecast, how I score the forecasts, and how I decide whether a difference is real rather than luck. Every chart uses the data from my preliminary run.


## 2. Volatility is how much a price swings

*On screen:* Germany's ETF in a calm quarter (Q4 2019) vs the COVID crash (Feb–May 2020): price and daily moves on shared scales; typical daily move for each.  ·  *about 90 s*

Before the model, one idea: volatility. Volatility measures how much a price swings from day to day, and it doesn't care about direction. A market that jumps three percent and a market that falls three percent are equally volatile. Both panels show the same thing, Germany's stock-market ETF, on the same scales. On the left is a calm quarter at the end of 2019: the price barely moves and the daily moves are tiny, about 0.7 percent on a typical day. On the right is February to May 2020, the COVID crash: the price falls by roughly a third and then recovers, and the daily moves are huge, about 3.9 percent on a typical day, around 5 times the calm level. That typical daily move is what we call realized volatility: the root-mean-square of recent daily returns, which is the standard deviation of returns measured from data. Why does it matter? Volatility is the standard measure of risk. It sets how much capital a bank must hold, what an option costs, and how a portfolio is hedged. And notice that the stormy days come in a cluster: big moves follow big moves. That clustering is what makes tomorrow's volatility forecastable at all, and forecasting it is what this thesis is about.


## 3. Forecasting tomorrow’s volatility

*On screen:* 20-day RV of the 10 ETFs, 2018–2024, with their average; COVID-19 and Russia–Ukraine windows shaded.  ·  *about 59 s*

Volatility is how much a market swings, up or down. Risk managers, option traders and portfolio managers all need tomorrow's volatility, not yesterday's. This is 20-day realized volatility for the ten country ETFs in my preliminary run, from 2018 to May 2024. Each grey line is one country and the black line is their average. The shaded spans are two of the crisis windows the thesis defines: COVID-19 and the Russia–Ukraine episode. Look at March 2020: every country's volatility jumps at the same moment. That co-movement is exactly why we model markets as a network instead of one at a time. One caveat for the whole talk: this preliminary run is a reduced-scope version of the pipeline, with three training seeds and a hundred surrogates, so it shows the format of the results, not the final findings.


## 4. Two research questions

*On screen:* RQ1 and RQ2 cards; thesis vs preliminary data scope.  ·  *about 48 s*

There are two research questions, and they're answered differently. RQ1 is the forecasting question: if I replace the Shannon-based graph with a Rényi-based graph, is the expected forecast loss, measured with QLIKE, lower than the baseline, H-ETE-GNN? That one gets a formal, one-sided hypothesis test. RQ2 is structural: when I measure the network with a tail-sensitive lens, do different countries come out as the main transmitters of volatility, especially during crises? That one is descriptive; I report what the two measures see and make no significance claim. The full thesis uses ten country ETFs from 2003 to 2024. The preliminary run you'll see uses the same ten ETFs from 2018 onward.


## 5. One dial: α

*On screen:* Dark statement slide: α = 0.5 (HR-ETE-GNN) vs α = 1 (H-ETE-GNN).  ·  *about 50 s*

Here is the key idea of the whole design. The two models are the same model with one dial set differently. The dial is alpha, the order of the Rényi entropy used to draw the network. At alpha equal to one, Rényi entropy becomes ordinary Shannon entropy, and the model is H-ETE-GNN, the Lee and Cho baseline. At alpha equal to one half, which I fixed in advance, the measure gives more weight to rare, extreme days, and that's HR-ETE-GNN. Everything else is held identical: the estimator, the surrogates, the number of edges, the architecture, the training, even the random seeds. So if the forecasts differ, the only possible reason is the entropy that drew the graph.


## 6. The pipeline

*On screen:* Six-box pipeline (RQ1) with the RQ2 branch under the ER-TE graph.  ·  *about 49 s*

This is the whole pipeline on one slide. The top row, left to right, is RQ1. Daily prices become returns and then realized volatility, the thing we forecast. From the returns I build a directed graph of which country influences which; this orange box is the only place where the two models differ. The graph feeds a graph neural network that outputs one volatility forecast per country per day. Each forecast is scored with QLIKE, and the scores go into a Giacomini–White test. RQ2 branches off right after the graph is built: I compare the network structure directly, and that branch never touches the forecasts. I'll now walk through each box using the preliminary run.


## 7. From prices to realized volatility

*On screen:* Return and RV formulas; Germany's (EWG) returns and RV.  ·  *about 46 s*

Step one is the data. From each day's closing price I take the log-return, which is roughly the percentage move from yesterday. The top chart is Germany's daily returns from 2018 to 2024: small most days, very large in March 2020. Then I compute realized volatility: square the last twenty returns, average them, take the square root. That's the bottom chart. It measures how big the moves have been lately and ignores their direction, so a crash and a jump of the same size count the same. Germany's realized volatility peaks at about 6.1 percent a day in 2020. The forecasting target is the next day's realized volatility.


## 8. Measuring influence with transfer entropy

*On screen:* UK today and Germany today pointing to UK tomorrow; TE formula.  ·  *about 59 s*

To draw arrows between countries I need a number that says whether one country carries information about another. Transfer entropy asks: if I already know the UK's own recent history, does also knowing Germany's today make the UK's tomorrow less surprising? The grey arrow is what we already know. The orange arrow is the extra information Germany might add. Formally, it's the uncertainty about the UK's tomorrow given the UK's today, minus the same uncertainty once Germany's today is also known. If Germany helps, the second term is smaller and transfer entropy is positive. Two properties matter: it has a direction, so Germany to UK can differ from UK to Germany, and it catches non-linear links a correlation would miss. The H in this formula is an entropy, and choosing which entropy is exactly the dial alpha.


## 9. Shannon versus Rényi weighting

*On screen:* Weight on each size of move under Shannon vs Rényi, from the 10 ETFs' training-block returns.  ·  *about 58 s*

Why would a different entropy matter? Entropy is built from the probabilities of outcomes. These two charts use the actual daily returns of all ten ETFs in the training period and show how much weight each size of move gets inside the entropy. On the left is Shannon: every outcome counts in proportion to how often it happens, so ordinary days dominate, and moves bigger than three percent carry only 4.5 percent of the weight. On the right is Rényi with alpha one half. Raising each probability to the power one half flattens the distribution, and the extreme moves, in orange, now carry 20.7 percent. Crashes are exactly these rare days. Lee and Cho found their model weakened in crises, and a lens that looks harder at the tails might see crisis-time influence that Shannon dilutes.


## 10. Removing chance with surrogates

*On screen:* 100 surrogates for the strongest pair, the real value, the bias; ER-TE and p.  ·  *about 67 s*

With a finite sample, the estimator produces a non-zero value even between two unrelated series. The fix is surrogates. This is the strongest pair in the preliminary run, Taiwan to South Korea, under alpha one half. I rotate Taiwan's series in time by a random offset, which keeps its own rhythm but breaks its alignment with South Korea, and recompute. I do that a hundred times; that's the grey histogram. Its average, the dashed line, is the bias: what chance alone produces. The orange line is the real value, -0.046. It sits to the right of every one of the hundred surrogates. Subtracting the bias gives the effective Rényi transfer entropy, 0.102. The p-value is the share of surrogates at or above the real value, counting the real one too, so with a hundred surrogates the smallest possible p is one over a hundred and one, 0.010. That floor matters: the full run uses a thousand surrogates.


## 11. The adjacency matrix · α = 0.5

*On screen:* 10×10 ER-TE matrix at α = 0.5; the 20 kept edges outlined.  ·  *about 48 s*

Repeating this for all ninety ordered pairs of countries gives the adjacency matrix. Rows are the country sending information, columns the country receiving it, and each cell is its effective Rényi transfer entropy at alpha one half. Darker orange means stronger. The outlined cells are the edges the model keeps: the twenty strongest. Keeping exactly twenty edges for every alpha means neither model gets a denser graph, so any difference comes from which edges and how they're weighted. Notice the Germany row: it has the most outlined cells. All of this is estimated on the training block only, 2018 to late 2021, so the graph never sees the days it's later tested on.


## 12. The adjacency matrix · α = 1

*On screen:* 10×10 ER-TE matrix at α = 1; the 20 kept edges outlined.  ·  *about 46 s*

And this is the same matrix for the baseline, Shannon, alpha equal to one. Same twenty-edge budget, outlined again. Many of the same cells are kept, twelve of the twenty in fact, but not all, and the values are on a different scale from the Rényi matrix. Germany's row is again the strongest. The next two slides draw both matrices as graphs. One number worth knowing: when I instead keep only edges that pass a significance test with a false-discovery correction, thirty-six Rényi edges survive but zero Shannon edges do, at a hundred surrogates. That's why the forecasts use the matched twenty-edge graphs rather than the significance-filtered ones.


## 13. The graph · α = 0.5

*On screen:* The 20 kept Rényi edges as a network with values; top transmitters; edges passing BH.  ·  *about 27 s*

Here are the twenty kept Rényi edges drawn as a graph. Each circle is a country, each arrow a kept link, and the number is its effective transfer entropy; thicker means stronger. "Out" counts each country's outgoing arrows, which RQ2 uses. Germany is the hub: it sends 5 arrows, followed by Canada (tied), UK (tied). That matches Table 4.1.3 in the manuscript.


## 14. The graph · α = 1

*On screen:* The 20 kept Shannon edges as a network with values; top transmitters; edges passing BH.  ·  *about 38 s*

And the Shannon graph for the baseline. Germany is again the hub with 6 outgoing arrows, but the next transmitters change: Taiwan, South Africa (tied), UK (tied). Taiwan, for instance, sends 4 arrows under Shannon but 2 under Rényi, while Canada goes the other way. Also compare the sizes on the arrows with the previous slide: Rényi and Shannon values sit on different scales. If I fed raw values to the network, a larger scale alone would make the graph speak louder. That's why the next step normalises them.


## 15. Same budget, different weights

*On screen:* Normalised incoming weights for the two busiest receivers; HR vs H component table.  ·  *about 48 s*

Row normalisation rescales each country's incoming weights to sum to one, which puts both models on the same scale. On the left are the two countries that listen to the most senders, Australia and UK. A dash means that arrow isn't in that model's graph. So the two models differ in two ways: which arrows exist, and how the weight is split among them. On the right is the full comparison: the edge measure is the only row that differs, which is what makes this a clean ablation. And since alpha equal to one is a special case of the Rényi model, the two models are nested, which matters for the test later.


## 16. Hurst regime detection

*On screen:* Hurst exponent of MSCI World (URTH) with regime bands and flips; the three-part rule.  ·  *about 55 s*

The H in both model names is the Hurst exponent, the regime-adaptation part inherited from Lee and Cho. Markets alternate between persistent stretches, where moves tend to continue, and anti-persistent ones, where they tend to reverse. Above one half is persistent, below is anti-persistent. This is the real Hurst exponent of the MSCI World ETF, estimated with rescaled-range analysis over a rolling 250-day window, from 2019. The switching rule needs at least six of the last ten days to agree. When the label flips, marked in orange, the model rebuilds its graph on the most recent 750 days and fine-tunes its weights. Over this stretch the rule flipped 6 times. Over the full 2003 to 2024 sample the same code flips fifty times, against forty-seven in Lee and Cho.


## 17. From a 20-day window to one forecast

*On screen:* Six-stage shape pipeline; traced forecast vs actual for all 10 countries on the first test day.  ·  *about 57 s*

Now step two: how the network turns the graph into a single number. Every forecast starts from a small table: for each of the ten countries, its last twenty days of realized volatility. First, a multi-scale convolution slides small pattern detectors over each country's history, three, five and seven days wide, giving thirty-six features that describe what its recent volatility looks like. Then three graph layers let each country mix in what its senders know; with three layers, information can travel three arrows away. A final layer maps each country to one number, and a softplus keeps it positive, because volatility can't be negative. The tables are a real trace through one trained Rényi model for the first test day, 2023-02-28. Brazil, for example, was forecast at 1.68 against an actual 1.59.


## 18. Inside a graph layer

*On screen:* GCN layer formula; one country's incoming weights under both α.  ·  *about 51 s*

This is one graph layer. Each country's new features are its own features, the first term, plus a weighted average of its senders' features, the second term. The weights, A-hat, come from the normalised transfer-entropy matrix; W1 and W2 are learned. The graph itself is fixed, which is why the two models differ only in A-hat. Take Australia. The bars show whom it listens to under each model; a dash means that arrow isn't in the graph. If each country's feature were just its latest volatility on the first test day, the neighbour mix would be 0.863 under Rényi and 1.063 under Shannon. The real layer does the same arithmetic on thirty-two-number feature vectors, three layers deep, every day.


## 19. Training protocol

*On screen:* Train / validation / test timeline with real dates; MSE·Adam, early stop, seeds.  ·  *about 55 s*

Training follows the usual discipline for time series. The samples are split in time order, never shuffled. In the preliminary run that's 942 training days from 2018 to November 2021, 314 validation days, and 314 test days from February 2023 to May 2024. The graph, the choice of alpha and the stopping point all come from the first two blocks, so the test block is untouched until scoring. The network minimises mean squared error with the Adam optimiser, checks the validation block after every pass, and keeps the best weights. Because a neural network depends on its random starting point, each model is trained from several seeds, the same seeds for both models, and the forecasts are averaged. That's three seeds here and twenty in the full run.


## 20. Out-of-sample forecasts

*On screen:* Germany, test block: actual vs both forecasts.  ·  *about 41 s*

Here's the result of step two for Germany over the whole test block, February 2023 to May 2024. Black is what actually happened, blue is the Shannon baseline, orange dashed is the Rényi model. Both follow the general level and lag the turning points. This test block happens to be a calm stretch: there are no crisis windows in it. That matters for the thesis, and I'll come back to it at the end. The two forecast lines are close, which is why we need a loss function and a formal test rather than eyeballing.


## 21. Scoring with QLIKE

*On screen:* QLIKE formulas; penalty curve with ½× and 2× marked.  ·  *about 50 s*

Step three: scoring. For every country on every test day, I compare the forecast with what actually happened using QLIKE. Take the ratio of actual variance to forecast variance; QLIKE is that ratio minus its log minus one, and a perfect forecast gives zero. Two reasons for QLIKE. Realized volatility is itself a noisy measurement, and Patton showed QLIKE still ranks models correctly under that noise. And it depends on the ratio rather than the size of the error, so a few enormous crisis days can't dominate the average. It's also asymmetric: forecasting half the true volatility costs 1.61, forecasting double costs 0.64. Under-predicting risk is the expensive mistake, which is what a risk manager wants.


## 22. From penalties to a daily differential

*On screen:* First five test days: QLIKE for 4 countries, 10-country daily means, dₜ.  ·  *about 46 s*

Here's the arithmetic on the real test block. Each cell is one country's QLIKE on one day, baseline then Rényi; four of the ten countries are shown to fit the slide. For each day I average across all ten countries, which gives one score per model per day. The daily loss differential, d-t, is the baseline's score minus the Rényi model's score, so a positive d-t means Rényi did better that day. Why average across countries before testing, instead of treating every country-day as its own observation? Because on any given day the countries' losses move together. Counting them as ten independent observations would overstate the evidence.


## 23. The daily differential over the test block

*On screen:* dₜ bars and running total for 314 test days; mean QLIKE of each model and d̄ (Table 4.1.1).  ·  *about 40 s*

This is d-t for all 314 test days: orange bars are days Rényi won, blue bars days Shannon won. Below is the running total. If one model had a genuine, consistent edge, that line would climb steadily. Here it drifts downward, meaning the Shannon baseline tends to win. On average the baseline's QLIKE is 0.01360 and the Rényi model's is 0.01430, a mean differential of -0.000699, or -5.14 percent of the baseline's loss. These are the numbers in Table 4.1.1. The question for step four is whether a gap that size could be luck.


## 24. Why the standard error needs HAC

*On screen:* Autocorrelation of dₜ; naive vs HAC standard error.  ·  *about 46 s*

Step four: is the difference real? The test divides the average differential by its standard error. The textbook standard error assumes every day is independent, and volatility isn't like that: good and bad days come in clusters. The chart shows the autocorrelation of d-t in the preliminary test block: at lag one it's 0.62, so today's differential strongly predicts tomorrow's. Those days aren't independent pieces of evidence. The Newey–West estimator, HAC for heteroskedasticity- and autocorrelation-consistent, adds back the covariance between nearby days up to a bandwidth L, which for 314 test days is 5. The honest standard error is 0.000363, 62 percent larger than the naive one.


## 25. The Giacomini–White test

*On screen:* Hypotheses; null bell curve with t = −1.93 and 1.645; result table (Table 4.1.2).  ·  *about 52 s*

The test is Giacomini–White in its unconditional form. The null hypothesis says both models have equal expected loss, so the expected differential is zero. The alternative, fixed in advance, is one-sided: the Rényi model has lower expected loss. The statistic is the observed mean differential minus zero, divided by its HAC standard error, and I reject only if it exceeds 1.645, the dashed line. I use Giacomini–White rather than Diebold–Mariano because it stays valid for models whose parameters are estimated, and for nested models like these two. In the preliminary run, t is -1.928. The shaded area to its right, the p-value, is 0.973. So the null of equal accuracy is not rejected, and the observed differential actually points the other way.


## 26. Reading the verdict

*On screen:* Pre-registered outcome table with the preliminary row highlighted; three protections.  ·  *about 64 s*

Before looking at any result I fixed how each outcome is reported. If t exceeds 1.645 and the gain is at least five percent of the baseline's loss, the improvement is significant and meaningful. If it's significant but below five percent, it's detectable but economically immaterial. Between the critical values there's no evidence of a difference. And if the differential is negative with a large statistic, I report that the Shannon baseline is better. The preliminary run lands in that last row, highlighted: t of -1.93. I report it as favouring Shannon, and I don't switch to a two-sided test to rescue the result. Three choices protect the test: alpha one half was fixed beforehand, the direction is fixed, and the minimum meaningful effect is fixed. The preliminary run is reduced in scope, so this is a demonstration of the format; the confirmatory answer comes from the full run.


## 27. Describing the network (RQ2)

*On screen:* Three RQ2 measures; calm vs crisis expectation.  ·  *about 52 s*

RQ2 asks what the two measurements see rather than how well they forecast. I compare three things. First, each country's out-degree, the number of arrows it sends, and the top three transmitters by that count. Second, how many of the twenty kept arrows appear in both graphs. Third, Kendall's tau between the two out-degree rankings, with a bootstrap confidence interval. The theory gives a direction. Jizba and colleagues show that for Gaussian returns, which calm markets roughly are, every alpha gives the same answer, and as returns get fat-tailed, as in crises, the answers drift apart. So the full run compares calm and crisis windows separately. The preliminary run computed all three on one window, the training block, which includes COVID.


## 28. Dominant transmitters

*On screen:* Out-degree bars for all 10 countries under both α; top 3 each; shared edges (Table 4.1.3).  ·  *about 48 s*

Here is the preliminary answer to the first RQ2 measure, on the training window from 2018 to late 2021. Each country has two bars: orange is its out-degree under Rényi, blue under Shannon. Germany leads under both, with 5 and 6 outgoing arrows. After Germany the lists part ways: under Rényi the next transmitters are Canada and the UK; under Shannon they're Taiwan, then the UK tied with South Africa. That's Table 4.1.3. The two graphs share 12 of their 20 edges, far more than the 4.4 you'd expect if the two had picked edges at random, so the measures largely agree on the network while reshuffling the second tier of senders.


## 29. Agreement between the two networks

*On screen:* Slope chart of the two out-degree rankings; Kendall τ and its interval (T6).  ·  *about 56 s*

The third measure asks whether the two rankings put the countries in the same order. Left is the Shannon out-degree ranking, right is Rényi, and each line connects the same country. Lines that cross are pairs whose order flips. Out of the forty-five pairs of countries, 24 keep the same order, 7 flip, and the rest are tied in one ranking. Kendall's tau puts that into one number: 0.453, with a ninety-five percent bootstrap interval from 0.270 to 0.919. So the rankings are moderately similar, and the interval is wide: it's compatible with anything from weak to near-perfect agreement. That's why RQ2 is reported descriptively. And a structural difference is never used as evidence for or against RQ1: two measures can see different networks without the forecasts changing, and the reverse.


## 30. Next: the full-scale run

*On screen:* Next steps: full sample, 1,000 surrogates, 20 seeds, crisis vs calm.  ·  *about 49 s*

To close: the preliminary run shows that the full pipeline works end to end on real data, and it shows the exact format the results will take. It isn't the confirmatory answer, for one reason above all: its test block, 2023 to 2024, is calm. The thesis is about crises, and there were almost no turbulent days to test on. The full run starts in 2003, so the test period covers the global financial crisis, COVID and the Russia–Ukraine episode. It also uses a thousand surrogates instead of a hundred, twenty seeds per model instead of three, and it compares the network in crisis and calm windows separately. Thank you, and I welcome your questions.


---

## Anticipated panel questions

**Why QLIKE and not RMSE?**

Realized volatility is a noisy estimate of true volatility. Patton (2011) showed that QLIKE and MSE are the losses that still rank models correctly when the target is noisy, and QLIKE depends on the ratio of forecast to actual, so a few crisis days cannot dominate the average. RMSE and MAE are still reported as secondary numbers.

**Why a one-sided test?**

The hypothesis is directional and was fixed in advance: the Rényi graph lowers expected loss. When the differential comes out negative, as in the preliminary run, I report that it favours the Shannon baseline. I never switch to a two-sided test after the fact.

**Where is the “expected difference” in your t-statistic?**

It's zero. H₀ says both models have equal expected loss, so t = (d̄ − 0) / SE. The observed difference is d̄, the average of the daily loss differentials.

**Why Giacomini–White and not Diebold–Mariano?**

Diebold–Mariano treats forecasts as given, and its asymptotics break down for nested models. These two models are nested (α = 1 is a special case) and their parameters are estimated. Giacomini–White remains valid under a fixed estimation window. Diebold–Mariano is reported as a robustness check.

**Why α = 0.5 and not whichever α does best?**

Choosing the best α on the test set would invalidate every p-value computed on that test set. α = 0.5 was fixed before any result.

**Why row-normalise the adjacency matrix?**

Rényi and Shannon values sit on different scales, because α enters through 1/(1 − α). On raw weights, a larger scale alone would make the graph louder, so normalising keeps the comparison about structure. It is applied identically to both models.

**Why keep the 20 strongest edges instead of only the significant ones?**

With 100 surrogates, 36 Rényi edges but 0 Shannon edges passed the false-discovery correction. The baseline would have had no graph at all. Matching the edge count makes the comparison about which edges, not how many. The 1/101 p-value floor at 100 surrogates is also why the full run uses 1,000.

**The preliminary result favours Shannon. Doesn't that sink the thesis?**

No. It is a reduced-scope run whose test block, 2023–24, contains almost no turbulence: 4 turbulent days out of 314. The thesis is about crisis periods, so the confirmatory test needs the full 2003–2024 sample. The preliminary run demonstrates the pipeline and the reporting format.

**Why does the out-degree top 3 for Shannon list three countries when four could qualify?**

The UK and South Africa are tied on 3 outgoing edges under Shannon. The slide marks the tie. Table 4.1.3's prose lists the UK only.

**If the networks look different, doesn’t that prove the Rényi model is better?**

No. The structural comparison and the forecasting test are independent by construction. RQ2 is never used as evidence for RQ1.

---

## Before the defence: prep notes for you (do not read aloud)

The deck reproduces the preliminary run exactly as it was run, so it matches Chapter IV to the last digit (t = -1.928, p = 0.9731). Three issues in the thesis pipeline (`.build/core.py`) affect those numbers, and one of them may come up in questions.

1. **Message direction.** `GCNLayer` uses `einsum('ij,…')` on a matrix whose `A[i, j]` means i → j, so in the run behind these numbers each country aggregated from the countries it *sends to*. Slides 14 and 17 describe the intended direction (a country listens to its senders), as §3.4 does.
2. **Two days ahead.** `make_splits` builds inputs from `rv[t-20:t]` but targets `rv[t+1]`, so the GNN never sees the day before the forecast date. The random walk and HAR-RV do.
3. **Hurst refits.** Slide 15 describes the regime-triggered rebuild and fine-tune from §3.6. The thesis pipeline currently trains once and keeps one fixed graph.

**What happens when 1 and 2 are fixed.** Same data, same graphs, same seeds and settings:

| | As run (Chapter IV, this deck) | Fixed direction + window |
|---|---|---|
| Mean QLIKE, H-ETE-GNN | 0.01360 | 0.00847 |
| Mean QLIKE, HR-ETE-GNN | 0.01430 | 0.00773 |
| d̄ (positive = Rényi better) | -0.000699 (-5.1%) | +0.000739 (+8.7%) |
| GW t, one-sided p | -1.93, 0.973 | 3.67, 0.0001 |
| Random walk / HAR-RV QLIKE | 0.00601 / 0.00619 | same |

The sign of the preliminary H1a result flips once the two bugs are fixed. Don't present the fixed numbers as a finding either: every seed in both runs hit the 200-epoch cap (none early-stopped), there are only 3 seeds, and the H1a sign has flipped before on a surrogate-count change. The lesson is that the preliminary number is fragile, and the full run should use the fixed code. Even fixed, both GNNs still trail the random walk and HAR-RV on this calm test block, which a panelist may ask about.

If you fix the code before the defence, the deck can be regenerated on the corrected run in one pass.
