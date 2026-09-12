# Communications

Drafts for talking about Digital Ghost publicly: a LinkedIn post, a blog post, and a preprint skeleton.

> [!IMPORTANT]
> **No experiment has been run. This repository contains no results.**
>
> Every draft below is written for the **pre-data stage** — announcing a design, not a finding. Passages that can only be written once data exists are marked `[PLACEHOLDER: …]`. **Do not fill a placeholder with an estimate, an expectation, or a plausible-sounding number.** Publishing a predicted effect as an observed one is the single most damaging thing that could happen to this project's credibility, and it is exactly the kind of error that is easy to make while a draft sits half-finished.
>
> A results-stage variant of the LinkedIn post is included separately, to be used *only* after `digital-ghost analyze` has produced real output.

## Before publishing anything

Three decisions to settle first. None of them are mine to make, but publishing without settling them is worse than delaying.

**1. Naming the subject.** The drafts name Charlie Kirk, because the study is about one specific person's likeness and the claim is not meaningful otherwise. He is also a real, identifiable person who was killed, and this material concerns face-swap content depicting him. Weigh whether the public-facing versions should name him, describe him generically ("a prominent political commentator"), or name him only in the preprint where the methodological necessity is clearest. The repository and the rating question currently name him.

**2. Ethics review status.** If review is pending or not yet sought, say so plainly rather than omitting it. Reviewers and readers notice the gap, and disclosing it costs far less than being asked about it later.

**3. Scope discipline.** The single most likely misreading is *"AI models are contaminated by meme content."* This study is a controlled fine-tune of one open-weights model. It cannot speak to what is inside any deployed commercial model. Every draft below states the limit explicitly; keep it in, especially in the shortest formats where it is most tempting to cut.

---

## 1. LinkedIn post — design stage (usable now)

> **Length:** ~230 words. LinkedIn truncates around 210 characters, so the first two lines carry the click.

---

I built a research instrument and I'm publishing the design before I have any results.

The question: if you fine-tune an image model on enough face-swap memes of one real person, does his face start showing up in prompts that never asked for him?

That's easy to ask and hard to answer honestly. Almost any fine-tune will make *some* face appear, so a result only means something if the design can rule out the boring explanations.

Digital Ghost is built around that problem:

→ **Three arms.** Press photos (does the pipeline imprint a likeness at all?), face-swap memes (the condition under test), and an unrelated political figure (does *any* face appear, or this one?).

→ **A dose ladder.** 10 / 25 / 50 / 100 / 200 images, three seeds each — 45 training runs. One data point is an anecdote; a curve is a finding.

→ **Captions held constant.** Identical neutral, name-free captioning across every arm, so caption content can't explain a difference.

→ **30 prompts frozen before any run.** None name anyone. They're committed to git first, so nothing can be tuned toward a result afterward.

→ **Blinded human raters**, weighted by hidden calibration pairs, fitted against an untrained stock-model floor.

No data collected yet. I'm publishing the design first so it can be criticised before I have results I'd be tempted to defend.

Code, design and open questions: [REPO URL]

---

## 2. LinkedIn post — results stage (use only after real analysis output)

> Keep the structure, replace every placeholder with measured values. If a result is null or messy, **say so in the first line** — a clean null reported plainly is worth more than a hedged positive.

---

`[PLACEHOLDER: one-sentence headline finding, stated flatly — e.g. "Across 45 fine-tunes, X." If the result is null, lead with the null.]`

Six months ago I posted the design for this before running it. Here's what happened.

The setup: three arms — press photos, face-swap memes, and an unrelated political figure — used to fine-tune SDXL at five exposure levels, three seeds each. 45 runs. Every checkpoint evaluated on the same 30 prompts, frozen before any training, none of which name anyone. `[PLACEHOLDER: N]` blinded raters, `[PLACEHOLDER: N]` pairwise judgements.

What we found:

→ **Meme arm:** `[PLACEHOLDER: fitted strength vs. stock baseline, with interval]`
→ **Press-photo arm:** `[PLACEHOLDER: — the positive control]`
→ **Control figure:** `[PLACEHOLDER: — the negative control]`
→ **Dose-response:** `[PLACEHOLDER: shape, threshold if any, or explicitly "no monotonic relationship"]`

What this does **not** show: that any deployed commercial model is contaminated. This is a controlled fine-tune of one open-weights model, not an audit of anyone's training data.

`[PLACEHOLDER: the most interesting thing that surprised you, including anything that went wrong]`

Full data, code and preprint: [REPO URL]

---

## 3. Blog post — design stage (usable now)

> **Length:** ~1,300 words. Publish under a title like *"Measuring whether a face leaks into an image model"* or *"I built the instrument before I had the data."*

---

### Measuring whether a face leaks into an image model

There is a question I couldn't stop thinking about, and it turned out to be much harder to answer honestly than to ask.

If you take an image generator and fine-tune it on enough face-swap memes of one real person — not photographs of him, but synthetic content built from his likeness — does that face start appearing in images you never asked it for? Not when you type his name. When you type *"a man speaking at a podium."*

I've built the experiment to test this. I haven't run it. I'm publishing the design first, deliberately, and I'll explain why at the end.

#### Why the obvious version of this experiment is worthless

The naive approach is: fine-tune a model on meme images, generate some pictures, look at them, decide whether the face is there.

That result would mean nothing, for at least four reasons.

**Almost any fine-tune imprints something.** Train an image model on 200 photos of anyone and it will start producing faces that resemble them. If you only run the meme condition, you cannot distinguish *"meme content specifically causes this"* from *"any 200 images of a person cause this."*

**Captions are a confound hiding in plain sight.** Meme images often carry text, watermarks, and a different visual register from press photography. If the meme arm's captions differ from the control's, any difference in output might be explained by caption content rather than image content.

**One data point is an anecdote.** "It happened at 200 images" tells you almost nothing without knowing what happens at 10, at 50, at 100. The interesting object is the shape of the curve, not a single point on it.

**Your own eyes are the least trustworthy instrument available.** The person who designed the study is the last person who should be judging whether a generated face resembles the subject. Motivated perception is not a character flaw; it's the default.

#### What the design does about each of those

**Three arms, not one.** The `meme` arm is the condition under test. The `standard` arm — ordinary press photographs — is a positive control: if full-dose press photos *don't* produce a recognisable likeness, then the hyperparameters are simply wrong and every other result is uninterpretable. The `control` arm is an unrelated political figure with no meme presence, which separates "this pipeline makes faces appear" from "this specific exposure makes *this* face appear."

Without the negative control, a positive finding is unpublishable. Without the positive control, a null finding is unpublishable. You need both, and you need them in the same run on the same hardware.

**Captions held constant.** Every image in every arm gets the same neutral, name-free captioning scheme. No caption anywhere in the study names anyone. Caption content therefore cannot explain a difference between arms — it's identical by construction.

**A dose ladder, with seeds.** Five exposure levels — 10, 25, 50, 100, 200 images — at three random seeds each. That's 45 separate fine-tuning runs. The seeds matter: without them you cannot tell a real dose effect from the noise of one lucky initialisation.

**Thirty prompts, frozen before anything runs.** Ten "near" (podium, interview, rally), ten "mid", ten "far". None name anyone. They're written once, committed to version control, and never regenerated. This is the cheapest and most important safeguard in the whole design: if the prompt set can be edited after seeing results, the study can be quietly tuned toward whatever it finds.

**Blinded human raters, and a floor.** Raters see two images generated from the same prompt and answer which one, if either, shows the subject. They are never told which arm or dose either image came from. A hidden fraction of pairs are calibration pairs with a known answer, used to weight each rater by demonstrated accuracy. Everything is fitted with a tie-aware Bradley-Terry model against unmodified stock SDXL as the pinned reference — so every number reads as *"how much more likely than the untrained model"*, not as an absolute anyone has to interpret.

#### The unglamorous half

Most of the work was not the statistics. It was making a day-long unattended GPU run trustworthy.

Cells are compared against each other, so they have to be comparable. The sweep records a hardware fingerprint — GPU, driver, CUDA, library versions — and **refuses to continue if it changes mid-run**, because finishing the last ten cells on a different card would put a confound inside the comparison the whole study rests on. There's an override, and it writes the change into every affected cell's metadata rather than hiding it.

Every cell is validated before it counts as done: images that actually decode, aren't all black, aren't all identical, and a checkpoint whose LoRA matrices aren't still all zero — which would mean no gradient ever reached the adapter and the run trained nothing while reporting success. That last one is the genuinely frightening failure, because it looks exactly like success from the outside and would make every arm identical to baseline for reasons that have nothing to do with the research question.

Writing the tests found real bugs. My favourite: the budget cap, which exists to *prevent* spending, had a path where hitting it erased the record of a completed run — so the next resume would retrain and re-pay for a checkpoint already sitting on disk. The safety feature was causing the harm it was built to stop.

#### What this can't tell you

It cannot tell you that any model you actually use is contaminated. This is a controlled fine-tune of one open-weights model under conditions I chose. It probes a mechanism; it does not audit anyone's training data. If this work gets summarised as "AI models are full of meme faces," that summary will be wrong, and I'd rather say so now than argue about it later.

#### Why publish the design first

Because I don't have results yet, which means I have nothing to defend.

Once the numbers exist, every design choice I made becomes something I have an interest in justifying. Right now the frozen prompts are just a file; later they'd be the prompts that produced my finding. The honest moment to expose a methodology to criticism is while changing it is still free.

So: the design is public, the code is public, and the open questions are listed in the repository. If something here is wrong, the cheapest possible time to tell me is before I collect the data.

`[PLACEHOLDER: results section — add only after the sweep and analysis have actually run. Report the null as prominently as a positive.]`

---

## 4. Preprint skeleton

> Target venue: arXiv `cs.CY` or `cs.CV`, cross-listed. Aim ~6–8 pages plus appendices. Every section below carries a note on what belongs in it; nothing is pre-written where it would require data.

### Title

Working: *Dose-Response Measurement of Identity Bleed from Synthetic Meme Content in Text-to-Image Fine-Tuning*

Alternative, plainer: *How Much Synthetic Content Does It Take Before a Face Appears Uninvited?*

### Abstract

`[PLACEHOLDER: 150–200 words, written last.]` Structure: (1) synthetic likeness content circulates at scale and enters training corpora; (2) whether it transfers identity into a model under fine-tuning is untested; (3) we run a three-arm dose-response experiment, 45 fine-tunes, frozen name-free prompts, blinded pairwise human rating; (4) the result, stated plainly including if null; (5) the scope limit, in the abstract itself, not deferred to the discussion.

### 1. Introduction

- The phenomenon: synthetic likeness content, especially face-swap memes, now exists at volume for public figures.
- The gap: plenty of work on deepfake *detection* and on memorisation of *training images*; little on whether synthetic derivative content transfers *identity* into a model such that it surfaces unprompted.
- The specific question, stated as the paper will test it: does exposure to synthetic meme content produce identity bleed in prompts that never name the subject, and is the relationship dose-dependent?
- Contributions: (i) a controlled three-arm dose-response design with both positive and negative controls; (ii) a pre-committed frozen evaluation protocol; (iii) `[PLACEHOLDER: the empirical result]`; (iv) released code, configs and raw judgements.

### 2. Related work

Four threads, with the gap this sits in:

- **Memorisation and extraction in diffusion models** — verbatim training-image reproduction. Different phenomenon: this concerns identity transfer from derived content, not reproduction of specific images.
- **Personalisation methods** (DreamBooth, LoRA, textual inversion) — establish that few images can bind an identity. This study asks whether that binding occurs *unintentionally* and *leaks into unrelated prompts*.
- **Deepfakes and synthetic likeness** — largely detection and provenance. The training-data pathway is the gap.
- **Dataset contamination and data poisoning** — closest in structure; typically adversarial and deliberate, whereas meme circulation is organic and uncoordinated.

### 3. Method

**3.1 Arms.** Definition and sourcing of `standard`, `meme`, `control`, with the role of each as control stated explicitly. `[PLACEHOLDER: final collection counts and inclusion/exclusion criteria]`.

**3.2 Provenance.** Every image carries source URL, date, platform and (where known) generating tool. Report the collection window and platform distribution.

**3.3 Captioning.** The neutral name-free scheme, identical across arms, with the template bank and the banned-terms guard. Emphasise that captions are constant by construction and therefore cannot explain between-arm differences.

**3.4 Fine-tuning.** SDXL base 1.0, LoRA rank 32 on attention projections, 1024px, fp16, gradient checkpointing, 1000 steps, identical hyperparameters across all 45 cells. Vendored trainer pinned to diffusers v0.40.0 with a single documented patch; upstream hash recorded and verified by test. State the hardware, and that consistency was enforced programmatically.

**3.5 Dose ladder and seeding.** 5 doses × 3 seeds × 3 arms. Deterministic seed derivation; dose subsamples drawn without replacement from a fixed per-arm pool.

**3.6 Evaluation protocol.** 30 prompts, 10 per tier, frozen and committed before any training run — cite the commit hash. 5 generation seeds per prompt. Stock SDXL baseline as reference item.

**3.7 Human rating.** Recruitment and sample `[PLACEHOLDER]`. Blinding: raters see neither arm nor dose. Pair-sampling weights. Salted calibration pairs and how rater weights are derived. Exposure survey and its role in stratification.

**3.8 Statistical model.** Tie-aware Bradley-Terry (Davidson, 1970); write the likelihood. Stock SDXL pinned as reference so strengths are interpretable against an untrained floor. Weighted and unweighted fits reported side by side. State the connectivity requirement on the comparison graph and that disconnected items are reported as such rather than estimated.

**3.9 Deviations from the pre-specified design.** `[PLACEHOLDER: enumerate every change made after the design was frozen, with dates and reasons — or state explicitly that there were none.]` This section is not optional, and it belongs before the results, not buried at the end.

### 4. Results

`[PLACEHOLDER — entire section. Nothing here can be written before the analysis runs.]`

Planned contents:
- Table: fitted strength per (arm, dose) with intervals, weighted and unweighted.
- Figure: dose-response curves, three arms, with error bars across seeds.
- Figure: curves split by prompt tier (near / mid / far).
- Positive control: does `standard` at top dose exceed baseline as expected? **If it does not, that result is reported and the rest of the analysis is caveated accordingly** — this is not a reason to withhold the paper.
- Negative control: does `control` stay at baseline?
- Inter-rater agreement and the distribution of calibration-derived weights.
- Sensitivity: does the conclusion survive dropping the rater weighting entirely?

### 5. Discussion

- What the curve shape implies about the mechanism. A threshold, a linear response, and a null each mean different things — say which, before knowing the answer if possible.
- Whether prompt tier modulates the effect, and what that suggests about how the identity is bound.
- Implications for dataset curation and for people whose likeness circulates as meme content.

### 6. Limitations

Written honestly and at length. At minimum:

- **One base model, one adaptation method.** Nothing here generalises to other architectures without testing.
- **Not an audit.** This measures a mechanism under controlled fine-tuning. It says nothing about the contents of any deployed model.
- **The rating question names the subject**, which primes raters. The argument that this inflates absolute level rather than distorting the dose-response shape — since the same question is asked of every pair, and both the baseline and control anchor the scale — should be made explicitly rather than assumed. `[PLACEHOLDER: if a no-name elicitation control was run, report it here.]`
- **Rater pool composition** and its likely effect on familiarity-dependent judgements.
- **The meme arm is a convenience sample** of what was collectable, not a representative sample of circulating content.
- `[PLACEHOLDER: anything that actually went wrong during the run — failed cells, excluded data, hardware events.]`

### 7. Ethics statement

- Human subjects: review status `[PLACEHOLDER]`, consent procedure, withdrawal mechanism, data retention and deletion.
- The subject is a real, identifiable, deceased person. State why naming him is methodologically necessary, and what the study does not assert about him.
- Why the collected meme images are **not** released, and what is released in their place (provenance manifests, hashes).
- Dual-use: this work could in principle inform someone trying to *cause* identity bleed. Address it rather than ignoring it — the mitigation is that the design measures a threshold rather than optimising one.

### 8. Data and code availability

- Repository, commit hash, and the frozen-prompt commit specifically.
- Released: all configs, the full analysis pipeline, raw pairwise judgements (rater-anonymised), per-cell metadata, cost ledger.
- Withheld: raw meme images. Provenance manifests and perceptual hashes released instead.
- Note the vendored trainer's pinned upstream hash so the training code is reconstructible exactly.

### 9. Author contributions and acknowledgements

CRediT taxonomy roles. Acknowledge raters as a group without identifying them.

### References

Seed set to build from: Davidson (1970) on ties in paired comparisons; Bradley & Terry (1952); Ruiz et al. on DreamBooth; Hu et al. on LoRA; Podell et al. on SDXL; Carlini et al. on extracting training data from diffusion models; Somepalli et al. on diffusion memorisation and copying.
