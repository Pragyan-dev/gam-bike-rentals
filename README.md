# gam-bike-rentals

Fitting a GAM to daily bike rentals, using the [UCI Bike Sharing][uci] dataset (731 days
of rentals in Washington DC, 2011-2012).

The question the notebook asks: how does temperature affect how many bikes get rented?

If you fit a straight line you get one slope, and one slope can only say "hotter is
better, forever". That's obviously wrong at 35 °C. A GAM fixes it without turning the
model into a black box.

## What's a GAM

Linear regression adds up terms like `coefficient * feature`:

```
y = b0 + b1*x1 + b2*x2 + ...
```

A GAM swaps each of those for a function it learns from the data:

```
y = b0 + f1(x1) + f2(x2) + ...
```

Each `f` can be any smooth shape. The adding up is what stays. That's the whole point:
one term per feature means you can plot `f1` by itself and see what the model thinks
temperature does, which you can't really do once features start multiplying together in
a boosted tree.

The flip side is that anything that isn't a plain sum has to be asked for explicitly. If
temperature and humidity genuinely interact, an additive model can't express it.

## How the fit works

Each `f` is a weighted sum of a bunch of small bump-shaped basis functions spread across
the range of the feature. So fitting the curve is really fitting one weight per bump,
which means it's still a linear fit, just on a wider matrix. That's why it's fast.

The bumps sit at fixed positions (knots). More knots, more places the curve can bend.
Here they're spread evenly over the observed temperatures.

Now, enough bumps will happily fit the noise too. So instead of minimising error alone,
the fit minimises

```
error + lam * (how wiggly the curve is)
```

with wiggliness coming from the second derivative. Big `lam`, bending is expensive and
the curve straightens out. Small `lam`, it chases every point. That single knob is why
you don't have to agonise over the number of knots: throw in enough and let the penalty
decide how much of that flexibility actually gets used.

`gridsearch` picks `lam` for you by fitting over a grid and keeping the best cross
validation score.

Two things worth knowing once it's fit:

**Effective degrees of freedom.** Because of the penalty the curve doesn't use all the
parameters it nominally has. This counts how many it really uses. If it comes back at
1.0, the penalty flattened your curve into a straight line and the GAM has quietly
turned into a linear model. Always worth a look.

**Linear tails.** Past the outer knots there are no basis functions left, so the curve
just keeps going at whatever slope it had at the edge. Nothing knows that bike rentals
can't be negative. You get no warning, just a number. The last plot in the notebook is
entirely about this.

## What the notebook does

1. Downloads the UCI daily table and caches it as `day.csv`.
2. Puts the weather columns back into real units. UCI ships them scaled to 0-1, so
   temperature comes out as 0.68 instead of 27.9 °C. Multiply them back, otherwise the
   fitted curve is unreadable, which defeats the purpose.
3. Fits `LinearGAM(s(0))` on temperature, `lam` chosen by `gridsearch`, plus a plain OLS
   line to compare against.
4. Plots line vs curve over the scatter, with a confidence band → `figures/01_linear_vs_gam.png`
5. Plots the temperature effect on its own, via `partial_dependence` → `figures/02_partial_effect.png`
6. Forces the model to predict outside the observed range and plots what it says out
   there → `figures/03_extrapolation.png`
7. Prints a table of every number quoted anywhere, computed rather than typed, so you
   can diff it after messing with the fit.

## Running it

```bash
git clone https://github.com/Pragyan-dev/gam-bike-rentals.git
cd gam-bike-rentals
pip install -r requirements.txt
jupyter lab gam_bike_rentals.ipynb
```

Runs in about ten seconds. Data downloads on first run, plots land in `figures/`.

## Gotchas

**pyGAM won't extrapolate by default.** It clips to the observed range, which is a sane
default and also exactly what I needed to defeat for the third plot. The workaround is
refitting with `edge_knots` set by hand.

**The UCI download 403s from some clients.** `load_bikes()` tries the official zip first
and falls back to a GitHub copy of the same file.

**Confidence bands aren't prediction intervals.** `confidence_intervals` is uncertainty
about the average day. `prediction_intervals` is where a single day might land, and it's
so much wider that it swallows the curve. The plots use the first one.

**Scaling divisors**, in case you want other columns: `temp / 41 C`, `atemp / 50 C`,
`hum / 100 %`, `windspeed / 67 km/h`.

## Data

Fanaee-T, H. & Gama, J. (2013). *Event labeling combining ensemble detectors and
background knowledge*. Progress in Artificial Intelligence, 2(2-3), 113-127. From the
[UCI Machine Learning Repository][uci], CC BY 4.0.

## License

MIT, see [LICENSE](LICENSE). Dataset has its own license, above.

[uci]: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
