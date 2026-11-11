"""
Discrete Probability Distributions - AIML Laboratory
Run with:  streamlit run app.py
"""

import math

import altair as alt
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Discrete Probability Distributions",
    page_icon="Σ",
    layout="wide",
)

NAVY = "#1F3A5F"
INK = "#1B2430"
MUTED = "#5B6675"
RULE = "#D9DEE5"
PAPER = "#F4F6F9"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif;
        color: {INK};
    }}
    .stApp {{ background: {PAPER}; }}
    header[data-testid="stHeader"], #MainMenu, footer {{ visibility: hidden; height: 0; }}
    .block-container {{ max-width: 1080px; padding-top: 1.5rem; padding-bottom: 3rem; }}

    /* Masthead */
    .masthead {{
        background: {NAVY};
        color: #FFFFFF;
        padding: 1.6rem 2rem;
        border-radius: 6px 6px 0 0;
    }}
    .masthead h1 {{
        font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
        font-weight: 600;
        font-size: 1.85rem;
        letter-spacing: 0.1px;
        margin: 0 0 0.25rem 0;
        padding: 0;
        color: #FFFFFF;
    }}
    .masthead p {{ margin: 0; font-size: 0.95rem; color: #C9D4E3; }}

    /* Navigation (tabs) */
    div[data-baseweb="tab-list"] {{
        background: #FFFFFF;
        border: 1px solid {RULE};
        border-top: none;
        border-radius: 0 0 6px 6px;
        padding: 0 1rem;
        gap: 0.5rem;
    }}
    button[data-baseweb="tab"] {{ height: 3rem; padding: 0 1.1rem; }}
    button[data-baseweb="tab"] p {{ font-size: 0.97rem; font-weight: 500; }}
    div[data-baseweb="tab-highlight"] {{ background-color: {NAVY}; height: 3px; }}
    div[data-baseweb="tab-panel"] {{ padding-top: 1.75rem; }}

    /* Headings */
    h2.page-title {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 600;
        font-size: 1.6rem;
        color: {NAVY};
        margin: 0 0 0.35rem 0;
        padding: 0;
    }}
    p.lede {{ color: {MUTED}; margin: 0 0 1.5rem 0; max-width: 46rem; line-height: 1.6; }}
    h3.section {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-weight: 600;
        font-size: 1.2rem;
        color: {NAVY};
        margin: 2.1rem 0 0.9rem 0;
        padding: 0 0 0.45rem 0;
        border-bottom: 1px solid {RULE};
    }}

    /* Panels and cards */
    .panel {{
        background: #FFFFFF;
        border: 1px solid {RULE};
        border-radius: 6px;
        padding: 1.1rem 1.3rem;
    }}
    .panel .label {{ color: {MUTED}; font-size: 0.85rem; margin-bottom: 0.35rem; }}
    .panel .expr {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.55rem;
        color: {NAVY};
        font-weight: 600;
    }}
    .panel .detail {{ color: {MUTED}; font-size: 0.9rem; margin-top: 0.4rem; }}

    .stat {{
        background: #FFFFFF;
        border: 1px solid {RULE};
        border-top: 3px solid {NAVY};
        border-radius: 6px;
        padding: 0.9rem 1rem 0.95rem 1rem;
        height: 100%;
    }}
    .stat .name {{ color: {MUTED}; font-size: 0.85rem; font-weight: 500; }}
    .stat .value {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: {INK};
        margin: 0.2rem 0 0.15rem 0;
    }}
    .stat .formula {{ color: {MUTED}; font-size: 0.8rem; }}

    .result {{
        background: #FFFFFF;
        border: 1px solid {RULE};
        border-left: 4px solid {NAVY};
        border-radius: 6px;
        padding: 1.1rem 1.4rem;
        margin-top: 0.5rem;
    }}
    .result .value {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: {NAVY};
    }}
    .result .meaning {{ color: {INK}; margin-top: 0.35rem; line-height: 1.55; font-size: 0.93rem; }}
    .result {{ height: 100%; }}

    /* Home cards */
    .module {{
        background: #FFFFFF;
        border: 1px solid {RULE};
        border-radius: 6px;
        padding: 1.1rem 1.3rem;
        height: 100%;
    }}
    .module .title {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: {NAVY};
    }}
    .module .desc {{ color: {MUTED}; font-size: 0.9rem; margin: 0.4rem 0 0.7rem 0; line-height: 1.5; }}

    /* Inputs */
    div[data-testid="stNumberInput"] label p {{ font-weight: 500; font-size: 0.92rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Probability maths
# --------------------------------------------------------------------------
def finish_stats(mean, variance, skewness):
    return {
        "mean": mean,
        "variance": variance,
        "std": math.sqrt(variance),
        "second_moment": variance + mean**2,
        "skewness": skewness,
    }


# Binomial
def binomial_stats(n, p):
    var = n * p * (1 - p)
    return finish_stats(n * p, var, (1 - 2 * p) / math.sqrt(var) if var > 0 else None)


def binomial_pmf(n, k, p):
    """C(n,k) p^k (1-p)^(n-k), computed in log space so large n is safe."""
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    log_comb = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(log_comb + k * math.log(p) + (n - k) * math.log1p(-p))


# Poisson
def poisson_stats(lam):
    return finish_stats(lam, lam, 1 / math.sqrt(lam))


def poisson_pmf(lam, k):
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


# Geometric: X = number of trials until the first success, k = 1, 2, 3, ...
def geometric_stats(p):
    var = (1 - p) / p**2
    return finish_stats(1 / p, var, (2 - p) / math.sqrt(1 - p) if p < 1 else None)


def geometric_pmf(p, k):
    return (1 - p) ** (k - 1) * p


def fmt(x, digits=4):
    if x is None:
        return "undefined"
    if x != 0 and abs(x) < 1e-4:
        return f"{x:.3e}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def plural(k, one, many):
    return one if k == 1 else many


OUTCOMES = {"success": "successes", "failure": "failures", "defect": "defects"}


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def page_header(title, lede):
    st.markdown(f'<h2 class="page-title">{title}</h2><p class="lede">{lede}</p>', unsafe_allow_html=True)


def section(title):
    st.markdown(f'<h3 class="section">{title}</h3>', unsafe_allow_html=True)


def selected_panel(expression, detail):
    st.markdown(
        f'<div class="panel"><div class="label">Selected distribution</div>'
        f'<div class="expr">{expression}</div><div class="detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def outcome_picker(key):
    choice = st.radio(
        "Outcome being counted",
        ["Success", "Failure", "Defect"],
        horizontal=True,
        key=key,
        help="The outcome whose probability is p. Choose Failure or Defect when p is a failure or defect rate.",
    )
    return choice.lower()


def stats_section(s, formulas):
    """formulas: (mean, variance, std, second moment, skewness) formula strings."""
    section("Summary statistics")
    cards = [
        ("Mean", fmt(s["mean"]), formulas[0]),
        ("Variance", fmt(s["variance"]), formulas[1]),
        ("Standard deviation", fmt(s["std"]), formulas[2]),
        ("Second moment E[X²]", fmt(s["second_moment"]), formulas[3]),
        ("Skewness", fmt(s["skewness"]), formulas[4]),
    ]
    for col, (name, value, formula) in zip(st.columns(5, gap="small"), cards):
        col.markdown(
            f'<div class="stat"><div class="name">{name}</div><div class="value">{value}</div>'
            f'<div class="formula">{formula}</div></div>',
            unsafe_allow_html=True,
        )
    if s["skewness"] is None:
        st.caption("Skewness is undefined here because the variance is zero.")


def results(k, pmf, cdf, upper, context, eq_event, le_event, ge_event):
    boxes = [
        (f"P(X = {k})", pmf, eq_event),
        (f"P(X ≤ {k})", cdf, le_event),
        (f"P(X ≥ {k})", upper, ge_event),
    ]
    for col, (label, prob, event) in zip(st.columns(3, gap="small"), boxes):
        col.markdown(
            f'<div class="result"><div class="value">{label} = {fmt(prob, 6)}</div>'
            f'<div class="meaning">{context} there is a {fmt(prob * 100, 2)}% chance of {event}.</div></div>',
            unsafe_allow_html=True,
        )


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit versions
        st.altair_chart(chart, use_container_width=True)


def pmf_chart(xs, ys, k, xlabel):
    df = pd.DataFrame({"k": xs, "p": ys, "selected": [x == k for x in xs]})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("k:O", title=xlabel, axis=alt.Axis(labelAngle=0, labelOverlap=True)),
            y=alt.Y("p:Q", title="Probability"),
            color=alt.condition("datum.selected", alt.value(NAVY), alt.value("#A9B8CC")),
            tooltip=[alt.Tooltip("k:O", title=xlabel), alt.Tooltip("p:Q", title="P(X = k)", format=".6f")],
        )
        .properties(height=320)
    )
    show_chart(chart)
    if k not in xs:
        st.caption(f"k = {k} lies outside the plotted range, where probabilities are negligible.")
    else:
        st.caption("The highlighted bar marks the selected k.")


def probability_table(xs, pmf_fn, first):
    """Table of P(X = k), P(X ≤ k) and P(X ≥ k) for every plotted k, so values can be read off together."""
    full = list(range(first, xs[-1] + 1))
    pmfs = [pmf_fn(x) for x in full]
    cdfs, running = [], 0.0
    for v in pmfs:
        running += v
        cdfs.append(min(1.0, running))
    rows = []
    for i, x in enumerate(full):
        if x >= xs[0]:
            before = cdfs[i - 1] if i > 0 else 0.0
            rows.append({"k": x, "P(X = k)": pmfs[i], "P(X ≤ k)": cdfs[i], "P(X ≥ k)": max(0.0, 1 - before)})
    df = pd.DataFrame(rows)
    config = {c: st.column_config.NumberColumn(format="%.6f") for c in ["P(X = k)", "P(X ≤ k)", "P(X ≥ k)"]}
    height = min(35 * len(df) + 40, 300)
    try:
        st.dataframe(df, hide_index=True, column_config=config, height=height, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=True, column_config=config, height=height, use_container_width=True)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def render_home():
    page_header(
        "Welcome",
        "This laboratory explores three discrete probability distributions. "
        "Choose a module from the navigation above to begin.",
    )
    modules = [
        ("Binomial Distribution", "Number of successes, failures or defects in a fixed number of independent trials."),
        ("Poisson Distribution", "Number of events in a fixed interval occurring at a constant average rate."),
        ("Geometric Distribution", "Number of trials needed to obtain the first success, failure or defect."),
    ]
    for col, (title, desc) in zip(st.columns(3, gap="medium"), modules):
        col.markdown(
            f'<div class="module"><div class="title">{title}</div><div class="desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )


def render_binomial():
    page_header(
        "Binomial Distribution",
        "X counts how many times the chosen outcome occurs in n independent trials, "
        "each with the same probability p.",
    )
    section("Parameters")
    left, right = st.columns(2, gap="large")
    with left:
        o = outcome_picker("bin_outcome")
        n = st.number_input("Number of trials (n)", min_value=1, max_value=1000, value=20,
                            step=1, format="%d", help="A positive integer.", key="bin_n")
        p = st.number_input(f"Probability of {o} (p)", min_value=0.0, max_value=1.0, value=0.35,
                            step=0.0025, format="%.4f", help="A value between 0 and 1.", key="bin_p")
    errors = []
    if n is None or int(n) != n or n < 1:
        errors.append("The number of trials n must be a positive integer.")
    if p is None or not (0 <= p <= 1):
        errors.append("The probability p must be between 0 and 1.")
    if errors:
        for e in errors:
            st.error(e)
        return
    n, p = int(n), float(p)
    many = OUTCOMES[o]
    with right:
        selected_panel(f"X ~ Binomial(n = {n}, p = {fmt(p, 4)})",
                       f"{n} trials &nbsp;|&nbsp; probability of {o}: {fmt(p, 4)} "
                       f"&nbsp;|&nbsp; probability of no {o}: {fmt(1 - p, 4)}")

    stats_section(binomial_stats(n, p),
                  ("np", "np(1 − p)", "√ Variance", "Variance + Mean²", "(1 − 2p) / √(np(1 − p))"))

    section("Probability calculator")
    k_col, f_col = st.columns([1, 1.4], gap="large")
    with k_col:
        k = st.number_input(f"Number of {many} (k)", min_value=0, max_value=n, value=min(7, n),
                            step=1, format="%d", help=f"An integer from 0 to {n}.")
        if k is None or int(k) != k or not (0 <= k <= n):
            st.error(f"k must be an integer between 0 and n = {n}.")
            return
    k = int(k)
    with f_col:
        st.latex(r"P(X=k) = \binom{n}{k}\, p^{k}\,(1-p)^{\,n-k}")
        st.latex(r"P(X\leq k) = \sum_{i=0}^{k} P(X=i) \qquad P(X\geq k) = \sum_{i=k}^{n} P(X=i)")
    pmf = binomial_pmf(n, k, p)
    cdf = min(1.0, sum(binomial_pmf(n, i, p) for i in range(k + 1)))
    upper = min(1.0, sum(binomial_pmf(n, i, p) for i in range(k, n + 1)))
    st.latex(rf"P(X={k}) = \binom{{{n}}}{{{k}}}\,({fmt(p, 4)})^{{{k}}}\,({fmt(1 - p, 4)})^{{{n - k}}} = {fmt(pmf, 6)}")
    noun = plural(k, o, many)
    results(k, pmf, cdf, upper,
            f"In {n} independent trials with a {o} probability of {fmt(p, 4)},",
            f"getting exactly {k} {noun}",
            f"getting at most {k} {noun}",
            f"getting at least {k} {noun}")

    # Wide supports are trimmed to the region where probabilities are visible.
    all_x = list(range(n + 1))
    all_p = [binomial_pmf(n, x, p) for x in all_x]
    if n > 40:
        shown = [x for x in all_x if all_p[x] >= 1e-4] or [k]
        xs = list(range(min(shown + [k]), max(shown + [k]) + 1))
    else:
        xs = all_x

    section("Probability mass function")
    pmf_chart(xs, [all_p[x] for x in xs], k, f"Number of {many} (k)")

    section("Probability table")
    probability_table(xs, lambda x: all_p[x], 0)


def render_poisson():
    page_header(
        "Poisson Distribution",
        "X counts the events in a fixed interval when events occur independently at a constant average rate λ.",
    )
    section("Parameters")
    left, right = st.columns(2, gap="large")
    with left:
        lam = st.number_input("Average number of events (λ)", min_value=0.01, max_value=100.0, value=6.0,
                              step=0.5, format="%.2f", help="A positive number.", key="poi_lam")
    if lam is None or lam <= 0:
        st.error("The average number of events λ must be greater than 0.")
        return
    lam = float(lam)
    with right:
        selected_panel(f"X ~ Poisson(λ = {fmt(lam, 2)})",
                       f"on average {fmt(lam, 2)} events per interval")

    stats_section(poisson_stats(lam), ("λ", "λ", "√λ", "Variance + Mean²", "1 / √λ"))

    section("Probability calculator")
    k_col, f_col = st.columns([1, 1.4], gap="large")
    with k_col:
        k = st.number_input("Number of events (k)", min_value=0, max_value=500, value=int(round(lam)),
                            step=1, format="%d", help="A non-negative integer.")
        if k is None or int(k) != k or k < 0:
            st.error("k must be a non-negative integer.")
            return
    k = int(k)
    with f_col:
        st.latex(r"P(X=k) = \frac{e^{-\lambda}\,\lambda^{k}}{k!}")
        st.latex(r"P(X\leq k) = \sum_{i=0}^{k} P(X=i) \qquad P(X\geq k) = 1 - P(X\leq k-1)")
    pmf = poisson_pmf(lam, k)
    cdf = min(1.0, sum(poisson_pmf(lam, i) for i in range(k + 1)))
    tail_end = k + int(20 * math.sqrt(lam) + 60)
    upper = min(1.0, sum(poisson_pmf(lam, i) for i in range(k, tail_end + 1)))
    st.latex(rf"P(X={k}) = \frac{{e^{{-{fmt(lam, 2)}}}\,({fmt(lam, 2)})^{{{k}}}}}{{{k}!}} = {fmt(pmf, 6)}")
    noun = plural(k, "event", "events")
    results(k, pmf, cdf, upper,
            f"When events occur at an average rate of {fmt(lam, 2)} per interval,",
            f"observing exactly {k} {noun}",
            f"observing at most {k} {noun}",
            f"observing at least {k} {noun}")

    xs = list(range(int(lam + 5 * math.sqrt(lam) + 5) + 1))

    section("Probability mass function")
    pmf_chart(xs, [poisson_pmf(lam, x) for x in xs], k, "Number of events (k)")

    section("Probability table")
    probability_table(xs, lambda x: poisson_pmf(lam, x), 0)

    section("Effect of λ on the distribution")
    cols = st.columns(3, gap="medium")
    defaults = [2.0, 6.0, 10.0]
    rates = []
    for i, (col, default) in enumerate(zip(cols, defaults), start=1):
        value = col.number_input(f"Rate {i} (λ)", min_value=0.1, max_value=100.0, value=default,
                                 step=0.5, format="%.2f", key=f"poi_cmp_{i}")
        if value not in rates:
            rates.append(float(value))
    labels = [f"λ = {fmt(r, 2)}" for r in rates]
    top = max(rates)
    upper_x = int(top + 5 * math.sqrt(top) + 5)
    rows = [{"k": x, "p": poisson_pmf(r, x), "rate": lab}
            for r, lab in zip(rates, labels) for x in range(upper_x + 1)]
    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", title="Number of events (k)", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("p:Q", title="Probability"),
            color=alt.Color("rate:N", title=None,
                            scale=alt.Scale(domain=labels, range=[NAVY, "#B07A2A", "#4F7F6B"][: len(labels)])),
            tooltip=[alt.Tooltip("rate:N", title="Rate"), alt.Tooltip("k:Q", title="k"),
                     alt.Tooltip("p:Q", title="P(X = k)", format=".6f")],
        )
        .properties(height=320)
    )
    show_chart(chart)


def render_geometric():
    page_header(
        "Geometric Distribution",
        "X is the number of independent trials needed to obtain the first occurrence of the chosen "
        "outcome (k = 1, 2, 3, …).",
    )
    section("Parameters")
    left, right = st.columns(2, gap="large")
    with left:
        o = outcome_picker("geo_outcome")
        p = st.number_input(f"Probability of {o} (p)", min_value=0.01, max_value=1.0, value=0.25,
                            step=0.01, format="%.2f", help="A value greater than 0 and at most 1.",
                            key="geo_p")
    if p is None or not (0 < p <= 1):
        st.error("The probability p must be greater than 0 and at most 1.")
        return
    p = float(p)
    with right:
        selected_panel(f"X ~ Geometric(p = {fmt(p, 2)})",
                       f"trials continue until the first {o} &nbsp;|&nbsp; probability of {o}: {fmt(p, 2)}")

    stats_section(geometric_stats(p),
                  ("1 / p", "(1 − p) / p²", "√ Variance", "Variance + Mean²", "(2 − p) / √(1 − p)"))

    section("Probability calculator")
    k_col, f_col = st.columns([1, 1.4], gap="large")
    with k_col:
        k = st.number_input("Trial number (k)", min_value=1, max_value=1000, value=3,
                            step=1, format="%d", help="A positive integer.")
        if k is None or int(k) != k or k < 1:
            st.error("k must be a positive integer.")
            return
    k = int(k)
    with f_col:
        st.latex(r"P(X=k) = (1-p)^{k-1}\,p")
        st.latex(r"P(X\leq k) = 1-(1-p)^{k} \qquad P(X\geq k) = (1-p)^{k-1}")
    pmf = geometric_pmf(p, k)
    cdf = 1 - (1 - p) ** k
    upper = (1 - p) ** (k - 1)
    st.latex(rf"P(X={k}) = ({fmt(1 - p, 2)})^{{{k - 1}}}\,({fmt(p, 2)}) = {fmt(pmf, 6)}")
    trials = plural(k, "trial", "trials")
    results(k, pmf, cdf, upper,
            f"With a {o} probability of {fmt(p, 2)} on every trial,",
            f"the first {o} occurring on exactly trial {k}",
            f"the first {o} occurring within the first {k} {trials}",
            f"needing at least {k} {trials} to get the first {o}")

    if p >= 1:
        last = 5
    else:
        last = min(200, max(10, math.ceil(math.log(0.001) / math.log(1 - p))))
    xs = list(range(1, last + 1))

    section("Probability mass function")
    pmf_chart(xs, [geometric_pmf(p, x) for x in xs], k, f"Trial of first {o} (k)")

    section("Probability table")
    probability_table(xs, lambda x: geometric_pmf(p, x), 1)


# --------------------------------------------------------------------------
# Layout: header + navigation
# --------------------------------------------------------------------------
st.markdown(
    '<div class="masthead"><h1>Discrete Probability Distributions</h1>'
    "<p>AIML Laboratory &nbsp;|&nbsp; Interactive explorations of Binomial, Poisson and "
    "Geometric distributions</p></div>",
    unsafe_allow_html=True,
)

home_tab, binomial_tab, poisson_tab, geometric_tab = st.tabs(["Home", "Binomial", "Poisson", "Geometric"])
with home_tab:
    render_home()
with binomial_tab:
    render_binomial()
with poisson_tab:
    render_poisson()
with geometric_tab:
    render_geometric()