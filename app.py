import random
from typing import List, Tuple, Dict, Optional

import pandas as pd
import streamlit as st

# Kriteriji


def wald_choice(payoffs: List[List[int]]) -> Tuple[int, List[int]]:
    """Wald (maximin): bira akciju s najvećim minimumom."""
    row_mins = [min(row) for row in payoffs]
    best = max(range(len(row_mins)), key=lambda i: row_mins[i])
    return best, row_mins


def laplace_choice(payoffs: List[List[int]]) -> Tuple[int, List[float]]:
    """Laplace: bira akciju s najvećim prosjekom."""
    row_avgs = [sum(row) / len(row) for row in payoffs]
    best = max(range(len(row_avgs)), key=lambda i: row_avgs[i])
    return best, row_avgs


def savage_choice(payoffs: List[List[int]]) -> Tuple[int, List[List[int]], List[int], List[int]]:
    """Savage (minimax regret): bira akciju s najmanjim najgorim regretom."""
    n = len(payoffs)
    m = len(payoffs[0])

    col_maxs = [max(payoffs[i][j] for i in range(n)) for j in range(m)]

    regrets = []
    for i in range(n):
        regrets.append([col_maxs[j] - payoffs[i][j] for j in range(m)])

    worst_regret = [max(regrets[i]) for i in range(n)]
    best = min(range(n), key=lambda i: worst_regret[i])
    return best, regrets, worst_regret, col_maxs


def hurwicz_choice(payoffs: List[List[int]], alpha: float) -> Tuple[int, List[float], List[int], List[int]]:
    """Hurwicz: H = α·min + (1-α)·max; bira najveći H."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha mora biti u [0,1]")

    scores, mins, maxs = [], [], []
    for row in payoffs:
        mn, mx = min(row), max(row)
        mins.append(mn)
        maxs.append(mx)
        scores.append(alpha * mn + (1 - alpha) * mx)

    best = max(range(len(scores)), key=lambda i: scores[i])
    return best, scores, mins, maxs


# Provjera izjednačenja (tie)


def unique_best_index(values: List[float], want: str) -> Optional[int]:
    """Vraća indeks jedinstvenog najboljeg; ako je tie vraća None."""
    best_val = max(values) if want == "max" else min(values)
    winners = [i for i, v in enumerate(values) if v == best_val]
    return winners[0] if len(winners) == 1 else None


def criteria_picks_no_ties(payoffs: List[List[int]], alpha: float) -> Optional[Dict[str, int]]:
    """Vraća odabire kriterija samo ako nema tie-a."""
    w_i, w_mins = wald_choice(payoffs)
    if unique_best_index(list(map(float, w_mins)), "max") is None:
        return None

    l_i, l_avgs = laplace_choice(payoffs)
    if unique_best_index(l_avgs, "max") is None:
        return None

    s_i, _, worst_regret, _ = savage_choice(payoffs)
    if unique_best_index(list(map(float, worst_regret)), "min") is None:
        return None

    h_i, h_scores, _, _ = hurwicz_choice(payoffs, alpha)
    if unique_best_index(h_scores, "max") is None:
        return None

    return {"Wald": w_i, "Laplace": l_i, "Savage": s_i, "Hurwicz": h_i}


# Generator (1): svi kriteriji biraju istu akciju


def generate_all_same(n: int, vmax: int = 60, margin_min: int = 1, margin_max: int = 8) -> List[List[int]]:
    payoffs = [[random.randint(0, vmax) for _ in range(n)] for _ in range(n)]
    winner = random.randrange(n)

    # winner je najbolji u svakom stupcu
    for j in range(n):
        other_max = max(payoffs[i][j] for i in range(n) if i != winner)
        payoffs[winner][j] = other_max + random.randint(margin_min, margin_max)

    return payoffs

# Generator (2): svaki kriterij bira različitu akciju


def generate_all_different(
    n: int,
    alpha: float,
    vmax: int = 120,
    max_tries: int = 250000,
) -> List[List[int]]:
    for _ in range(max_tries):
        payoffs = [[random.randint(0, vmax) for _ in range(n)] for _ in range(n)]
        picks = criteria_picks_no_ties(payoffs, alpha)
        if picks is None:
            continue
        if len(set(picks.values())) == 4:
            return payoffs

    raise RuntimeError("Nisam našao tablicu. Povećaj vmax ili max_tries, ili promijeni alpha.")



# prikaz


def a(i: int) -> str:
    return f"a{i+1}"


def theta(j: int) -> str:
    return f"θ{j+1}"


def payoff_df(payoffs: List[List[int]]) -> pd.DataFrame:
    """DataFrame s prvim stupcem 'akcija'."""
    n = len(payoffs)
    df = pd.DataFrame(payoffs, columns=[theta(j) for j in range(n)])
    df.insert(0, "akcija", [a(i) for i in range(n)])
    return df


def regret_df(regrets: List[List[int]]) -> pd.DataFrame:
    """DataFrame regreta s prvim stupcem 'akcija'."""
    n = len(regrets)
    df = pd.DataFrame(regrets, columns=[theta(j) for j in range(n)])
    df.insert(0, "akcija", [a(i) for i in range(n)])
    return df


def hurwicz_sweep(payoffs: List[List[int]], steps: int = 41) -> pd.DataFrame:
    rows = []
    for k in range(steps):
        alpha = k / (steps - 1)
        h_i, _, _, _ = hurwicz_choice(payoffs, alpha)
        rows.append({"alpha": round(alpha, 2), "odabir": a(h_i)})
    return pd.DataFrame(rows)





st.set_page_config(page_title="TOMI - 3.dz", layout="wide")
st.title("Kriteriji odlučivanja: Wald, Hurwicz, Savage, Laplace")

with st.sidebar:
    st.header("Postavke")

    n = int(
        st.number_input(
            "Upisati n (broj akcija, n ≥ 5)",
            min_value=5,
            max_value=50,
            value=10,
            step=1,
        )
    )

    alpha = st.slider("Hurwicz α", 0.0, 1.0, 0.60, 0.05)

    vmax = int(
        st.number_input(
            "vmax (maks. isplata u tablici)",
            min_value=20,
            max_value=1000,
            value=150 if n >= 10 else 120,
            step=10,
        )
    )

    mode = st.radio(
        "Odabrati cilj:",
        ["svi kriteriji biraju istu akciju", "svaki kriterij bira različitu akciju"],
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("Generiraj", use_container_width=True):
            if mode.startswith("(1)"):
                st.session_state.payoffs = generate_all_same(n, vmax=vmax)
            else:
                st.session_state.payoffs = generate_all_different(n, alpha=alpha, vmax=vmax)

    with colB:
        if st.button("Reset", use_container_width=True):
            st.session_state.payoffs = None


    payoffs = st.session_state.get("payoffs", None)
    if payoffs is None:
        st.info("Postaviti n i α , odabrati cilj i kliknuti Generiraj!")
        st.stop()
# Izračuni
w_i, w_mins = wald_choice(payoffs)
l_i, l_avgs = laplace_choice(payoffs)
s_i, regrets, worst_regret, col_maxs = savage_choice(payoffs)
h_i, h_scores, h_mins, h_maxs = hurwicz_choice(payoffs, alpha)

# Odabiri
picks = {
    "Wald (maximin)": a(w_i),
    "Laplace (prosjek)": a(l_i),
    "Savage (minimax regret)": a(s_i),
    f"Hurwicz (α={alpha:.2f})": a(h_i),
}

left, right = st.columns([1.25, 1.0], gap="large")

with left:
    st.subheader("Tablica isplata")
    st.dataframe(payoff_df(payoffs), use_container_width=True)
    st.caption("Prvi stupac su akcije (a1..an), stupci su stanja (θ1..θn).")

with right:
    st.subheader("Odabiri kriterija")
    st.write(picks)

    chosen = [w_i, l_i, s_i, h_i]
    if len(set(chosen)) == 1:
        st.success("Sva 4 kriterija biraju istu akciju.")
    elif len(set(chosen)) == 4:
        st.success("Svaki kriterij bira različitu akciju.")
    else:
        st.warning("Fail' ( opet generirati ili povećati vmax).")

st.divider()

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.subheader("Wald")
    st.write({a(i): int(w_mins[i]) for i in range(n)})
    st.caption("Min po retku, zatim max min.")

with c2:
    st.subheader("Laplace")
    st.write({a(i): round(l_avgs[i], 3) for i in range(n)})
    st.caption("Prosjek po retku, zatim max prosjek.")

with c3:
    st.subheader("Hurwicz")
    st.write(
        {
            a(i): {
                "min": int(h_mins[i]),
                "max": int(h_maxs[i]),
                "H": round(h_scores[i], 3),
            }
            for i in range(n)
        }
    )
    st.caption("H = α·min + (1−α)·max.")

st.divider()

st.subheader("Savage – regreti")
st.write({"max po stanju": {theta(j): int(col_maxs[j]) for j in range(n)}})
st.write({"min regret po akciji": {a(i): int(worst_regret[i]) for i in range(n)}})
st.dataframe(regret_df(regrets), use_container_width=True)

st.divider()

st.subheader("Hurwicz: promjena odabira s α")
st.dataframe(hurwicz_sweep(payoffs, steps=41), use_container_width=True)
st.caption("Pomicanjem slidera α, mijenja se i odabir Hurwicza.")
