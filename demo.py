# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Setup
#

# %%
import pandas as pd
import numpy as np
import duckdb
import sys
import json
from poibin import PoiBin

# %%
hmac = json.load(open("../g-credentials-hmac.json"))
duckdb.sql(
    f"""
INSTALL httpfs;
LOAD httpfs;
SET s3_endpoint='storage.googleapis.com';
SET s3_access_key_id='{hmac['s3_access_key_id']}';
SET s3_secret_access_key='{hmac['s3_secret_access_key']}';
"""
)

pd.set_option("future.no_silent_downcasting", True)

# %% [markdown]
# ## Lecture des données
#
# - FBRef comme source
# - [Avec le détail des xG et PSxG par tir](https://fbref.com/en/matches/652abd24/Montpellier-Strasbourg-August-18-2024-Ligue-1)
# - Modèles alimentés par StatsBomb avant, par Opta désormais (moins bien)
#

# %% [markdown]
# ### On récupère les fixtures pour les 5 grands championnats
#

# %%
# on récupère les fixtures
fixtures_df = duckdb.query(
    """
    SELECT * FROM 'gs://serac/pointu/fbref/*/2024-2025/fixtures.parquet'
"""
).df()

fixtures_df.head()

# %% [markdown]
# ### On récupère les stats des gardiens
#

# %%
# on récupère les stats de gardien par match
keepers_stats_df = duckdb.query(
    """
    SELECT 
        squad_id, 
        squad_label,
        match_id,
        player_label,
        player_id,
        minutes
    FROM 'gs://serac/pointu/fbref/*/2024-2025/players-stats.parquet'
    WHERE position = 'GK'
"""
).df()

keepers_stats_df.head()

# %% [markdown]
# ### On récupère les tirs cadrés, i.e. avec un PSxG défini
#
# > PSxG : Si les xG représentent la charnière sur laquelle s’articule le soccer analytics, les PSxG en sont un heureux prolongement correspondant à la probabilité de marquer un but, mais calculée après le tir. La nuance est de taille car ces modèles disposent d’informations sur la frappe comme sa vitesse ou sa trajectoire. Ainsi, seuls les ballons cadrés et non-contrés sont pris en compte. En faisant l’outil de référence pour estimer la capacité d’un portier à préserver sa cage.
#

# %%
shots_df = duckdb.query(
    """
    SELECT 
        * EXCLUDE (minute),
        -- on garde la minute en string pour l'affichage et le tri
        minute as minute_str,
        -- on convertit les minutes en int
        regexp_extract(minute, '\\d+', 0)::int AS minute
    FROM 'gs://serac/pointu/fbref/*/2024-2025/shots.parquet'
    WHERE 
        -- on écarte les tirs sans PSxG
        psxg_shot IS NOT NULL 
        AND psxg_shot > 0
        -- on écarte les tirs sur la barre
        AND outcome NOT IN ('Woodwork')
"""
).df()

shots_df

# %% [markdown]
# On a normalement 2 issues possibles à un tir cadré
#

# %%
assert (
    shots_df["outcome"].isin(["Saved", "Goal"]).all()
), "Il y a plus de 2 issues possibles à un tir cadré"
shots_df["outcome"].value_counts()

# %% [markdown]
# ## Normalisation des données
#

# %% [markdown]
# ### Deux gardiens dans le même match
#
# > Pour nos calculs, on a besoin de déterminer quel gardien est sur le terrain au moment d'un tir dans les cas où il y a plusieurs gardiens dans un même match.
#

# %%
assert (
    keepers_stats_df.groupby(["match_id", "squad_label"])
    .filter(lambda x: len(x) > 1)
    .empty
), "Il y a plusieurs gardiens dans le même match"

# %% [markdown]
# #### On calcule grossièrement l'entrée et sortie des gardiens pour pouvoir faire le lien avec les tirs
#

# %%
keepers_stats_enhanced_df = (
    keepers_stats_df.groupby(["match_id", "squad_label"])
    .apply(
        lambda x: x.assign(
            start=(x["minutes"] + 1).cumsum().shift(1).fillna(0),
            end=x["minutes"].cumsum(),
        )
    )
    .reset_index(drop=True)
)

keepers_stats_enhanced_df


# %% [markdown]
# ### On crée une méthode, là aussi approximative, pour retrouver les tirs cadrés sur un gardien
#

# %%
def get_shots_against(player_id):
    return (
        keepers_stats_enhanced_df
        # Juste les stats de notre gardien
        .loc[lambda x: x.player_id == player_id]
        # On récupère les tirs contre ce gardien
        .merge(
            shots_df,
            on="match_id",
            how="left",
            suffixes=("", "_shot"),
        )
        .loc[
            lambda x: (x["squad_id"] != x["team_id"])
            & (x["minute"] <= x["end"])
            & (x["minute"] >= x["start"])
        ]
        # On ajoute les données des matchs
        .merge(fixtures_df[["match_id", "date"]])
        .sort_values(["date", "minute"])[
            [
                "squad_id",
                "is_penalty",
                "squad_label",
                "match_id",
                "player_label",
                "player_id",
                "minute",
                "outcome",
                "xg_shot",
                "psxg_shot",
                "team_label",
                "player_label_shot",
            ]
        ]
        .rename(
            columns={
                "team_label": "opponent_label",
            }
        )
        # On normalise le résultat du tir
        .assign(outcome=lambda x: x["outcome"].replace({"Saved": 0, "Goal": 1}))
    )


# %% [markdown]
# Faisons un test avec un gardien donné
#

# %%
shots_rulli_df = get_shots_against("625c144a")
shots_rulli_df


# %% [markdown]
# > On notera qu'on a finalement un très faible volume de tirs cadrés pour un gardien, même s'il est titulaire systématiquement
#
# > On tient compte de penaltys, c'est un choix méthodologique
#

# %% [markdown]
# ## Calcul de la performance avec la loi Poisson binomiale (une généralisation de la loi binomiale)
#

# %% [markdown]
# > La loi Poisson binomiale est une généralisation de la loi binomiale. Elle permet de calculer pour une somme de tirages (les tirs), la probabilité d'encaisser N buts
#
# > Le score qu'on va calculer pour chaque gardien, c'est la P(d'encaisser autant ou moins de buts), soit la CDF.
#

# %% [markdown]
# ### On crée une fonction pour calculer la performance d'un gardien
#

# %%
def compute_dist_performance(id):
    shots_against_df = get_shots_against(id)
    pb = PoiBin(shots_against_df["psxg_shot"])

    # Le nombre de buts encaissés
    goals = shots_against_df["outcome"].sum()
    # Le nombre de buts moyens qu'on aurait dû encaisser d'après le modèle
    psxg = shots_against_df["psxg_shot"].sum()

    return pd.Series(
        {
            "prob_cum": pb.cdf(goals),
            "goals": goals,
            "psxg": psxg,
            "keeper_id": id,
        }
    )


# %% [markdown]
# ### On aggrège les stats par gardien
#

# %%
keepers_df = (
    keepers_stats_df.groupby(["player_id", "player_label"])
    .agg(
        minutes=("minutes", "sum"),
        teams=("squad_label", lambda x: ", ".join(x.unique().tolist())),
    )
    .reset_index()
)

keepers_df

# %% [markdown]
# ### On lance le calcul
#

# %%
threshold_minutes = 450
threshold_goals = 1

ranking_df = (
    # On calcule la performance de chaque gardien
    pd.DataFrame(
        [
            compute_dist_performance(player_id)
            for player_id in keepers_stats_enhanced_df["player_id"].unique()
        ]
    )
    # On ajoute les stats par gardien
    .merge(keepers_df, left_on="keeper_id", right_on="player_id", how="left")
    # On trie par performance
    .sort_values("prob_cum", ascending=True)
    # On filtre sur le temps de jeu et le nombre de buts
    .query("minutes >= @threshold_minutes")
    .query("goals > @threshold_goals")[
        [
            "keeper_id",
            "player_label",
            "teams",
            "minutes",
            "goals",
            "psxg",
            "prob_cum",
        ]
    ]
    # On calcule le classement
    .assign(
        rank=lambda x: x["prob_cum"].rank(method="min", ascending=True),
        diff=lambda x: x["psxg"] - x["goals"],
    )
    .round(4)
)

# %%
ranking_df

# %% [markdown]
# ### Analyse
#

# %%
labels = [
    "Gianluigi Donnarumma",
    "Lucas Chevalier",
    "Ederson",
    "Geronimo Rulli",
    "Brice Samba",
    "Alisson",
    "Mike Maignan",
]
teams = [
    "Barcelona",
    "Paris-Saint-Germain",
    "Manchester-City",
    "Liverpool",
    "Real-Madrid",
    "Bayern-Munich",
    "Manchester-United",
]

(
    ranking_df.query("player_label in @labels or teams in @teams")
    .style.format(
        {
            "prob_cum": "{:0.4f}",
            "goals": "{:0.2f}",
            "psxg": "{:0.2f}",
            "diff": "{:0.2f}",
            "rank": "{:0.0f}",
        },
    )
    .background_gradient(
        subset=["prob_cum"],
        cmap="RdYlGn_r",
        vmin=0,
        vmax=1,
    )
    .background_gradient(
        subset=["diff"],
        cmap="RdYlGn",
        vmin=ranking_df["diff"].min(),
        vmax=ranking_df["diff"].max(),
    )
    .hide(axis=1, subset=["keeper_id"])
    # no index
    .hide(axis=0)
)

# %% [markdown]
# ## Les questions à se poser
#

# %% [markdown]
# ### La variance d'une saison à l'autre
#

# %% [markdown]
# > Comme souvent en football, l'échantillon est très limité : [Donnarumma était largment premier de la L1 l'an dernier](https://pointu.substack.com/p/comment-evaluer-un-gardien)
#

# %% [markdown]
# ### Le rôle d'un gardien ne se limite plus à faire des arrêts
#

# %% [markdown]
# > Avec des entraîneurs comme Bielsa, Sampaoli, Guardiola notamment, [le rôle d'un gardien ne se limite plus aux seuls arrêts](https://serac.vercel.app/2022-gardien-position)
#

# %% [markdown]
# ### L'importance des coups de pied arrêtés, des relances, ...
#
# > Il manque beaucoup d'autres éléments qui permettent d'évaluer plus complètement la qualité d'un gardien.
#
# - Est-ce qu'il est serein sur les coups de pied arrêtés ?
# - Est-ce qu'il ne crée pas des occasions adverses par ses approximations ou sa fébrilité ?
# - Quelle est l'importance de sa défense dans le résultat ?
#
# D'où l'idée d'utiliser des modèles qui estiment la valeur d'un joueur à travers ses actions (VAEP, expected threat, ...)
#

# %% [markdown]
# ### La qualité du modèle
#
# > Il suffit de comparer les valeurs d'un modèle aux extraits vidéo pour voir que le modèle est souvent déficiant. Un problème qui tend à se résorber avec les nouvelles données, comme [les 360 de StatsBomb](https://github.com/statsbomb/open-data).
#
