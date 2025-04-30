import re
import numpy as np
import pandas as pd


# - on supprime les colonnes qui contiennent "pct"
# TODO : certaines ratio ne contiennent pas le mot "pct"
def remove_pct_columns(df):
    return df.loc[:, ~df.columns.str.contains("pct")]


# - on supprime les stats du gardien
# - on garde les rows des gardiens en revanche
def remove_gk_columns(df):
    return df.loc[:, ~df.columns.str.startswith("gk_")].drop(columns=["keeper_stats"])


# - on supprime les colonnes inutiles, comme le numéro de maillot
def remove_unused_columns(df):
    return df.drop(
        columns=[
            "shirtnumber",
        ]
    )


# - on filtre par temps de jeu
def filter_by_minutes(df, min_minutes=450):
    return df.loc[lambda x: x["minutes"] >= min_minutes]


# - on ajoute des colonnes dérivées
def inject_derived(df):
    # TODO re-add pct columns
    # pct team
    # pct team et possession
    return df.assign(
        # passes
        passes_completed_pct=lambda x: (
            x["passes_completed"] / x["passes"] * 100
        ).round(2),
        passes_long_pct=lambda x: (
            x["passes_completed_long"] / x["passes_long"] * 100
        ).round(2),
        passes_short_pct=lambda x: (
            x["passes_completed_short"] / x["passes_short"] * 100
        ).round(2),
        passes_medium_pct=lambda x: (
            x["passes_completed_medium"] / x["passes_medium"] * 100
        ).round(2),
        challenge_tackles_pct=lambda x: (
            x["challenge_tackles"] / x["challenges"] * 100
        ).round(2),
        take_ons_won_pct=lambda x: (x["take_ons_won"] / x["take_ons"] * 100).round(2),
        shots_on_target_pct=lambda x: (x["shots_on_target"] / x["shots"] * 100).round(
            2
        ),
        # pens ratés
        pens_missed=lambda x: x["pens_att"] - x["pens_made"],
    ).assign(
        shots_on_target_pct=lambda x: x["shots_on_target_pct"].fillna(
            x["shots_on_target_pct"].mean(skipna=True)
        ),
        take_ons_won_pct=lambda x: x["take_ons_won_pct"].fillna(
            x["take_ons_won_pct"].mean(skipna=True)
        ),
        challenge_tackles_pct=lambda x: x["challenge_tackles_pct"].fillna(
            x["challenge_tackles_pct"].mean(skipna=True)
        ),
        passes_medium_pct=lambda x: x["passes_medium_pct"].fillna(
            x["passes_medium_pct"].mean(skipna=True)
        ),
        passes_short_pct=lambda x: x["passes_short_pct"].fillna(
            x["passes_short_pct"].mean(skipna=True)
        ),
        passes_long_pct=lambda x: x["passes_long_pct"].fillna(
            x["passes_long_pct"].mean(skipna=True)
        ),
    )


# - on cacule les percentile
def as_percentile(df):
    return (
        df.select_dtypes(include="number")
        .rank(pct=True, axis=0, na_option="keep")
        .add_suffix("_pctile")
    )


# - on cacule les zscore
def as_zscore(df):
    return (
        df.select_dtypes(include="number")
        .apply(lambda x: (x - x.mean()) / x.std())
        .add_suffix("_zscore")
    )


# - on cacule les minmax
def as_minmax(df):
    return (
        df.select_dtypes(include="number")
        .apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        .add_suffix("_minmax")
    )


# agg stats by player
def agg_stats(df):
    return (
        df.pipe(remove_pct_columns)
        .pipe(remove_gk_columns)
        .pipe(remove_unused_columns)
        .select_dtypes(include="number")
        .agg("sum")
        .dropna(how="all")
        .fillna(0)
    )


# agg stats by team
def agg_stats_by_team(df):
    return (
        df.pipe(remove_pct_columns)
        .pipe(remove_gk_columns)
        .pipe(remove_unused_columns)
        .drop(columns=["minutes"])
        .select_dtypes(include="number")
        .agg("sum")
        .dropna(how="all")
    )


# per 90 stats
# on supprime les pourcentages et les minutes
def as_per_90(df):
    columns_to_drop = [
        "minutes",
        *[
            c
            for c in df.select_dtypes(include="number").columns.to_list()
            if "pct" in c
        ],
    ]
    return (
        df.select_dtypes(include="number")
        .drop(columns=columns_to_drop)
        .div(df["minutes"], axis=0)
        .mul(90)
        .add_suffix("_p90")
    )


# add per 90 stats
def add_per_90(
    df,
    only_90=True,
):
    if only_90:
        return pd.concat([df[["minutes"]], as_per_90(df)], axis=1)
    else:
        return pd.concat([df, as_per_90(df)], axis=1)


# compute rankings: need to be done after per 90
def add_rankings(df_with_p90, rankings=["percentile", "zscore", "minmax"]):
    return pd.concat(
        [
            df_with_p90,
            *[
                df_with_p90.pipe(func)
                for func in [as_percentile, as_zscore, as_minmax]
                if func.__name__[3:] in rankings
            ],
        ],
        axis=1,
    )


# sort columns by name
def sort_columns(df):
    # on sort sans tenir compte des suffixes
    return df.reindex(
        sorted(
            df.columns, key=lambda x: re.sub(r"_p90|_pctile|_zscore|_minmax", "", x)
        ),
        axis=1,
    )


# each column that contains "distance" is converted to meter
def transform_distance_to_meter(df):
    return df.apply(
        lambda col: (col / 0.9144).round(0)
        if re.match(r".*_distance$", col.name)
        else col
    )


# get stats for field players only
def get_field_stats(
    players_stats,
    min_minutes=450,
    only_90=True,
    rankings=["percentile", "zscore", "minmax"],
):
    return (
        players_stats.loc[lambda x: x["keeper_stats"].isna()]
        .groupby(["player_id"])
        .apply(agg_stats)
        .pipe(inject_derived)
        .pipe(transform_distance_to_meter)
        .pipe(add_per_90, only_90=only_90)
        .pipe(filter_by_minutes, min_minutes)
        .pipe(add_rankings, rankings=rankings)
        .pipe(sort_columns)
        .reset_index()
        .round(2)
    )


def get_team_minutes(players_stats):
    return (
        players_stats.groupby(["squad_label", "match_id"], as_index=False)
        .agg({"minutes": "max"})
        .groupby("squad_label")
        .agg({"minutes": "sum"})
        .reset_index()
    )


def get_fixtures_for_team(team_label, fixtures):
    return fixtures.loc[
        lambda x: (x["squad_a_label"] == team_label)
        | (x["squad_b_label"] == team_label)
    ]


def get_opponent_stats_for_team(team_label, players_stats, fixtures):
    # get fixtures for team
    team_fixtures = get_fixtures_for_team(team_label, fixtures)

    return (
        # get stats for these fixtures
        players_stats.loc[lambda x: x["match_id"].isin(team_fixtures["match_id"])]
        # remove stats for team
        .loc[lambda x: x["squad_label"] != team_label]
        .assign(squad_label=team_label)
    )


def get_opponent_stats(players_stats, team_minutes, fixtures):
    return (
        pd.concat(
            [
                get_opponent_stats_for_team(team_label, players_stats, fixtures)
                for team_label in team_minutes["squad_label"]
            ],
            axis=0,
            ignore_index=True,
        )
        .groupby(["squad_label"])
        .apply(agg_stats_by_team)
        .pipe(inject_derived)
        .pipe(transform_distance_to_meter)
        .merge(team_minutes, on="squad_label")
        .set_index("squad_label")
        .add_suffix("_opponent")
        .drop(columns=["minutes_opponent"])
        # .rename(columns={"minutes_opponent": "minutes"})
        # .pipe(add_per_90)
        .pipe(sort_columns)
        .round(3)
        .reset_index()
    )


def compute_team_ratio(df):
    # on supprime les minutes et on ne considère que les colonnes number, sans p90 et sans opponent et sans pct
    columns_without_opp = df.columns[
        (~df.columns.str.contains("opponent"))
        & (~df.columns.str.contains("minutes"))
        & (~df.columns.str.contains("p90"))
        & (~df.columns.str.contains("pct"))
    ]
    return df.assign(
        # ratio
        **{
            column + "_ratio": (df[column] / (df[column + "_opponent"] + df[column]))
            * 100
            for column in columns_without_opp
        },
    )


def get_team_stats(
    players_stats,
    fixtures,
    only_90=True,
    rankings=["percentile", "zscore", "minmax"],
):
    # on calcule le temps de jeu de l'equipe
    team_minutes = get_team_minutes(players_stats)
    # on calcule les stats des équipes adverses
    opponent_stats = get_opponent_stats(players_stats, team_minutes, fixtures)

    return (
        players_stats.groupby(["squad_label"])
        .apply(agg_stats_by_team)
        .pipe(inject_derived)
        .pipe(transform_distance_to_meter)
        .merge(team_minutes, on="squad_label")
        .merge(opponent_stats, on="squad_label")
        .set_index("squad_label")
        .pipe(add_per_90, only_90=only_90)
        .pipe(compute_team_ratio)
        .pipe(add_rankings, rankings=rankings)
        .pipe(sort_columns)
        .round(3)
        .reset_index()
    )
