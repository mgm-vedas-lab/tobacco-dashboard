import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from typing import List, Optional

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="Tobacco Use Dashboard", layout="wide")
st.title("🚭 Tobacco Use Data Analysis")

uploaded_file = st.file_uploader("📁 Upload Excel File", type=["xlsx"])


# =========================================================
# BASIC HELPERS
# =========================================================
def find_first(patterns: List[str], columns: List[str]) -> Optional[str]:
    patterns = [p.lower() for p in patterns]
    for c in columns:
        lc = c.lower()
        for p in patterns:
            if p in lc:
                return c
    return None


def pct(num: int, denom: int) -> float:
    return (num / denom * 100) if denom else 0.0


def pct_str(num: int, denom: int) -> str:
    return f"{pct(num, denom):.1f}%"


def short_demographic_summary(
    df: pd.DataFrame,
    age_col: Optional[str],
    gender_col: Optional[str],
    occupation_col: Optional[str],
) -> List[str]:
    lines = []
    if df.empty:
        return ["No data."]
    try:
        if age_col and age_col in df.columns and not df[age_col].dropna().empty:
            lines.append(
                f"Most users in age group: {df[age_col].value_counts().idxmax()}"
            )
    except Exception:
        pass
    try:
        if (
            gender_col
            and gender_col in df.columns
            and not df[gender_col].dropna().empty
        ):
            lines.append(
                f"Most common gender: {df[gender_col].value_counts().idxmax()}"
            )
    except Exception:
        pass
    try:
        if (
            occupation_col
            and occupation_col in df.columns
            and not df[occupation_col].dropna().empty
        ):
            lines.append(
                f"Top occupation: {df[occupation_col].value_counts().idxmax()}"
            )
    except Exception:
        pass
    if not lines:
        lines.append("No demographic fields available.")
    return lines


def format_bullets(lines: List[str]) -> str:
    return "\n".join([f"- {l}" for l in lines])


# =========================================================
# CHART HELPERS (used in Objectives 1–4)
# =========================================================
def responsive_pie_chart_with_legend_and_pct(labels, sizes, chart_title, colors=None):
    if not labels or not sizes:
        st.warning("No data for chart.")
        return
    if colors is None:
        colors = plt.get_cmap("tab20c")(np.linspace(0, 1, len(labels)))
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=160)
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 2 else "",
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.50),
        textprops=dict(fontsize=12, color="black"),
    )
    ax.axis("equal")
    ax.set_title(chart_title, fontsize=18)
    total = sum(sizes) if sum(sizes) else 1
    legend_labels = [
        f"{lbl} ({size} users, {size/total*100:.1f}%)"
        for lbl, size in zip(labels, sizes)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Category",
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        fontsize=11,
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def responsive_bar_chart(labels, sizes, chart_title, colors=None):
    if not labels or not sizes:
        st.warning("No data for chart.")
        return
    num_bars = len(labels)
    figsize = (max(7, num_bars * 0.55), max(5, num_bars * 0.42))
    if colors is None:
        colors = plt.get_cmap("tab20c")(np.linspace(0, 1, num_bars))
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    bars = ax.barh(labels, sizes, color=colors)
    total = sum(sizes) if sum(sizes) else 1
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_width() + max(sizes) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{sizes[i]} ({(sizes[i]/total*100):.1f}%)",
            va="center",
            fontsize=11,
            color="black",
        )
    ax.set_xlabel("Users", fontsize=11)
    ax.set_title(chart_title, fontsize=16)
    ax.tick_params(axis="y", labelsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


# =========================================================
# LESION HELPERS
# =========================================================
def _normalize_lesion_token(t: str) -> str:
    t0 = str(t or "").strip().lower()
    if t0 in {"no lesion", "nil lesion", "none"}:
        return "No lesion"
    if t0 in {"leukoplakia", "leucoplakia"}:
        return "Leukoplakia"
    if t0 == "erythroplakia":
        return "Erythroplakia"
    if t0 in {
        "oral sub-mucous fibrosis",
        "oral submucous fibrosis",
        "oral sub mucous fibrosis",
        "oral sub-mucus fibrosis",
        "osmf",
    }:
        return "Oral submucous fibrosis (OSMF)"
    if t0 in {"tobacco pouch keratosis", "tp keratosis", "tpk"}:
        return "Tobacco pouch keratosis"
    if "any other tobacco" in t0:
        return "Other tobacco-associated lesion"
    return "Other tobacco-associated lesion"


def parse_lesions_cell(cell) -> list:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return ["No lesion"]
    parts = [p.strip() for p in str(cell).split(",") if p and p.strip()]
    if not parts:
        return ["No lesion"]
    normalized = [_normalize_lesion_token(p) for p in parts]
    unique = []
    for x in normalized:
        if x not in unique:
            unique.append(x)
    if len(unique) > 1 and "No lesion" in unique:
        unique = [u for u in unique if u != "No lesion"]
    return unique or ["No lesion"]


PRIMARY_PRIORITY = {
    "Erythroplakia": 50,
    "Oral submucous fibrosis (OSMF)": 40,
    "Leukoplakia": 30,
    "Tobacco pouch keratosis": 20,
    "Other tobacco-associated lesion": 10,
    "No lesion": 0,
}


def choose_primary_lesion(lesions_list: list) -> str:
    if not lesions_list:
        return "No lesion"
    return sorted(
        lesions_list, key=lambda x: PRIMARY_PRIORITY.get(x, -1), reverse=True
    )[0]


_STAGE_MAP = {
    "precontemplation": "Precontemplation",
    "pre-contemplation": "Precontemplation",
    "contemplation": "Contemplation",
    "preparation": "Preparation",
    "action": "Action",
    "maintenance": "Maintenance",
    "maintainence": "Maintenance",
    "maintainance": "Maintenance",
}
STAGE_COLORS = {
    "Precontemplation": "#AAB2BD",
    "Contemplation": "#5DA5DA",
    "Preparation": "#60BD68",
    "Action": "#F17CB0",
    "Maintenance": "#B2912F",
}


def normalize_stage_label(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return _STAGE_MAP.get(str(x).strip().lower(), str(x).strip())


# =========================================================
# MAIN
# =========================================================
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    total_users = len(df)
    st.markdown(f"### 👥 Grand Total Users: {total_users}")

    # Detect columns
    name_col = find_first(["name"], list(df.columns)) or df.columns[0]
    age_col = find_first(["age"], list(df.columns))
    gender_col = find_first(["sex", "gender"], list(df.columns))
    occupation_col = find_first(["occupation"], list(df.columns))
    address_col = find_first(["address"], list(df.columns))

    reason_col = "16) Reasons for use of Tobacco Products"
    dep_smoking_col = "18) Severity of Nicotine Dependence (as per Fragerstrom Nicotine Dependence scale) (Smoking)"
    dep_smokeless_col = "19) Severity of Nicotine Dependence (as per Modified Fragerstrom Nicotine Dependence scale) (Smokeless)"
    stage_col = "23) Stage of behavior change"
    lesion_col = "27) Intra-oral examination"

    def classify_tobacco_use(val):
        v = str(val).lower()
        if "both" in v:
            return "both"
        if "smokeless" in v:
            return "smokeless"
        if "smoked" in v or "smoke" in v:
            return "smoked"
        return "unknown"

    tobacco_col = find_first(
        ["type of tobacco use", "tobacco use", "type of tobacco"], list(df.columns)
    )
    if not tobacco_col:
        st.error(
            "❌ 'Type of tobacco use' column not found. Please adjust column names."
        )
        st.stop()
    df["TobaccoCategory"] = df[tobacco_col].apply(classify_tobacco_use)

    # Sidebar navigation
    objective = st.sidebar.radio(
        "Go to Objective",
        [
            "Tobacco Users by Occupation",
            "Gender vs Reason Correlation",
            "Nicotine Dependence vs Stage of Behavior Change",
            "Oral Lesions vs Stage of Behavior Change",
            "Demographic Description of Users",
        ],
    )

    # Helper: short demo block
    def render_short_demo_block(df_subset, title_suffix=""):
        lines = short_demographic_summary(
            df_subset, age_col, gender_col, occupation_col
        )
        st.markdown(f"#### 🧾 Quick Demographics {title_suffix}")
        st.info(format_bullets(lines))

    # -----------------------------------------------------
    # OBJECTIVE 1
    # -----------------------------------------------------
    if objective == "Tobacco Users by Occupation":
        st.header("📊 Objective 1: Tobacco Users by Occupation")
        col1, col2, col3 = st.columns(3)
        categories = ["smoked", "smokeless", "both"]
        for category, col in zip(categories, [col1, col2, col3]):
            with col:
                subset = df[df["TobaccoCategory"] == category]
                count = len(subset)
                pct_cat = pct_str(count, total_users)
                st.markdown(f"### {category.title()} ({count}, {pct_cat})")
                if count == 0:
                    st.warning("No users.")
                    continue
                occ_groups = subset.groupby(occupation_col, dropna=False)
                occ_sorted = sorted(occ_groups, key=lambda x: len(x[1]), reverse=True)
                with st.expander("Users grouped by occupation"):
                    for occ, occ_df in occ_sorted:
                        occ_ct = len(occ_df)
                        occ_pct = pct_str(occ_ct, count)
                        with st.expander(
                            f"{occ} ({occ_ct}, {occ_pct})", expanded=False
                        ):
                            table = occ_df[
                                [name_col, age_col, address_col, occupation_col]
                            ].fillna("N/A")
                            table.columns = ["Name", "Age", "Address", "Occupation"]
                            st.dataframe(table, use_container_width=True, height=260)
                render_short_demo_block(subset, f"({category.title()} Users)")
        cat_counts = df["TobaccoCategory"].value_counts()
        responsive_pie_chart_with_legend_and_pct(
            cat_counts.index.tolist(),
            cat_counts.values.tolist(),
            "Tobacco Type Distribution",
        )

    # -----------------------------------------------------
    # OBJECTIVE 2
    # -----------------------------------------------------
    elif objective == "Gender vs Reason Correlation":
        st.header("📊 Objective 2: Gender vs Reason for Tobacco Use")
        col1, col2, col3 = st.columns(3)

        def gender_panel(label, subset, container):
            total_g = len(subset)
            with container:
                st.markdown(f"### {label} ({total_g}, {pct_str(total_g, total_users)})")
                if total_g == 0:
                    st.warning("No data.")
                    return
                if reason_col in subset.columns:
                    reason_counts = (
                        subset[reason_col].value_counts().sort_values(ascending=False)
                    )
                    with st.expander("Reasons detail"):
                        for reason, count in reason_counts.items():
                            reason_pct = pct_str(count, total_g)
                            with st.expander(f"{reason} ({count}, {reason_pct})"):
                                tbl = subset[subset[reason_col] == reason][
                                    [name_col, age_col, address_col]
                                ].fillna("N/A")
                                tbl.columns = ["Name", "Age", "Address"]
                                st.dataframe(tbl, use_container_width=True, height=240)
                    responsive_bar_chart(
                        reason_counts.index.tolist(),
                        reason_counts.values.tolist(),
                        f"{label} - Reasons",
                    )
                else:
                    st.warning("Reasons column not found.")
                render_short_demo_block(subset, f"({label})")

        male_df = (
            df[df[gender_col].str.lower() == "male"] if gender_col else df.iloc[0:0]
        )
        female_df = (
            df[df[gender_col].str.lower() == "female"] if gender_col else df.iloc[0:0]
        )
        other_df = (
            df[~df[gender_col].str.lower().isin(["male", "female"])]
            if gender_col
            else df.iloc[0:0]
        )

        gender_panel("Male", male_df, col1)
        gender_panel("Female", female_df, col2)
        gender_panel("Other / Unspecified", other_df, col3)

    # -----------------------------------------------------
    # OBJECTIVE 3
    # -----------------------------------------------------
    elif objective == "Nicotine Dependence vs Stage of Behavior Change":
        st.header("📊 Objective 3: Nicotine Dependence vs Stage of Behavior Change")
        if dep_smoking_col not in df.columns or dep_smokeless_col not in df.columns:
            st.warning("Dependence columns not found.")
        dep_choice = st.selectbox(
            "Select Dependence Type",
            [dep_smoking_col, dep_smokeless_col],
            index=0,
        )
        if dep_choice not in df.columns:
            st.error("Dependence column missing.")
            st.stop()

        valid_df = df[[dep_choice, stage_col, name_col, age_col, address_col]].dropna(
            subset=[dep_choice, stage_col]
        )
        total_dep = len(valid_df)
        st.markdown(f"**Valid dependence records:** {total_dep}")

        levels = ["Minimally", "Moderately", "Highly"]
        titles = {
            "Minimally": "Minimally Dependent",
            "Moderately": "Moderately Dependent",
            "Highly": "Highly Dependent",
        }
        c1, c2, c3 = st.columns(3)
        col_map = {"Minimally": c1, "Moderately": c2, "Highly": c3}

        for lvl in levels:
            with col_map[lvl]:
                lvl_df = valid_df[
                    valid_df[dep_choice].str.lower().str.contains(lvl.lower(), na=False)
                ]
                count = len(lvl_df)
                pct_lvl = pct_str(count, total_dep)
                st.markdown(f"### {titles[lvl]} ({count}, {pct_lvl})")
                if count == 0:
                    st.warning("No users.")
                    continue
                stage_counts = lvl_df[stage_col].value_counts()
                responsive_pie_chart_with_legend_and_pct(
                    stage_counts.index.tolist(),
                    stage_counts.values.tolist(),
                    f"{lvl} - Stage Split",
                )
                with st.expander("Stage-wise users"):
                    for stg, scount in stage_counts.items():
                        stg_pct = pct_str(scount, count)
                        with st.expander(f"{stg} ({scount}, {stg_pct})"):
                            tbl = lvl_df[lvl_df[stage_col] == stg][
                                [name_col, age_col, address_col]
                            ].fillna("N/A")
                            tbl.columns = ["Name", "Age", "Address"]
                            st.dataframe(tbl, use_container_width=True, height=240)
                render_short_demo_block(lvl_df, f"({titles[lvl]})")

        st.markdown("### Overall Dependence Population Summary")
        render_short_demo_block(valid_df, "(All with dependence & stage data)")

    # -----------------------------------------------------
    # OBJECTIVE 4
    # -----------------------------------------------------
    elif objective == "Oral Lesions vs Stage of Behavior Change":
        st.header("📊 Objective 4: Oral Lesions vs Stage of Behavior Change")

        if lesion_col not in df.columns or stage_col not in df.columns:
            st.error("Required lesion or stage column missing.")
            st.stop()

        work_df = df[[name_col, age_col, address_col, lesion_col, stage_col]].copy()
        work_df["lesions_list"] = work_df[lesion_col].apply(parse_lesions_cell)
        work_df["primary_lesion"] = work_df["lesions_list"].apply(choose_primary_lesion)
        work_df["stage_norm"] = work_df[stage_col].apply(normalize_stage_label)

        total_local = len(work_df)
        with_lesion = (work_df["primary_lesion"] != "No lesion").sum()
        pct_lesion = pct_str(with_lesion, total_local)
        st.markdown(
            f"**Users with any lesion (primary): {with_lesion} ({pct_lesion})**"
        )

        render_short_demo_block(work_df, "(Lesion Analysis Population)")

        lesion_counts = work_df["primary_lesion"].value_counts()
        responsive_bar_chart(
            lesion_counts.index.tolist(),
            lesion_counts.values.tolist(),
            "Primary Lesion Distribution",
        )

        st.markdown("### Lesion Details")
        for lesion in lesion_counts.index:
            ldf = work_df[work_df["primary_lesion"] == lesion]
            lcount = len(ldf)
            lesion_pct = pct_str(lcount, total_local)
            stage_counts = ldf["stage_norm"].value_counts()
            with st.expander(f"{lesion} ({lcount}, {lesion_pct})"):
                responsive_pie_chart_with_legend_and_pct(
                    stage_counts.index.tolist(),
                    stage_counts.values.tolist(),
                    f"{lesion} - Stage Split",
                )
                if len(stage_counts):
                    lines = [
                        f"- {stg}: {scount} ({pct_str(scount, lcount)})"
                        for stg, scount in stage_counts.items()
                    ]
                    st.markdown("\n".join(lines))
                ready_stages = ["Preparation", "Action", "Maintenance"]
                ready = ldf[ldf["stage_norm"].isin(ready_stages)]
                st.markdown(
                    f"- Ready / quitting stages: {len(ready)} ({pct_str(len(ready), lcount)})"
                )
                tbl = ldf[[name_col, age_col, address_col]].fillna("N/A")
                tbl.columns = ["Name", "Age", "Address"]
                st.dataframe(tbl, use_container_width=True, height=250)

    # -----------------------------------------------------
    # OBJECTIVE 5 (NO DISTRIBUTION TABLES/CHARTS – ONLY SUMMARIES)
    # -----------------------------------------------------
    elif objective == "Demographic Description of Users":
        st.header("📊 Objective 5: Demographic Description of Users")

        # Short snapshots for Objectives 1–4
        st.subheader("🔎 Short Demographic Snapshots (Objectives 1–4)")

        # Objective 1 snapshot
        cat_counts = df["TobaccoCategory"].value_counts()
        obj1_lines = [
            "Tobacco categories: "
            + ", ".join(
                [f"{c}={v} ({pct_str(v,total_users)})" for c, v in cat_counts.items()]
            )
        ] + short_demographic_summary(df, age_col, gender_col, occupation_col)
        st.markdown("**Objective 1 – Tobacco Users by Occupation (Summary)**")
        st.info(format_bullets(obj1_lines))

        # Objective 2 snapshot
        if gender_col in df.columns and reason_col in df.columns:
            try:
                reason_vc = df[reason_col].value_counts()
                top_reason = reason_vc.idxmax()
                top_reason_pct = pct_str(reason_vc.iloc[0], reason_vc.sum())
            except Exception:
                top_reason = "N/A"
                top_reason_pct = "0.0%"
            gender_counts = df[gender_col].value_counts()
            obj2_lines = [
                "Genders: "
                + ", ".join(
                    [
                        f"{g}={n} ({pct_str(n,total_users)})"
                        for g, n in gender_counts.items()
                    ]
                ),
                f"Top overall reason: {top_reason} ({top_reason_pct})",
            ]
        else:
            obj2_lines = ["Gender or reason column missing."]
        obj2_lines += short_demographic_summary(df, age_col, gender_col, occupation_col)
        st.markdown("**Objective 2 – Gender vs Reason (Summary)**")
        st.info(format_bullets(obj2_lines))

        # Objective 3 snapshot (FIXED: removed erroneous .str_contains, cast to string)
        obj3_lines = []
        for dep_col in [dep_smoking_col, dep_smokeless_col]:
            if dep_col in df.columns:
                subset = df[dep_col].dropna()
                if not subset.empty:
                    total_dep_rows = subset.shape[0]
                    subset_str = subset.astype(str)  # ensure string
                    mins = subset_str.str.contains(
                        "Minimally", case=False, na=False
                    ).sum()
                    mods = subset_str.str.contains(
                        "Moderately", case=False, na=False
                    ).sum()
                    highs = subset_str.str.contains(
                        "Highly", case=False, na=False
                    ).sum()
                    obj3_lines.append(
                        f"{'Smoking' if dep_col==dep_smoking_col else 'Smokeless'} Min/Mod/High: "
                        f"{mins} ({pct_str(mins,total_dep_rows)}) / "
                        f"{mods} ({pct_str(mods,total_dep_rows)}) / "
                        f"{highs} ({pct_str(highs,total_dep_rows)})"
                    )
        if not obj3_lines:
            obj3_lines = ["Dependence columns missing."]
        obj3_lines += short_demographic_summary(df, age_col, gender_col, occupation_col)
        st.markdown("**Objective 3 – Dependence vs Stage (Summary)**")
        st.info(format_bullets(obj3_lines))

        # Objective 4 snapshot
        if lesion_col in df.columns:
            temp_lesions = df[lesion_col].apply(parse_lesions_cell)
            primaries = temp_lesions.apply(choose_primary_lesion)
            lesion_counts = primaries.value_counts()
            obj4_lines = [
                "Top lesion(s): "
                + ", ".join(
                    [
                        f"{k}={v} ({pct_str(v,len(primaries))})"
                        for k, v in lesion_counts.head(3).items()
                    ]
                )
            ]
            any_lesion = (primaries != "No lesion").sum()
            obj4_lines.append(
                f"With any lesion: {any_lesion} ({pct_str(any_lesion,len(primaries))})"
            )
        else:
            obj4_lines = ["Lesion column missing."]
        obj4_lines += short_demographic_summary(df, age_col, gender_col, occupation_col)
        st.markdown("**Objective 4 – Oral Lesions vs Stage (Summary)**")
        st.info(format_bullets(obj4_lines))

        st.divider()
        # Overall demographics summary ONLY (no distributions)
        st.subheader("👥 Overall Demographics Summary (All Users)")
        overall_summary = short_demographic_summary(
            df, age_col, gender_col, occupation_col
        )
        st.info(format_bullets(overall_summary))

else:
    st.info("⬆️ Please upload an Excel (.xlsx) file to begin.")
