#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLATFORMS = ["AMD", "INTEL", "3090", "5060ti"]
PLATFORM_ORDER = {p:i for i,p in enumerate(PLATFORMS)}
PRACTICAL_TOLERANCE = 0.02


def roots(script_file: str | Path) -> tuple[Path, Path, Path]:
    script = Path(script_file).resolve()
    project_root = script.parents[3]
    result_dir = project_root / "ALL AUDIT" / "CONV2D" / "results"
    figure_dir = result_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return project_root, result_dir, figure_dir


def markdown_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    return "_None._" if df.empty else df.head(max_rows).to_markdown(index=False)


def load_individual(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries=[]; sessions=[]; provenance=[]
    for platform in PLATFORMS:
        result = project_root / platform / "results" / "CONV2D"
        complete = result / "ANALYSIS_COMPLETE.json"
        if not complete.is_file():
            raise RuntimeError(f"Missing completed individual analysis: {complete}")
        meta=json.loads(complete.read_text(encoding="utf-8"))
        if meta.get("status") != "PASS":
            raise RuntimeError(f"Individual analysis is not PASS: {complete}")
        s=pd.read_csv(result/"configuration_summary.csv")
        m=pd.read_csv(result/"session_medians.csv")

        # Directory name is the authoritative platform identifier.
        # In particular, pandas may otherwise infer "3090" as an integer.
        s["platform"] = str(platform)
        m["platform"] = str(platform)

        s["selected_campaign"]=meta.get("campaign")
        m["selected_campaign"]=meta.get("campaign")
        summaries.append(s); sessions.append(m)
        manifest=result/"campaign_manifest.csv"
        if manifest.is_file():
            p=pd.read_csv(manifest); p["platform"]=platform; provenance.append(p)
    return pd.concat(summaries,ignore_index=True), pd.concat(sessions,ignore_index=True), (pd.concat(provenance,ignore_index=True) if provenance else pd.DataFrame())


def build_envelopes(summary: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (platform,shape),f in summary.groupby(["platform","problem_size"]):
        for objective,col,direction in [
            ("runtime","runtime_per_op_s_median","min"),
            ("energy","total_energy_per_op_j_median","min"),
            ("edp","edp_total_j_s_median","min"),
            ("throughput","throughput_gflops_median","max"),
            ("energy_efficiency","energy_efficiency_gflop_per_j_median","max"),
        ]:
            idx=f[col].idxmin() if direction=="min" else f[col].idxmax()
            r=f.loc[idx]
            rows.append({
                "platform":platform,"platform_label":r.platform_label,"device_kind":r.device_kind,
                "energy_domain":r.energy_domain,"problem_size":int(shape),"objective":objective,
                "configuration":r.configuration,"num_threads":r.num_threads,"value":r[col],
                "selected_campaign":r.selected_campaign,
            })
    return pd.DataFrame(rows)


def cross_platform_leaders(envelopes: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (shape,obj),f in envelopes.groupby(["problem_size","objective"]):
        direction="max" if obj in {"throughput","energy_efficiency"} else "min"
        best=f.value.max() if direction=="max" else f.value.min()
        tied=f.value >= best*(1-PRACTICAL_TOLERANCE) if direction=="max" else f.value <= best*(1+PRACTICAL_TOLERANCE)
        for idx,r in f[tied].iterrows():
            rows.append({"problem_size":int(shape),"objective":obj,"platform":r.platform,
                         "platform_label":r.platform_label,"configuration":r.configuration,
                         "value":r.value,"exact_winner":bool(np.isclose(r.value,best,rtol=0,atol=0)),
                         "within_2pct":True,"energy_domain":r.energy_domain})
    return pd.DataFrame(rows)


def global_pareto(summary: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for shape,f in summary.groupby("problem_size"):
        for idx,r in f.iterrows():
            runtime=float(r.runtime_per_op_s_median)
            energy=float(r.total_energy_per_op_j_median)
            other=f.drop(index=idx)
            dominated=((other.runtime_per_op_s_median<=runtime)&(other.total_energy_per_op_j_median<=energy)&((other.runtime_per_op_s_median<runtime)|(other.total_energy_per_op_j_median<energy))).any()
            practical=((other.runtime_per_op_s_median<=runtime*(1-PRACTICAL_TOLERANCE))&(other.total_energy_per_op_j_median<=energy*(1-PRACTICAL_TOLERANCE))).any()
            rows.append({
                "platform":r.platform,"platform_label":r.platform_label,"device_kind":r.device_kind,
                "problem_size":int(shape),"configuration":r.configuration,"num_threads":r.num_threads,
                "runtime_per_op_s":runtime,"total_energy_per_op_j":energy,
                "strict_pareto":not bool(dominated),"practical_pareto_2pct":not bool(practical),
            })
    return pd.DataFrame(rows)

def pairwise_ratios(envelopes: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (shape,obj),f in envelopes.groupby(["problem_size","objective"]):
        vals={r.platform:float(r.value) for _,r in f.iterrows()}
        for a in PLATFORMS:
            for b in PLATFORMS:
                if a>=b or a not in vals or b not in vals: continue
                rows.append({"problem_size":int(shape),"objective":obj,"platform_a":a,"platform_b":b,
                             "a_over_b":vals[a]/vals[b],"b_over_a":vals[b]/vals[a]})
    return pd.DataFrame(rows)


def platform_penalties(envelopes: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (shape,obj),f in envelopes.groupby(["problem_size","objective"]):
        direction="max" if obj in {"throughput","energy_efficiency"} else "min"
        best=f.value.max() if direction=="max" else f.value.min()
        for _,r in f.iterrows():
            penalty=(best/r.value-1)*100 if direction=="max" else (r.value/best-1)*100
            rows.append({"problem_size":int(shape),"objective":obj,"platform":r.platform,
                         "configuration":r.configuration,"value":r.value,"best_value":best,
                         "penalty_vs_global_best_pct":penalty})
    return pd.DataFrame(rows)


def plot_envelopes(envelopes: pd.DataFrame, pareto: pd.DataFrame, figure_dir: Path) -> None:
    mapping=[("runtime","Best runtime per convolution [s]","best_runtime_vs_shape.png"),
             ("energy","Best measured energy per convolution [J]","best_energy_vs_shape.png"),
             ("edp","Best EDP [J s]","best_edp_vs_shape.png"),
             ("throughput","Best logical throughput [GFLOP/s]","best_throughput_vs_shape.png")]
    for obj,ylabel,name in mapping:
        fig,ax=plt.subplots(figsize=(9,5.5))
        f=envelopes[envelopes.objective==obj]
        for platform,g in f.groupby("platform"):
            g=g.sort_values("problem_size")
            ax.plot(g.problem_size,g.value,marker="o",label=platform)
        ax.set_xlabel("CONV2D shape ID"); ax.set_ylabel(ylabel); ax.set_xticks(sorted(f.problem_size.unique())); ax.grid(True,alpha=.25); ax.legend(title="Platform")
        fig.tight_layout(); fig.savefig(figure_dir/name,dpi=180); plt.close(fig)

    for shape,f in pareto[pareto.strict_pareto].groupby("problem_size"):
        fig,ax=plt.subplots(figsize=(6.5,5.5))
        for _,r in f.iterrows():
            ax.scatter(r.runtime_per_op_s,r.total_energy_per_op_j,s=70)
            ax.annotate(str(r.platform),(r.runtime_per_op_s,r.total_energy_per_op_j),xytext=(5,5),textcoords="offset points")
        ax.set_xlabel("Best runtime per convolution [s]"); ax.set_ylabel("Best measured energy per convolution [J]"); ax.grid(True,alpha=.25)
        fig.tight_layout(); fig.savefig(figure_dir/f"pareto_shape_{int(shape)}.png",dpi=180); plt.close(fig)
