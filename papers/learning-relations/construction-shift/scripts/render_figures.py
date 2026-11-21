#!/usr/bin/env python3
"""Replot manuscript figures from frozen derived figure data; no model inference.

Requires matplotlib. Output is PNG for easy inspection; authored SVG drafts in
figures/ are preserved. Data source equality can be checked with check_figure_data.py.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PACKAGE = Path(__file__).resolve().parents[1]
DATA = json.loads((PACKAGE / "analysis/figure_data.json").read_text(encoding="utf-8"))
OUT = PACKAGE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = ["#008c87", "#ee9d38", "#6d59a6", "#b6534d"]
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 250})

def save(fig, name):
    fig.savefig(OUT / f"replot_{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(OUT / f"replot_{name}.png")

def fig1():
    fig, ax = plt.subplots(figsize=(11, 5.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(3, 94, "Construction shift and exposure arms", fontsize=18, weight="bold")
    for i, (code, label) in enumerate([("AA","Active / Active"),("PA","Passive / Active"),("AP","Active / Passive"),("PP","Passive / Passive")]):
        x=3+i*24.5
        ax.add_patch(Rectangle((x,65),22,16,facecolor="#dcefed" if code=="AP" else "#e8eef5",edgecolor="#cad5df"))
        ax.text(x+2,75,code,fontsize=14,weight="bold"); ax.text(x+2,68,label,fontsize=9)
    ax.text(3,55,"Literal AP ↔ PA reorder: same clauses and participants; event position and query references change",fontsize=11)
    arms=[("PA-only","7 AA + 7 PA + 2 PA"),("Indirect","7 AA + 7 PA + PP + AA"),("Direct","7 AA + 7 PA + PA + AP")]
    for i,(name,sched) in enumerate(arms):
        x=3+i*32.5
        ax.add_patch(Rectangle((x,17),30,23,facecolor="#edf5f8",edgecolor="#d1dce5"))
        ax.text(x+2,31,name,fontsize=11,weight="bold");ax.text(x+2,21,sched,fontsize=10)
    ax.text(3,7,"Confounds: source co-occurrence, absolute position and sentence length may differ.",fontsize=10)
    save(fig,"fig1")

def fig2():
    fig, (ax, bx) = plt.subplots(2,1,figsize=(10.5,8),gridspec_kw={"height_ratios":[2,1.2]})
    seeds = sorted(DATA["exp109"]["primary"]["per_seed"].items())
    v = [q*100 for _,q in seeds]
    ax.scatter(v,range(len(v)),s=45,color=COLORS[0],zorder=2)
    for i,x in enumerate(v): ax.plot([0,x],[i,i],color="#badbd8",lw=3,zorder=1)
    ax.set_yticks(range(len(seeds)),[seed for seed,_ in seeds]); ax.set_xlim(0,100)
    ax.set_xlabel("Literal PA – AP C_query (percentage points)");ax.set_title("EXP-109 primary: 8 independent seed means",loc="left")
    ax.grid(axis="x",alpha=.2)
    classes=["correct","reversed","collapsed","wrong_event_name"]
    labels=["Both correct","Reversed","Same participant","Wrong event"]
    arms=[("PA-only","pa_only"),("Indirect PP+AA","indirect_pp_aa"),("Direct PA+AP","direct_pa_ap")]
    for i,(lab,arm) in enumerate(arms):
        start=0
        for k,label,color in zip(classes,labels,COLORS):
            x=DATA["exp110"]["pairClasses"][arm][k]["mean"]*100
            bx.barh(i,x,left=start,color=color,height=.66,label=label if i==0 else None)
            if x>8: bx.text(start+x/2,i,f"{x:.1f}%",color="white",ha="center",va="center",fontsize=9,weight="bold")
            start+=x
    bx.set_yticks(range(3),[a for a,_ in arms]);bx.invert_yaxis();bx.set_xlim(0,100)
    bx.set_xlabel("AP second-event pair outcomes (%)");bx.set_title("EXP-110 exploratory native four-name decomposition",loc="left")
    bx.legend(ncol=2,fontsize=8,loc="lower center",bbox_to_anchor=(.5,-.56))
    fig.tight_layout(h_pad=2.4);save(fig,"fig2")

def fig3():
    fig,ax=plt.subplots(figsize=(10.5,5.4))
    rows=[("Input position edit",DATA["exp114"]["primary"]["estimate"]),
          ("Update minus late position",DATA["exp114"]["secondary1"]["estimate"]),
          ("Reciprocal update in PA",DATA["exp114"]["secondary2"]["estimate"])]
    for i,(label,record) in enumerate(rows):
        yy=2-i
        arr=[v*100 for v in record["per_seed"].values()]
        ax.scatter(arr,[yy+.09*(j-3.5)/3.5 for j in range(len(arr))],color="#a7bdcb",s=24)
        lo,hi=[x*100 for x in record["ci95"]]
        ax.plot([lo,hi],[yy,yy],lw=4,color=COLORS[0])
        ax.scatter([record["mean"]*100],[yy],s=75,color="#182b45",zorder=5)
        ax.text(61,yy,f'{record["mean"]*100:+.1f} pp',va="center",fontsize=10)
    ax.axvline(0,c="#77869b",ls="--",lw=1)
    ax.set_yticks([2,1,0],[r[0] for r in rows])
    ax.set_xlim(-65,75);ax.set_ylim(-.7,2.7)
    ax.set_xlabel("Native question-accuracy difference (percentage points)")
    ax.set_title("EXP-114: prespecified hierarchical contrasts (8 seeds, t 95% CI)",loc="left")
    ax.grid(axis="x",alpha=.2)
    fig.tight_layout();save(fig,"fig3")

def fig4():
    fig,ax=plt.subplots(figsize=(10.5,6.6))
    modes=DATA["exp116"]["paOnlyAP_modes"]
    for group,label,color in [("passive_agent","Agent-name only",COLORS[2]),
                               ("passive_patient","Patient-name only",COLORS[1]),
                               ("passive_names","Both names",COLORS[0])]:
        rows=[modes[group+"_"+dose] for dose in ["a025","a050","a075","a100"]]
        x=[r["native_correct"]["mean"]*100 for r in rows]
        y=[r["wrong_event"]["mean"]*100 for r in rows]
        ax.plot(x,y,"o-",color=color,lw=2,label=label)
        for j,(xx,yy) in enumerate(zip(x,y)):ax.annotate([".25",".50",".75","1.0"][j],(xx,yy),xytext=(3,4),textcoords="offset points",fontsize=8)
    ax.axhline(0,color="#8e9baa",ls="--",lw=1);ax.axvline(0,color="#8e9baa",ls="--",lw=1)
    ax.set_xlabel("Change in native AP second-event question accuracy (pp)")
    ax.set_ylabel("Change in wrong-event question rate (pp; positive = more errors)")
    ax.set_title("EXP-116: exploratory specificity screen (0/12 modes qualify)",loc="left")
    ax.legend(loc="upper left");ax.grid(alpha=.18);fig.tight_layout();save(fig,"fig4")

if __name__ == "__main__":
    fig1();fig2();fig3();fig4()
