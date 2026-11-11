"""Theme tokens, CSS, Plotly styling and small HTML components for the dashboard."""
from __future__ import annotations
import html
import plotly.graph_objects as go
import plotly.io as pio

THEMES = {
    "Dark": dict(
        bg="#0A0F1C", surface="#111A2E", surface2="#0E1626", border="#1E2A44", text="#E6EAF2",
        muted="#8C99B2", primary="#2DD4BF", blue="#60A5FA", amber="#FBBF24", red="#F87171",
        green="#34D399", orange="#FB923C", grid="rgba(140,153,178,0.14)", hero1="#0B1730", hero2="#0F3B46",
        shadow="0 1px 0 rgba(255,255,255,0.03), 0 8px 24px rgba(0,0,0,0.35)", input="#0E1626"),
    "Light": dict(
        bg="#F4F6FA", surface="#FFFFFF", surface2="#F8FAFC", border="#E2E8F0", text="#0F172A",
        muted="#5B6B82", primary="#0D9488", blue="#2563EB", amber="#D97706", red="#DC2626",
        green="#059669", orange="#EA580C", grid="rgba(15,23,42,0.08)", hero1="#0B1E3A", hero2="#0E5A5E",
        shadow="0 1px 2px rgba(15,23,42,0.05), 0 6px 18px rgba(15,23,42,0.06)", input="#FFFFFF"),
}
STATUS_ICON = {"Healthy": "●", "Monitor": "●", "Reorder Now": "▲", "Critical Stockout Risk": "■"}


def status_colors(t):
    return {"Healthy": t["green"], "Monitor": t["amber"], "Reorder Now": t["orange"],
            "Critical Stockout Risk": t["red"]}


def colorway(t):
    return [t["primary"], t["blue"], t["amber"], t["red"], "#A3E635", "#F472B6", "#94A3B8", "#22D3EE"]


def css(t, mode):
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
:root {{ --bg:{t['bg']}; --surface:{t['surface']}; --surface2:{t['surface2']}; --border:{t['border']};
  --text:{t['text']}; --muted:{t['muted']}; --primary:{t['primary']}; --blue:{t['blue']};
  --amber:{t['amber']}; --red:{t['red']}; --green:{t['green']}; --orange:{t['orange']}; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--bg) !important; color: var(--text);
  font-family: 'Inter', system-ui, sans-serif; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: var(--surface2) !important; border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] * {{ color: var(--text); }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.01em; color: var(--text) !important; }}
p, li, label, span, div {{ color: inherit; }}
.stMarkdown, .stCaption, [data-testid="stMarkdownContainer"] {{ color: var(--text); }}
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--muted) !important; }}
/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 5px; overflow-x: auto; flex-wrap: nowrap; }}
.stTabs [data-baseweb="tab"] {{ height: 40px; padding: 0 14px; border-radius: 10px; color: var(--muted);
  font-weight: 600; font-size: 0.86rem; white-space: nowrap; background: transparent; }}
.stTabs [aria-selected="true"] {{ background: linear-gradient(135deg, {t['primary']}22, {t['blue']}22);
  color: var(--text) !important; box-shadow: inset 0 0 0 1px {t['primary']}55; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
/* inputs */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stNumberInput input, .stTextInput input, .stDateInput input,
[data-baseweb="textarea"] {{ background: {t['input']} !important; border-color: var(--border) !important; color: var(--text) !important; }}
[data-baseweb="popover"] li, [data-baseweb="menu"] {{ background: var(--surface) !important; color: var(--text) !important; }}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{ border-radius: 10px; font-weight: 600;
  border: 1px solid var(--border); background: var(--surface); color: var(--text); }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{ background: linear-gradient(135deg, {t['primary']}, {t['blue']});
  color: #04121A; border: none; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--primary); color: var(--primary); }}
.stButton > button[kind="primary"]:hover {{ filter: brightness(1.08); color:#04121A; }}
[data-testid="stExpander"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }}
[data-testid="stExpander"] summary {{ color: var(--text); }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
[data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px; }}
[data-testid="stAlert"] {{ border-radius: 12px; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--surface); border: 1px dashed var(--border); }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ box-shadow: 0 0 0 4px {t['primary']}33; }}
code {{ font-family: 'JetBrains Mono', monospace !important; }}
.katex {{ color: var(--text); font-size: 1.12em; }}
/* hero */
.hero {{ position: relative; border-radius: 20px; padding: 26px 30px 22px; margin-bottom: 18px; overflow: hidden;
  background: radial-gradient(1200px 300px at 85% -20%, {t['primary']}40, transparent 60%),
              linear-gradient(120deg, {t['hero1']} 0%, {t['hero2']} 100%);
  border: 1px solid {t['primary']}33; box-shadow: {t['shadow']}; }}
.hero:after {{ content:""; position:absolute; inset:0; background-image: linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
  linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px); background-size: 28px 28px; pointer-events:none;
  mask-image: linear-gradient(90deg, transparent, black 60%); }}
.hero .eyebrow {{ font-family:'JetBrains Mono',monospace; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: #7EF0DD; }}
.hero h1 {{ color: #F4F8FF !important; font-size: 1.85rem; margin: 6px 0 4px; line-height: 1.15; }}
.hero .sub {{ color: #B9C6DA; font-size: .95rem; max-width: 900px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; position: relative; z-index: 1; }}
.chip {{ font-size:.76rem; padding:5px 11px; border-radius:999px; background: rgba(255,255,255,0.08); color:#E8F0FF;
  border:1px solid rgba(255,255,255,0.14); backdrop-filter: blur(6px); }}
.chip b {{ color:#7EF0DD; font-weight:600; }}
.chip.live:before {{ content:""; display:inline-block; width:7px; height:7px; border-radius:50%; background:#34D399; margin-right:6px;
  box-shadow: 0 0 0 3px rgba(52,211,153,.25); }}
/* KPI cards */
.kpi-grid {{ display:grid; grid-template-columns: repeat(var(--cols,4), minmax(0,1fr)); gap:14px; margin: 4px 0 14px; }}
@media (max-width: 1100px) {{ .kpi-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
@media (max-width: 640px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} .hero h1 {{ font-size: 1.35rem; }} }}
.kpi {{ position:relative; background: var(--surface); border:1px solid var(--border); border-radius:16px; padding:16px 18px 14px;
  box-shadow: {t['shadow']}; overflow:hidden; transition: transform .15s ease, border-color .15s ease; }}
.kpi:hover {{ transform: translateY(-2px); border-color: var(--accent); }}
.kpi:before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--accent); }}
.kpi .lbl {{ font-size:.74rem; color: var(--muted); text-transform: uppercase; letter-spacing:.08em; font-weight:600;
  display:flex; justify-content:space-between; align-items:center; }}
.kpi .ico {{ width:28px; height:28px; border-radius:8px; display:grid; place-items:center; background: color-mix(in srgb, var(--accent) 16%, transparent);
  color: var(--accent); font-size: .95rem; }}
.kpi .val {{ font-family:'JetBrains Mono', monospace; font-size:1.62rem; font-weight:600; color: var(--text); margin-top:6px; line-height:1.15; }}
.kpi .val small {{ font-size:.8rem; color: var(--muted); margin-left:4px; font-family:'Inter',sans-serif; font-weight:500; }}
.kpi .foot {{ font-size:.78rem; color: var(--muted); margin-top:6px; }}
.kpi .foot .up {{ color: var(--green); font-weight:600; }} .kpi .foot .down {{ color: var(--red); font-weight:600; }}
/* section headers */
.sec {{ display:flex; align-items:baseline; gap:10px; margin: 18px 0 8px; }}
.sec h3 {{ margin:0; font-size:1.12rem; }}
.sec .tag {{ font-size:.68rem; font-family:'JetBrains Mono',monospace; padding:3px 8px; border-radius:6px; text-transform:uppercase; letter-spacing:.08em;
  background: color-mix(in srgb, var(--primary) 14%, transparent); color: var(--primary); border:1px solid color-mix(in srgb, var(--primary) 30%, transparent); }}
.sec .tag.sim {{ background: color-mix(in srgb, var(--amber) 14%, transparent); color: var(--amber); border-color: color-mix(in srgb, var(--amber) 30%, transparent); }}
.sec .tag.cfg {{ background: color-mix(in srgb, var(--blue) 14%, transparent); color: var(--blue); border-color: color-mix(in srgb, var(--blue) 30%, transparent); }}
.sec p {{ margin:0; color: var(--muted); font-size:.85rem; }}
.card {{ background: var(--surface); border:1px solid var(--border); border-radius:16px; padding:16px 18px; box-shadow: {t['shadow']}; margin-bottom: 12px; }}
.card h4 {{ margin: 0 0 8px; font-size: .98rem; }}
.reco {{ display:flex; gap:12px; padding:11px 12px; border-radius:12px; border:1px solid var(--border); background: var(--surface2); margin-bottom:8px; }}
.reco .dot {{ width:9px; height:9px; border-radius:50%; margin-top:6px; flex: 0 0 auto; }}
.reco .t {{ font-weight:600; font-size:.9rem; color: var(--text); }} .reco .d {{ font-size:.82rem; color: var(--muted); }}
.pill {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:.74rem; font-weight:600; }}
.formula {{ background: var(--surface2); border:1px dashed var(--border); border-radius:12px; padding:6px 12px; }}
.legend-row {{ display:flex; flex-wrap:wrap; gap:14px; font-size:.8rem; color: var(--muted); }}
.footer {{ margin-top: 28px; padding: 14px 4px; border-top: 1px solid var(--border); color: var(--muted); font-size: .8rem;
  display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }}
.side-brand {{ border-radius:14px; padding:12px 14px; background: linear-gradient(135deg, {t['hero1']}, {t['hero2']}); border:1px solid {t['primary']}33; margin-bottom: 8px; }}
.side-brand .n {{ color:#F4F8FF; font-family:'Space Grotesk'; font-weight:700; font-size:1.02rem; }}
.side-brand .m {{ color:#B9C6DA; font-size:.75rem; line-height:1.45; }}
</style>
"""


def kpi(label, value, foot="", icon="◆", accent=None, unit=""):
    acc = f"style='--accent:{accent}'" if accent else "style='--accent:var(--primary)'"
    u = f"<small>{html.escape(unit)}</small>" if unit else ""
    return (f"<div class='kpi' {acc}><div class='lbl'><span>{html.escape(label)}</span><span class='ico'>{icon}</span></div>"
            f"<div class='val'>{value}{u}</div><div class='foot'>{foot}</div></div>")


def kpi_row(cards, cols=4):
    return f"<div class='kpi-grid' style='--cols:{cols}'>{''.join(cards)}</div>"


def section(title, desc="", tag=None, kind=""):
    tg = f"<span class='tag {kind}'>{html.escape(tag)}</span>" if tag else ""
    d = f"<p>{desc}</p>" if desc else ""
    return f"<div class='sec'><h3>{html.escape(title)}</h3>{tg}{d}</div>"


def pill(text, color):
    return (f"<span class='pill' style='background:color-mix(in srgb,{color} 16%,transparent);color:{color};"
            f"border:1px solid color-mix(in srgb,{color} 35%,transparent)'>{html.escape(text)}</span>")


def style_fig(fig: go.Figure, t: dict, height=360, legend=True, margin=None):
    fig.update_layout(
        template="plotly_dark" if t["bg"].startswith("#0") else "plotly_white",
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=t["text"], size=12),
        colorway=colorway(t), margin=margin or dict(l=10, r=10, t=86 if legend else 48, b=10),
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["border"], font=dict(color=t["text"], family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)")
        if legend else dict(), showlegend=legend,
        title=dict(font=dict(family="Space Grotesk, Inter", size=15, color=t["text"]), x=0.01, xanchor="left", y=0.985, yanchor="top"),
    )
    fig.update_xaxes(gridcolor=t["grid"], zerolinecolor=t["grid"], linecolor=t["border"])
    fig.update_yaxes(gridcolor=t["grid"], zerolinecolor=t["grid"], linecolor=t["border"])
    return fig
