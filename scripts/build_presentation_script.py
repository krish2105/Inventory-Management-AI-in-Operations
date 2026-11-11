"""Builds the spoken presentation script (PDF) for the Session II lab presentation."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Krishna_Mathur_AS25DXB018_Presentation_Script.pdf"
NAVY, TEAL, AMBER, MUTED = colors.HexColor("#0B1E3A"), colors.HexColor("#0D9488"), colors.HexColor("#B45309"), colors.HexColor("#475569")

H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=NAVY, spaceAfter=4)
SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=10, leading=14, textColor=MUTED)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=TEAL, spaceBefore=12, spaceAfter=3)
TIME = ParagraphStyle("time", fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=4)
SAY = ParagraphStyle("say", fontName="Helvetica", fontSize=11.2, leading=16.5, textColor=colors.HexColor("#0F172A"), spaceAfter=7)
DO = ParagraphStyle("do", fontName="Helvetica-Bold", fontSize=9.5, leading=13, textColor=AMBER, spaceAfter=6, leftIndent=6,
                    borderPadding=(4, 6, 4, 6), backColor=colors.HexColor("#FEF3C7"))
TIP = ParagraphStyle("tip", fontName="Helvetica-Oblique", fontSize=9, leading=12.5, textColor=MUTED, spaceAfter=6)
SMALL = ParagraphStyle("small", fontName="Helvetica", fontSize=9.3, leading=12.5)

story = []
def h2(t, time=None):
    story.append(Paragraph(t, H2))
    if time:
        story.append(Paragraph(time, TIME))
def say(*ps):
    for p in ps:
        story.append(Paragraph(p, SAY))
def do(t):
    story.append(Paragraph("&#9654; DO: " + t, DO))
def tip(t):
    story.append(Paragraph(t, TIP))

story += [Paragraph("Presentation Script — AI in Operations, Session II Lab", H1),
          Paragraph("AI-Powered Inventory Intelligence &amp; Reorder Point Optimization<br/>"
                    "Krishna Mathur · AS25DXB018 · MAIB September 2025 · SP Jain School of Global Management · Instructor: Dr. Sandip Kumar Roy", SUB),
          Spacer(1, 6)]
t = Table([["Total time", "≈ 8–9 minutes + questions"],
           ["Have open before you start", "1) Streamlit dashboard (Before vs After AI tab)   2) GitHub repo   3) the one-page PDF"],
           ["How to read this", "Normal text = what you say. Orange box = what you click. Grey italics = delivery tip."]],
          colWidths=[45 * mm, 125 * mm])
t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 9), ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
                       ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E6FFFA")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                       ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
story.append(t)

h2("1. Opening", "≈ 30 seconds")
say("Good morning, sir, and good morning everyone. I'm Krishna Mathur.",
    "For the Session II lab, the task was to take the basic reorder-point calculator and make it smarter. "
    "The slide asked for three things: machine-learning demand forecasting, dynamic safety stock with Monte Carlo simulation, "
    "and Bayesian optimisation instead of manual parameter sweeps.",
    "I built all three, and then I asked myself one honest question: <b>what does AI actually add here?</b> "
    "The reorder-point formula has been around for decades, long before AI. So in the next few minutes I'll show you "
    "the old way and the AI way side by side, on real data, and let the numbers speak.")
tip("Smile, slow down, look at the class, not the screen. Pause after the question “what does AI actually add here?”")

h2("2. The problem statement", "≈ 1 minute")
say("Let me start with the problem, because every business that holds stock has it.",
    "If you order too late, you run out. The customer leaves and you lose the sale. "
    "If you order too early or too much, your money sits on a shelf. So the real question is: "
    "<b>when should I reorder, and how much should I keep as a buffer?</b>",
    "The classic answer is the formula on the slide: safety stock equals Z times the square root of "
    "sigma-D squared times L, plus D-bar squared times sigma-L squared. Then reorder point equals average demand "
    "times lead time, plus that safety stock.",
    "The formula itself is fine. The problem is how it's usually used. You take an average of past demand, calculate it once, "
    "use one service level for every product, and never look at it again. Demand changes, seasons come, suppliers get late, "
    "and the number stays frozen. That's the gap I tried to close.")

h2("3. The dataset", "≈ 45 seconds")
say("I didn't want to use made-up numbers, so I used a real public dataset: <b>Online Retail II</b> from the UCI Machine Learning "
    "Repository. It's a UK online gift wholesaler with about <b>1.07 million real invoice lines</b>, from December 2009 to December 2011. "
    "That's two full years, so we can actually see seasonality.",
    "I cleaned it: removed cancellations, returns, non-product codes like postage, and a few extreme bulk orders. "
    "Then I kept the <b>40 best-selling products</b> that sold on at least 400 days, and turned that into daily demand.",
    "One thing I want to be transparent about: this dataset has sales, but it does <b>not</b> have stock levels, supplier lead times or promotions. "
    "So lead time and costs are settings you can change with sliders, and stock-on-hand is a simulated scenario. "
    "The dashboard labels clearly what is real data and what is an assumption. I didn't fake anything as real.")

h2("4. How I solved it — the three workstreams", "≈ 2 minutes")
story.append(Paragraph("<b>Workstream 1 — Demand forecasting</b>", SMALL)); story.append(Spacer(1, 3))
do("Open tab “1 · Demand Forecasting”. Point at the leaderboard and the forecast chart.")
say("For forecasting I compared seven models: simple ones like naive and moving average, classic time-series models like ETS and ARIMA, "
    "and machine-learning models: gradient boosting and a neural network.",
    "The important part is <i>how</i> they're tested. I never let a model see the future. The last 8 weeks are a locked test set, "
    "and the model is chosen on rolling validation before that. For our top product, the best model's error was about 32%, "
    "compared with 43% for the naive method.",
    "The slide mentioned LSTM. I looked at it honestly. With only about 105 weeks per product, an LSTM would just overfit, "
    "so I used a smaller neural network instead and compared it fairly.")
story.append(Paragraph("<b>Workstream 2 — Dynamic safety stock and Monte Carlo</b>", SMALL)); story.append(Spacer(1, 3))
do("Open tab “2 · Safety Stock &amp; Monte Carlo”. Move the service-level slider once, then switch to “Monte Carlo risk simulation”.")
say("Here is the exact formula from the slide, with live numbers. When I move this slider, everything recalculates.",
    "But a formula assumes things: normal demand, and lead time independent of demand. Real life isn't that neat. "
    "So I ran a Monte Carlo simulation with 100,000 scenarios. When the assumptions hold, simulation and formula agree. "
    "That's a good check that my maths is right. But when supplier delays happen together with busy periods, "
    "the formula says 2,930 units and the simulation says you actually need about 3,326. "
    "<b>In that case the formula under-protects you</b>, and only simulation shows it.")
story.append(Paragraph("<b>Workstream 3 — Bayesian optimisation</b>", SMALL)); story.append(Spacer(1, 3))
do("Open tab “3 · Bayesian Optimization”. Click “Run Bayesian optimisation” — it takes a few seconds.")
say("The slide said: don't do manual parameter sweeps, use Bayesian optimisation. So instead of trying hundreds of combinations, "
    "the optimiser builds a model of the cost surface and decides smartly where to test next.",
    "It looks for the cheapest reorder point and order quantity while respecting real limits: minimum fill rate, warehouse capacity, "
    "budget and minimum order quantity. On our main product it cut annual cost by about <b>18%</b> in just 40 tries, "
    "and I re-checked that on a completely fresh simulation so it isn't luck.")
do("Scroll down and click “Run manual sweep” to show the old way needs 100 evaluations.")

h2("5. Before AI vs After AI — the main result", "≈ 1.5 minutes")
do("Go to the first tab, “Before vs After AI”. Point at the KPI cards, then the inventory replay chart.")
say("This is the part I'm most proud of, because it answers the question: what does AI add?",
    "I took the last 26 weeks of <b>real</b> demand, including the Christmas rush, and replayed it day by day for all 40 products. "
    "Same suppliers, same costs. The only thing that changes is how decisions are made.",
    "<b>Before AI</b>: the textbook way — one average, one service level, set once.",
    "<b>After AI</b>: the forecast retrains every week, it notices when it's drifting and switches models, it corrects its own bias, "
    "it learns its buffer from its own past mistakes, and it picks a service level based on cost.",
    "The result: <b>23% fewer lost sales, 38% fewer stockout days</b>, fill rate up from 98.7 to 99 percent, "
    "forecast error down from 52 to 41 percent, and total cost about 2.4% lower.",
    "And to be honest with you: the AI holds about 12% more stock on average, and it's cheaper on 19 of the 40 products, not all of them. "
    "I've kept that visible on purpose. AI is not magic. It's better decisions, with trade-offs you can see.")
tip("This is your strongest moment. Slow down on the numbers. Pause after “what does AI add?”")

h2("6. Automation — the AI Autopilot", "≈ 1 minute")
do("Open “AI Autopilot”. Point at the 8-step pipeline, then the agent decisions table, then click “Exception digest”.")
say("Sir also said the system needs automation, so I built an Autopilot. It runs by itself whenever new data comes in.",
    "It goes step by step: take in the data, detect unusual demand with an Isolation Forest, check for forecast drift and retrain, "
    "forecast, group products by value and variability, simulate risk, then decide.",
    "For every product the agent says: <b>expedite, order, watch or hold</b>, and writes the reason in plain English. "
    "It drafts the purchase orders, but a human approves them. I think that's the right balance. "
    "Finally it writes a daily exception report, so a manager reads one page instead of checking 40 products by hand.")
do("Optional: open “Data &amp; Reports”, upload the CSV template, and show that everything refreshes.")

h2("7. Closing", "≈ 30 seconds")
say("So, to summarise: I took a static calculator from before the AI era and turned it into a system that "
    "<b>learns, monitors itself, optimises and acts</b>, on real data, with the three techniques from the slide.",
    "Everything is reproducible: the code is on GitHub, the notebook is fully executed, and there are 18 automated tests checking the maths. "
    "I also followed the rule on the slide: this is individual work, and no data was shared with peers.",
    "Thank you. I'm happy to take any questions.")

story.append(PageBreak())
h2("Numbers cheat-sheet (glance, don't read)")
rows = [["What", "Number"],
        ["Dataset", "UCI Online Retail II · 1,067,371 invoice lines · Dec 2009 – Dec 2011 · 40 SKUs"],
        ["Best forecast (top SKU 85099B)", "ARIMA/ETS ≈ 32% WAPE vs naive 43%"],
        ["Formula SS / ROP (top SKU)", "SS 1,202 · ROP 2,930 units (α 95%, 14 ± 4 days)"],
        ["Monte Carlo ROP, correlated case", "≈ 3,326 → formula under-protects (stockout 9.2% vs 5%)"],
        ["Bayesian optimisation", "−18.3% annual cost, 40 evaluations, all constraints met"],
        ["Before → After AI (26 weeks)", "Lost sales −23% · stockout days 53 → 33 · fill 98.7% → 99.0%"],
        ["", "WAPE 52% → 41% · cost −2.4% · avg stock +12% · cheaper on 19/40 SKUs"],
        ["Automation", "295 auto-retrains (92 drift-triggered) · Isolation Forest anomalies · agent drafts POs"],
        ["Quality", "18 automated tests · executed notebook · one-page PDF"]]
t = Table(rows, colWidths=[55 * mm, 115 * mm])
t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 9), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                       ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                       ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                       ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
story.append(t)

h2("Likely questions — short, honest answers")
qa = [("Where is the AI? Isn't this just a formula?",
       "The formula is the old part. The AI is what decides its inputs and acts on them: ML forecasts that retrain and switch models on drift, "
       "anomaly detection, clustering, conformal buffers learned from its own errors, Bayesian optimisation and an agent that drafts orders. "
       "The Before vs After tab measures exactly that difference."),
      ("Why no LSTM?",
       "About 105 weekly points per product is far too little for an LSTM; it would overfit. Competitions like M5 show simple models win on short retail series. "
       "I used a neural network (MLP) instead and let validation decide."),
      ("Your stock and lead times are not real, are they?",
       "Correct, and I say so on screen. The public data has no inventory or supplier fields, so those are labelled assumptions with sliders. "
       "The demand is 100% real."),
      ("Why does the AI hold more stock?",
       "Running out costs more than holding stock in this cost setup, so the AI protects service. That's a deliberate trade-off, "
       "and the optimiser can be given a budget or capacity limit to control it."),
      ("Why isn't the AI cheaper on every product?",
       "For some steady products the old average was already good enough. I kept those results visible instead of hiding them."),
      ("What is Bayesian optimisation, in simple words?",
       "It's a smart search: it learns from each try what the cost landscape looks like and picks the next try where it expects the biggest improvement, "
       "so it finds a good answer in about 40 tries instead of hundreds."),
      ("What is Monte Carlo here?",
       "I simulate 100,000 possible futures of demand and supplier delay, then read off how much stock covers 95% of them."),
      ("How does it handle new data?",
       "Upload a CSV in Data &amp; Reports. It's validated, duplicates are reconciled, models retrain and every number refreshes. It's on-demand batch, not a live feed."),
      ("Is it real-time?",
       "Not a live stream. It refreshes automatically every time new data arrives, which is how most planning systems run in practice.")]
for q, a in qa:
    story.append(KeepTogether([Paragraph(f"<b>Q: {q}</b>", SMALL), Spacer(1, 2), Paragraph(a, ParagraphStyle("a", parent=SMALL, textColor=MUTED, leftIndent=10)), Spacer(1, 7)]))

h2("Delivery tips")
for t_ in ["Speak to the class, not the laptop. Glance at this sheet only between sections.",
           "Use “I built… I tested… I found…”. It's your work, so own it.",
           "If a chart takes a second to load, keep talking. Don't apologise.",
           "If you don't know an answer: “Good question. I didn't test that, but here's how I would…”",
           "Finish on time. 8–9 minutes is perfect; it leaves room for questions."]:
    story.append(Paragraph("• " + t_, SMALL)); story.append(Spacer(1, 3))

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "Krishna Mathur · AS25DXB018 · AI in Operations — Session II presentation script")
    c.drawRightString(192 * mm, 10 * mm, f"Page {d.page}"); c.restoreState()

SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                  title="Presentation Script — Krishna Mathur", author="Krishna Mathur").build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
