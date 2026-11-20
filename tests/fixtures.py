"""Small FICTIONAL 10-K style filings used only by the tests. The companies,
people and numbers are invented. The HTML imitates the awkward parts of real
EDGAR files: a table of contents, text split across nested spans and divs,
hidden XBRL blocks, and tables with "$" and ")" in cells of their own."""

FILLER = (
    "The company continues to invest in its operations and monitors market conditions closely. "
    "Management reviews performance each quarter and adjusts plans as circumstances change. "
)


def build_10k(company: str, ticker: str, fy: int, revenue: int, net_income: int, business: str, risk: str, mdna: str) -> str:
    prior = int(revenue * 0.9)
    return f"""<html><head><title>{ticker} 10-K</title><style>.x{{color:red}}</style></head><body>
<div style="display:none"><ix:header>HIDDEN FACT 999999 should never be indexed</ix:header></div>
<div><span>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</span></div>
<div><span>FORM 10-K</span></div>
<div><span>For the fiscal year ended December 31, {fy}</span></div>
<div><span>{company}</span></div>
<table>
<tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>7</td></tr>
<tr><td>Item 7.</td><td>Management&#8217;s Discussion and Analysis</td><td>20</td></tr>
</table>
<div><span>Item 1.</span><span>&#160;Business</span></div>
<div><div><p><span>{business}</span></p></div><p>{FILLER * 4}</p></div>
<div><span>Item 1A. Risk Factors</span></div>
<p>{risk}</p>
<p>{FILLER * 4}</p>
<div><span>Item 7. Management&#8217;s Discussion and Analysis of Financial Condition</span></div>
<p>{mdna}</p>
<p>The following table summarizes results for the years presented (in thousands).</p>
<table>
<tr><td></td><td>{fy}</td><td></td><td>{fy - 1}</td></tr>
<tr><td>Total net revenue</td><td>$</td><td><ix:nonfraction>{revenue:,}</ix:nonfraction></td><td>$</td><td>{prior:,}</td></tr>
<tr><td>Net income</td><td>$</td><td>{net_income:,}</td><td>$</td><td>({int(net_income * 0.5):,}</td><td>)</td></tr>
<tr><td>Gross margin</td><td>41.5</td><td>%</td><td>39.2</td><td>%</td></tr>
</table>
<p>{FILLER * 3}</p>
</body></html>"""


ACME_2023 = build_10k(
    "Acme Robotics, Inc.", "ACME", 2023, 1250400, 88100,
    "Acme Robotics designs and sells warehouse robots that move pallets without human drivers. "
    "Our customers include logistics companies and retailers. We sell robots, software subscriptions and maintenance.",
    "Our robots depend on lithium battery cells from a small number of suppliers. A shortage of lithium battery cells "
    "or a disruption in our supply chain could delay shipments and reduce our revenue. Changes in exchange rates may also hurt results.",
    "Total net revenue increased 11 percent to $1,250,400 thousand in fiscal 2023, driven by higher robot shipments. "
    "Net income was $88,100 thousand. We expect capital spending to rise as we open a second factory.",
)
ACME_2022 = build_10k(
    "Acme Robotics, Inc.", "ACME", 2022, 1125360, 44050,
    "Acme Robotics designs and sells warehouse robots. In 2022 we launched our first software subscription.",
    "Our robots depend on lithium battery cells from a small number of suppliers. Supply chain delays raised our costs in 2022.",
    "Total net revenue was $1,125,360 thousand in fiscal 2022. Net income was $44,050 thousand.",
)
BOREALIS_2023 = build_10k(
    "Borealis Foods Corporation", "BORE", 2023, 642700, 30500,
    "Borealis Foods produces frozen vegetables and ready meals sold in grocery stores across North America.",
    "A disruption in our supply chain, including shortages of packaging and refrigerated trucking, could reduce our revenue. "
    "Crop failures caused by weather may raise the cost of vegetables.",
    "Total net revenue increased 4 percent to $642,700 thousand in fiscal 2023. Net income was $30,500 thousand.",
)
