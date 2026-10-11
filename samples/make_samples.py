"""Generates the demo files in this folder (needs: pip install reportlab pillow python-docx).
You do not need to run this - the generated files are already committed. Run it only to regenerate them."""
import shutil
import zipfile
import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

HERE = Path(__file__).parent
W, H = letter
LEFT = 72


# ---------------------------------------------------------------- 1) messy PDF
DISCLAIMER = ("This report contains forward-looking statements that involve risks and uncertainties. Actual results may "
              "differ materially from those projected, and Acme Corp undertakes no obligation to update them.")

PAGES = [
    [("title", "Quarterly Operations Report"),
     ("h1", "1. Executive Summary"),
     ("p", ["Acme Corp delivered solid results in the third quarter. Revenue grew by twelve percent compared",
            "with the previous quarter, driven mainly by strong demand for our cloud platform and by the",
            "successful rollout of new infra-",
            "structure in three regions. Customer satisfaction also improved, reaching its highest level in",
            "two years."]),
     ("li", ["\u2022 Revenue growth of 12% quarter over quarter", "\u2022 Two new data centers opened in Asia",
             "\u2022 Support response time reduced by 30%"]),
     ("h1", "2. Financial Results"),
     ("p", ["Operating income rose to 4.2 million dollars, helped by lower hosting costs and a disciplined",
            "hiring plan. The finance team expects the trend to continue into the next quarter, although",
            "currency movements remain a risk for"])],
    [("p", ["the regions that bill in local currency. The table below shows revenue by region (in millions)."]),
     ("table", [["Region", "Q2", "Q3", "Change"], ["Asia", "4.1", "4.8", "+17%"], ["Europe", "3.9", "4.2", "+8%"],
                ["Americas", "5.0", "5.4", "+8%"], ["Total", "13.0", "14.4", "+11%"]]),
     ("p", ["Asia remained the fastest growing region, which is why the board approved further investment there."]),
     ("p", [DISCLAIMER[:98], DISCLAIMER[98:]])],
    [("h1", "3. Outlook"),
     ("p", ["We expect continued growth in the fourth quarter, supported by a healthy sales pipeline and by the",
            "launch of two new products. Hiring will focus on engineering and customer support."]),
     ("p", [DISCLAIMER[:98], DISCLAIMER[98:]]),
     ("h1", "4. Risks"),
     ("li", ["1. Supply chain delays for networking hardware", "2. Currency volatility in emerging markets"])],
]
FOOTNOTES = {1: "1 Figures are unaudited and rounded to one decimal place.",
             2: "2 Currency effects are calculated at constant exchange rates.",
             3: "3 The outlook excludes any possible acquisitions."}


def make_messy_pdf(path):
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Quarterly Operations Report"); c.setAuthor("Acme Corp Finance")
    for n, items in enumerate(PAGES, 1):
        c.setFont("Helvetica", 9)
        c.drawString(LEFT, H - 40, "ACME Corp - Quarterly Report")
        c.drawCentredString(W / 2, 28, f"Confidential - Page {n} of {len(PAGES)}")
        y = H - 100
        for kind, content in items:
            if kind == "title":
                c.setFont("Helvetica-Bold", 20); c.drawString(LEFT, y, content); y -= 34
            elif kind == "h1":
                y -= 8; c.setFont("Helvetica-Bold", 14); c.drawString(LEFT, y, content); y -= 22
            elif kind in ("p", "li"):
                c.setFont("Helvetica", 10)
                for line in content:
                    c.drawString(LEFT, y, line); y -= 13
                y -= 8 if kind == "p" else 4
            elif kind == "table":
                t = Table(content, colWidths=[110, 70, 70, 70])
                t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black), ("FONTSIZE", (0, 0), (-1, -1), 10),
                                       ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
                _, th = t.wrapOn(c, 400, 200); t.drawOn(c, LEFT, y - th); y -= th + 14
        c.setLineWidth(0.5); c.line(LEFT, 110, LEFT + 120, 110)
        c.setFont("Helvetica", 8); c.drawString(LEFT, 98, FOOTNOTES[n])
        c.showPage()
    c.save()


# ---------------------------------------------------------------- 2) scanned image + scanned PDF
def make_scan(png_path, pdf_path):
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    random.seed(7)
    img = Image.new("L", (1000, 520), 245)
    d = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 46); body = ImageFont.truetype("DejaVuSans.ttf", 29)
    except OSError:
        big = ImageFont.load_default(46); body = ImageFont.load_default(29)
    d.text((60, 40), "Delivery Note", font=big, fill=20)
    for i, line in enumerate(["Order number 10254 was shipped on 12 March 2026.",
                              "Total amount due: $1,250.00 for 24 units.",
                              "Please contact support if the delivery is late.",
                              "Thank you for choosing Acme Logistics."]):
        d.text((60, 140 + i * 62), line, font=body, fill=30)
    for _ in range(2500):
        x, y = random.randrange(1000), random.randrange(520); d.point((x, y), fill=random.randrange(120, 200))
    img = img.rotate(0.8, fillcolor=245).filter(ImageFilter.GaussianBlur(0.7))
    img.save(png_path)
    img.convert("RGB").save(pdf_path, "PDF", resolution=150)


# ---------------------------------------------------------------- 3) DOCX v1 / v2 (with header, footer, table, footnotes)
FOOTNOTES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>
<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r><w:continuationSeparator/></w:r></w:p></w:footnote>
{notes}</w:footnotes>"""


def add_footnotes(path, notes):
    xml = FOOTNOTES_XML.format(notes="".join(
        f'<w:footnote w:id="{i}"><w:p><w:r><w:t>{t}</w:t></w:r></w:p></w:footnote>' for i, t in enumerate(notes, 1)))
    tmp = Path(str(path) + ".tmp")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.decode().replace("</Types>", '<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/></Types>').encode()
            if item.filename == "word/_rels/document.xml.rels":
                data = data.decode().replace("</Relationships>", '<Relationship Id="rIdFn1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes" Target="footnotes.xml"/></Relationships>').encode()
            zout.writestr(item, data)
        zout.writestr("word/footnotes.xml", xml)
    shutil.move(tmp, path)


def make_handbook(path, version):
    import docx
    d = docx.Document()
    d.core_properties.title = "Employee Handbook"; d.core_properties.author = "Acme HR"
    d.sections[0].header.paragraphs[0].text = "Acme HR - Internal use only"
    d.sections[0].footer.paragraphs[0].text = "Employee Handbook"
    d.add_paragraph("Employee Handbook", style="Title")
    d.add_heading("Leave Policy", 1)
    d.add_paragraph("All full-time employees are entitled to paid leave. Leave must be requested through the HR portal "
                    "at least two weeks in advance, except in emergencies.")
    d.add_heading("Annual Leave", 2)
    days = "20" if version == 1 else "25"
    d.add_paragraph(f"Employees receive {days} days of annual leave per calendar year. Unused days can be carried over "
                    "into the next year, up to a maximum of five days.")
    d.add_paragraph("Public holidays are not counted as annual leave.", style="List Bullet")
    d.add_paragraph("Leave balances are shown in the HR portal.", style="List Bullet")
    t = d.add_table(rows=3, cols=3); t.style = "Table Grid"
    for i, row in enumerate([["Seniority", "Extra days", "Notes"], ["0-2 years", "0", "Standard"], ["3+ years", "2", "Added to annual leave"]]):
        for j, val in enumerate(row):
            t.cell(i, j).text = val
    t.cell(1, 2).merge(t.cell(2, 2)) if version == 2 else None
    if version == 2:
        d.add_heading("Parental Leave", 2)
        d.add_paragraph("Parents are entitled to twelve weeks of paid parental leave, which can be taken within the first "
                        "year after the birth or adoption of a child.")
    d.add_heading("Remote Work", 1)
    d.add_paragraph("Employees may work remotely up to three days per week with manager approval. Remote workers must be "
                    "reachable during core hours and use the company VPN.")
    if version == 1:
        d.add_paragraph("Home office equipment is reimbursed up to 200 dollars per year.")
    d.save(path)
    add_footnotes(path, ["Leave year runs from 1 January to 31 December.", "Core hours are 10:00 to 16:00 local time."])


# ---------------------------------------------------------------- 4) HTML page with navigation, cookie banner, footnotes
HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Acme Support FAQ</title>
<meta name="author" content="Acme Support"><script>var tracking = "do not index me";</script>
<style>body{font-family:sans-serif}</style></head>
<body>
<nav><a href="/">Home</a> <a href="/products">Products</a> <a href="/contact">Contact</a></nav>
<div class="cookie-banner">We use cookies to improve your experience. Accept all cookies?</div>
<main><article>
<h1>Acme Support FAQ</h1>
<h2>Shipping</h2>
<p>Orders are shipped within two business days.<sup>1</sup> Express delivery is available for an extra fee.</p>
<table><caption>Delivery times</caption>
<tr><th>Region</th><th>Standard</th><th>Express</th></tr>
<tr><td>Domestic</td><td>3-5 days</td><td>1-2 days</td></tr>
<tr><td>International</td><td>7-14 days</td><td>3-5 days</td></tr></table>
<h2>Returns</h2>
<p>You can return any item within 30 days of delivery.<sup>2</sup> Items must be unused and in their original packaging.</p>
<ul><li>Print the return label from your account page</li><li>Pack the item securely</li><li>Drop it at any partner location</li></ul>
</article></main>
<section class="footnotes"><ol><li id="fn1">Business days exclude weekends and public holidays.</li><li id="fn2">Sale items can be returned for store credit only.</li></ol></section>
<footer>Copyright 2026 Acme Corp. All rights reserved. Privacy Policy | Terms</footer>
</body></html>"""


if __name__ == "__main__":
    make_messy_pdf(HERE / "messy_report.pdf")
    shutil.copy(HERE / "messy_report.pdf", HERE / "copy_of_messy_report.pdf")          # byte-identical copy -> duplicate demo
    (HERE / "broken.pdf").write_bytes((HERE / "messy_report.pdf").read_bytes()[:700])   # truncated -> parsing failure demo
    make_scan(HERE / "scan_note.png", HERE / "scanned_report.pdf")
    make_handbook(HERE / "handbook_v1.docx", 1)
    make_handbook(HERE / "handbook_v2.docx", 2)
    (HERE / "faq_page.html").write_text(HTML, encoding="utf-8")
    print("samples written to", HERE)
