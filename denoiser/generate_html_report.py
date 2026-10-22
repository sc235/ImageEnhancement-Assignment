import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
md_path = os.path.join(current_dir, "REPORT.md")
html_path = os.path.join(current_dir, "REPORT.html")

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Modern Learned Denoisers Academic Report</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.65;
            color: #24292e;
            max-width: 1050px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #f8f9fa;
        }
        .container {
            background: #ffffff;
            padding: 45px 55px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        h1 { border-bottom: 2px solid #eaecef; padding-bottom: 12px; color: #1a1f2c; margin-top: 0; }
        h2 { border-bottom: 1px solid #eaecef; padding-bottom: 8px; margin-top: 40px; color: #2b3137; }
        h3 { margin-top: 26px; color: #3f444e; }
        h4 { margin-top: 18px; color: #495057; }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 22px 0;
            font-size: 13.5px;
        }
        th, td {
            border: 1px solid #dfe2e5;
            padding: 10px 14px;
            text-align: left;
        }
        th {
            background-color: #f1f4f8;
            font-weight: 600;
            color: #1f2328;
        }
        tr:nth-child(2n) {
            background-color: #fafbfc;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid #e1e4e8;
            margin: 15px 0 5px 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }
        .img-caption {
            font-size: 12px;
            color: #57606a;
            margin-bottom: 18px;
            display: block;
        }
        code {
            background-color: #f6f8fa;
            padding: 2px 5px;
            border-radius: 4px;
            font-family: SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 85%;
        }
        pre {
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 13px;
            border: 1px solid #e1e4e8;
        }
        blockquote {
            border-left: 4px solid #0969da;
            margin: 15px 0;
            padding: 10px 20px;
            background-color: #f0f7ff;
            color: #1f2328;
            border-radius: 0 4px 4px 0;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            background-color: #2da44e;
            color: white;
        }
        @media print {
            body { background-color: #fff; padding: 0; }
            .container { box-shadow: none; padding: 0; }
        }
    </style>
</head>
<body>
<div class="container">
"""

lines = text.split("\n")
in_table = False
in_code = False

for line in lines:
    if line.startswith("```"):
        if in_code:
            html_content += "</pre>\n"
            in_code = False
        else:
            html_content += "<pre><code>"
            in_code = True
        continue
    if in_code:
        html_content += line + "\n"
        continue

    # Tables
    if line.strip().startswith("|"):
        if not in_table:
            html_content += "<table>\n"
            in_table = True
        cols = [c.strip() for c in line.strip().split("|")[1:-1]]
        if all(re.match(r"^[:-]+$", c) for c in cols):
            continue
        is_header = any(k in line for k in ["Technique", "Condition", "Setting", "Domain", "Model", "Author"])
        tag = "th" if is_header else "td"
        html_content += "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cols) + "</tr>\n"
        continue
    else:
        if in_table:
            html_content += "</table>\n"
            in_table = False

    # Headers
    if line.startswith("# "):
        html_content += f"<h1>{line[2:]}</h1>\n"
    elif line.startswith("## "):
        html_content += f"<h2>{line[3:]}</h2>\n"
    elif line.startswith("### "):
        html_content += f"<h3>{line[4:]}</h3>\n"
    elif line.startswith("#### "):
        html_content += f"<h4>{line[5:]}</h4>\n"
    elif line.startswith("---"):
        html_content += "<hr>\n"
    elif line.startswith("> "):
        html_content += f"<blockquote>{line[2:]}</blockquote>\n"
    elif line.strip().startswith("!["):
        m = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if m:
            alt, src = m.groups()
            html_content += f'<p><img src="{src}" alt="{alt}"><span class="img-caption">Figure: {alt}</span></p>\n'
    elif line.strip():
        p = line
        p = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", p)
        p = re.sub(r"`(.*?)`", r"<code>\1</code>", p)
        p = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', p)
        html_content += f"<p>{p}</p>\n"

if in_table:
    html_content += "</table>\n"
if in_code:
    html_content += "</code></pre>\n"

html_content += """
</div>
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated HTML report at: {html_path}")
