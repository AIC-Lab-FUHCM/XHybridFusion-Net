from pathlib import Path
import math
import xml.etree.ElementTree as ET
from html import escape
from PIL import Image, ImageDraw, ImageFont


# One layout produces editable draw.io cells, an SVG and a PNG preview.
OUT = Path(__file__).resolve().parent
W, H = 640, 865
nodes = []
edges = []


def box(key, text, x, y, w, h, fill="#ffffff", kind="box", size=19):
    nodes.append(dict(key=key, text=text, x=x, y=y, w=w, h=h,
                      fill=fill, kind=kind, size=size))


def edge(source, target, points, arrow=True):
    edges.append((source, target, points, arrow))


box("boundary", "", 12, 132, 614, 610, "#ffffff", "boundary")
box("ca_panel", "", 76, 308, 158, 295, "#d6e6f5", "panel")
box("sa_panel", "", 383, 210, 174, 383, "#fbe0cf", "panel")
box("input1_label", "From\nDenseNet121", 35, 15, 140, 48, kind="text", size=18)
box("input2_label", "From\nEfficientNetB0", 150, 15, 150, 48, kind="text", size=18)
for key,x,y,color in [("in1",70.5,77,"#7399d5"),("in2",190.5,77,"#e6effa")]:
    # A thin 3D feature plate, with connected front and right faces.
    box(key, "", x, y, 70, 27, color, "inputplate")
box("out_back", "", 435, 774, 70, 25, "#42679e", "slab")
box("out", "", 435, 767, 70, 25, "#7399d5", "slab")
box("x1", "x₁", 39, 77, 28, 27, kind="text", size=23)
box("x2", "x₂", 264, 77, 28, 27, kind="text", size=23)
box("x3", "x₃", 405, 765, 28, 27, kind="text", size=23)
box("concat", "C", 138, 150, 34, 34, "#ffffff", "ellipse", size=23)
box("concat_label", "Concat", 225, 151, 70, 30, kind="text", size=17)
box("proj", "1×1 Conv\n(C₁ + C₂) → 2n\nBN + ReLU", 91, 203, 128, 70, "#fff0cd", size=17)
box("f1", "", 152, 288, 6, 6, "#555555", "dot")
box("f1_label", "F₁", 107, 279, 30, 25, kind="text", size=24)
box("fc1", "2n → n/2", 94, 333, 122, 45, "#d6e6f5", "trapdown", size=22)
box("relu1", "ReLU", 104, 403, 102, 33, "#e6f1fa", size=21)
box("fc2", "n/2 → 2n", 94, 516, 122, 45, "#d6e6f5", "trapup", size=22)
box("sig1", "σ", 139, 577, 32, 32, "#edf6fc", "ellipse", size=25)
box("ca_label", "Channel Attention", 15, 614, 125, 25, kind="text", size=15)
box("mul1", "×", 139, 645, 32, 32, "#ffffff", "ellipse", size=30)
box("f2_label", "F₂", 103, 648, 29, 26, kind="text", size=24)
box("conv1", "7×7 Conv\n2n → n/2", 399, 226, 142, 53, "#fbe0cf", size=20)
box("bnrelu", "BN + ReLU", 399, 301, 142, 35, "#fbe0cf", size=20)
box("conv2", "7×7 Conv\nn/2 → 2n", 399, 417, 142, 53, "#fbe0cf", size=20)
box("bn2", "BN", 399, 493, 142, 35, "#fbe0cf", size=21)
box("sig2", "σ", 454, 552, 32, 32, "#fff0e7", "ellipse", size=25)
box("sa_label", "Spatial Attention", 310, 614, 145, 25, kind="text", size=17)
box("mul2", "×", 454, 645, 32, 32, "#ffffff", "ellipse", size=30)
box("f3_label", "F₃", 419, 648, 28, 26, kind="text", size=24)
box("tokens", "Flatten + Transpose", 381, 695, 178, 35, "#fff0cd", size=18)
box("legend", "C: concatenation     σ: sigmoid     ×: element-wise product", 20, 833, 600, 26, kind="text", size=16)
box("config", "n = 128, 2n = 256", 25, 751, 280, 25, kind="text", size=17)
box("perm_note", "Linear acts on channels\nvia NHWC permutation", 25, 780, 260, 40, kind="text", size=14)

edge("in1", "concat", [(95,104),(95,167),(138,167)])
edge("in2", "concat", [(215,104),(215,167),(172,167)])
edge("concat", "proj", [(155,184),(155,203)])
edge("proj", "f1", [(155,273),(155,291)], False)
edge("f1", "fc1", [(155,291),(155,333)])
edge("f1", "mul1", [(155,291),(268,291),(268,661),(171,661)])
for a,b,ys,yt in [("fc1","relu1",378,403),("relu1","fc2",436,516),("fc2","sig1",561,577),("sig1","mul1",609,645)]:
    edge(a,b,[(155,ys),(155,yt)])
edge("mul1", "conv1", [(155,677),(155,714),(317,714),(317,190),(470,190),(470,226)])
edge("mul1", "mul2", [(155,677),(155,714),(317,714),(317,190),(595,190),(595,661),(486,661)])
for a,b,ys,yt in [("conv1","bnrelu",279,301),("bnrelu","conv2",336,417),("conv2","bn2",470,493),("bn2","sig2",528,552),("sig2","mul2",584,645),("mul2","tokens",677,695),("tokens","out",730,767)]:
    edge(a,b,[(470,ys),(470,yt)])

# Close the removed dropout rows in both columns and their bypass routes.
def compact_y(y):
    if y <= 445:
        return y
    return y - 65 if y >= 510 else 445

# The spatial dropout row was higher than the channel dropout row.
for node in nodes:
    if node["key"] in {"conv2", "bn2", "sig2"}:
        node["y"] -= 65
    elif node["key"] == "sa_panel":
        node["h"] -= 65
for i, (source, target, points, arrow) in enumerate(edges):
    if source in {"bnrelu", "conv2", "bn2", "sig2"}:
        points = [(x, y - 65 if (source != "bnrelu" or j > 0) and not (target == "mul2" and j == len(points)-1) else y)
                  for j, (x, y) in enumerate(points)]
        edges[i] = (source, target, points, arrow)
for node in nodes:
    if node["key"] in {"conv2", "bn2", "sig2", "sa_panel"}:
        continue
    bottom = compact_y(node["y"] + node["h"])
    node["y"] = compact_y(node["y"])
    node["h"] = bottom - node["y"]
for i, (source, target, points, arrow) in enumerate(edges):
    if source in {"bnrelu", "conv2", "bn2", "sig2"}:
        if target == "mul2":
            points[-1] = (points[-1][0], compact_y(points[-1][1]))
    else:
        points = [(x, compact_y(y)) for x, y in points]
    edges[i] = (source, target, points, arrow)
H -= 65

mxfile = ET.Element("mxfile", host="app.diagrams.net", type="device")
diagram = ET.SubElement(mxfile, "diagram", name="GAM Fusion", id="gam-code")
model = ET.SubElement(diagram, "mxGraphModel", grid="1", gridSize="10", page="1",
                      pageScale="1", pageWidth=str(W), pageHeight=str(H), arrows="1")
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", id="0")
ET.SubElement(root, "mxCell", id="1", parent="0")
by_id = {n["key"]: n for n in nodes}
for n in nodes:
    style = f'rounded=1;whiteSpace=wrap;html=0;fontFamily=Times New Roman;fontSize={n["size"]};fillColor={n["fill"]};strokeColor=#253b56;strokeWidth=1.5;'
    if n["kind"] == "text":
        style += "fillColor=none;strokeColor=none;"
    elif n["kind"] == "boundary":
        style += "dashed=1;dashPattern=1 2;strokeWidth=2;arcSize=20;"
    elif n["kind"] == "panel":
        style += "strokeColor=none;arcSize=12;"
    elif n["kind"] in {"ellipse", "dot"}:
        style += "shape=ellipse;"
    elif n["kind"] in {"trapup", "trapdown"}:
        style += "shape=trapezoid;rounded=0;" + ("flipV=1;" if n["kind"] == "trapdown" else "")
    elif n["kind"] == "inputplate":
        # Embedded stencil keeps the complete plate as one editable draw.io shape.
        import base64
        import zlib
        from urllib.parse import quote
        stencil = '<shape w="70" h="27" aspect="variable" strokewidth="inherit"><connections><constraint x="0.5" y="1" perimeter="0"/></connections><background><path><move x="0" y="22"/><line x="21" y="0"/><line x="70" y="0"/><line x="70" y="5"/><line x="49" y="27"/><line x="0" y="27"/><close/></path></background><foreground><fillstroke/><path><move x="0" y="22"/><line x="49" y="22"/><line x="70" y="0"/><move x="49" y="22"/><line x="49" y="27"/></path><stroke/></foreground></shape>'
        compressor = zlib.compressobj(wbits=-15)
        compressed = compressor.compress(quote(stencil, safe="").encode()) + compressor.flush()
        encoded = base64.b64encode(compressed).decode()
        style += f"shape=stencil({encoded});rounded=0;"
    elif n["kind"] == "slab":
        style += "shape=parallelogram;rounded=0;"
    cell = ET.SubElement(root, "mxCell", id=n["key"], value=n["text"], style=style, vertex="1", parent="1")
    ET.SubElement(cell, "mxGeometry", x=str(n["x"]), y=str(n["y"]), width=str(n["w"]), height=str(n["h"]), **{"as":"geometry"})
for i, (source, target, points, arrow) in enumerate(edges):
    a, b = by_id[source], by_id[target]
    sx, sy = points[0]
    tx, ty = points[-1]
    style = f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;strokeColor=#465462;strokeWidth=1.7;endArrow={"block" if arrow else "none"};endFill=1;'
    style += f'exitX={(sx-a["x"])/a["w"]};exitY={(sy-a["y"])/a["h"]};exitPerimeter=0;entryX={(tx-b["x"])/b["w"]};entryY={(ty-b["y"])/b["h"]};entryPerimeter=0;'
    cell = ET.SubElement(root, "mxCell", id=f"edge{i}", source=source, target=target, edge="1", parent="1", style=style)
    geom = ET.SubElement(cell, "mxGeometry", relative="1", **{"as":"geometry"})
    arr = ET.SubElement(geom, "Array", **{"as":"points"})
    for x,y in points[1:-1]:
        ET.SubElement(arr, "mxPoint", x=str(x), y=str(y))
ET.indent(mxfile)
ET.ElementTree(mxfile).write(OUT / "GAM_Fusion.drawio", encoding="utf-8", xml_declaration=True)

# Draw the same primitives for a standalone, high-resolution preview.
scale = 2
image = Image.new("RGB", (W*scale, H*scale), "white")
draw = ImageDraw.Draw(image)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>']
def shape(n):
    x,y,w,h = [n[k] for k in ("x","y","w","h")]
    if n["kind"] == "text": return
    coords = (x*scale,y*scale,(x+w)*scale,(y+h)*scale)
    stroke = None if n["kind"] == "panel" else "#34465a"
    if n["kind"] == "inputplate":
        points = [(x,y+22),(x+21,y),(x+70,y),(x+70,y+5),(x+49,y+27),(x,y+27)]
        draw.polygon([(a*scale,b*scale) for a,b in points], fill=n["fill"])
        draw.line([(a*scale,b*scale) for a,b in points+[points[0]]], fill=stroke, width=3)
        svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in points)+f'" fill="{n["fill"]}" stroke="{stroke}" stroke-width="1.5"/>')
        for line in [[(x,y+22),(x+49,y+22),(x+70,y)],[(x+49,y+22),(x+49,y+27)]]:
            draw.line([(a*scale,b*scale) for a,b in line], fill=stroke,width=3)
            svg.append('<polyline points="'+' '.join(f'{a},{b}' for a,b in line)+f'" fill="none" stroke="{stroke}" stroke-width="1.5"/>')
    elif n["kind"] in {"slab", "trapup", "trapdown"}:
        if n["kind"] == "slab":
            points = [(x+19,y),(x+w,y),(x+w-19,y+h),(x,y+h)]
        elif n["kind"] == "trapup":
            points = [(x+12,y),(x+w-12,y),(x+w,y+h),(x,y+h)]
        else:
            points = [(x,y),(x+w,y),(x+w-12,y+h),(x+12,y+h)]
        draw.polygon([(a*scale,b*scale) for a,b in points], fill=n["fill"])
        draw.line([(a*scale,b*scale) for a,b in points+[points[0]]], fill=stroke, width=3)
        svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in points)+f'" fill="{n["fill"]}" stroke="{stroke}" stroke-width="1.5"/>')
    elif n["kind"] in {"ellipse","dot"}:
        draw.ellipse(coords, fill=n["fill"], outline=stroke, width=3)
        svg.append(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" fill="{n["fill"]}" stroke="{stroke}" stroke-width="1.5"/>')
    else:
        radius = 65 if n["kind"] == "boundary" else (25 if n["kind"] == "panel" else 5)
        draw.rounded_rectangle(coords, radius=radius*scale, fill=n["fill"], outline=None if n["kind"] == "boundary" else stroke, width=3)
        if n["kind"] == "boundary":
            # Dotted rounded outline, matching the supplied figure.
            path=[]
            for ax,ay,start in [(x+w-radius,y+radius,-90),(x+w-radius,y+h-radius,0),(x+radius,y+h-radius,90),(x+radius,y+radius,180)]:
                for deg in range(start,start+91,2):
                    path.append((ax+radius*math.cos(math.radians(deg)),ay+radius*math.sin(math.radians(deg))))
            path.append(path[0])
            distance=0
            for (ax,ay),(bx,by) in zip(path,path[1:]):
                length=math.hypot(bx-ax,by-ay)
                steps=max(1,int(length))
                for step in range(steps):
                    distance+=length/steps
                    if distance>=5:
                        px=ax+(bx-ax)*step/steps; py=ay+(by-ay)*step/steps
                        draw.ellipse(((px-1)*scale,(py-1)*scale,(px+1)*scale,(py+1)*scale),fill=stroke)
                        distance=0
        dash = 'stroke-dasharray="3 4"' if n["kind"] == "boundary" else ""
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{n["fill"]}" stroke="{stroke or "none"}" stroke-width="1.5" {dash}/>')
for n in nodes:
    if n["kind"] in {"boundary", "panel"}: shape(n)
for _,_,points,arrow in edges:
    draw.line([(x*scale,y*scale) for x,y in points], fill="#465462", width=3)
    svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+'" fill="none" stroke="#465462" stroke-width="1.7"/>')
    if arrow:
        x,y=points[-1]; px,py=points[-2]
        angle=math.atan2(y-py,x-px)
        triangle=[(x,y),(x-10*math.cos(angle)+4*math.sin(angle),y-10*math.sin(angle)-4*math.cos(angle)),(x-10*math.cos(angle)-4*math.sin(angle),y-10*math.sin(angle)+4*math.cos(angle))]
        draw.polygon([(a*scale,b*scale) for a,b in triangle],fill="#465462")
        svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in triangle)+'" fill="#465462"/>')
for n in nodes:
    if n["kind"] not in {"boundary","panel"}: shape(n)
    if not n["text"]: continue
    font=ImageFont.truetype("C:/Windows/Fonts/times.ttf",n["size"]*scale)
    lines=n["text"].splitlines(); line_height=n["size"]*1.25
    cx=n["x"]+n["w"]/2; cy=n["y"]+n["h"]/2
    for i,line in enumerate(lines):
        yy=cy+(i-(len(lines)-1)/2)*line_height
        draw.text((cx*scale,yy*scale),line,font=font,anchor="mm",fill="#1c2938")
        svg.append(f'<text x="{cx}" y="{yy}" text-anchor="middle" dominant-baseline="central" font-family="Times New Roman, serif" font-size="{n["size"]}" fill="#1c2938">{escape(line)}</text>')
svg.append('</svg>')
(OUT/"GAM_Fusion.svg").write_text('\n'.join(svg),encoding="utf-8")
image.save(OUT/"GAM_Fusion.png")
print("Created GAM_Fusion.drawio, GAM_Fusion.svg and GAM_Fusion.png")
