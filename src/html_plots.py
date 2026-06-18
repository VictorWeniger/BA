import json

from .utils import project_path


def write_3d_scatter_html(coords, labels, title, output_path):
    """Write an interactive Plotly 3D scatter plot as HTML."""

    payload = {
        "x": coords[:, 0].round(5).tolist(),
        "y": coords[:, 1].round(5).tolist(),
        "z": coords[:, 2].round(5).tolist(),
        "labels": [str(label) for label in labels],
    }

    html = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #ffffff;
    }}
    #plot {{
      width: 100vw;
      height: 100vh;
    }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const data = {json.dumps(payload)};
    const uniqueLabels = [...new Set(data.labels)].sort();
    const labelToIndex = Object.fromEntries(uniqueLabels.map((label, index) => [label, index]));
    const colorValues = data.labels.map(label => labelToIndex[label]);

    const trace = {{
      type: "scatter3d",
      mode: "markers",
      x: data.x,
      y: data.y,
      z: data.z,
      text: data.labels.map(label => `Label: ${{label}}`),
      marker: {{
        size: 3,
        color: colorValues,
        colorscale: "Viridis",
        opacity: 0.78,
        colorbar: {{ title: "Label index" }}
      }}
    }};

    const layout = {{
      title: "{title}",
      scene: {{
        xaxis: {{ title: "PC 1" }},
        yaxis: {{ title: "PC 2" }},
        zaxis: {{ title: "PC 3" }}
      }},
      margin: {{ l: 0, r: 0, b: 0, t: 48 }}
    }};

    Plotly.newPlot("plot", [trace], layout, {{ responsive: true }});
  </script>
</body>
</html>
"""

    output_path = project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
