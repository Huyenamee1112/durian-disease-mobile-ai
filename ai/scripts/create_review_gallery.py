from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "ai" / "data" / "supabase_dataset"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ai" / "data" / "review"
REVIEW_COLUMNS = [
    "id",
    "label",
    "disease",
    "split",
    "image_path",
    "local_path",
    "created_at",
    "review_status",
    "reject_reason",
    "review_note",
]


def load_review_rows(dataset_dir: Path) -> list[dict[str, str]]:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    rows: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("download_status") not in {"downloaded", "existing"}:
                continue
            local_path = row.get("local_path", "")
            if not local_path or not (dataset_dir / local_path).exists():
                continue
            rows.append(
                {
                    "id": row.get("id", ""),
                    "label": row.get("label", ""),
                    "disease": row.get("disease", ""),
                    "split": row.get("split", ""),
                    "image_path": row.get("image_path", ""),
                    "local_path": local_path,
                    "created_at": row.get("created_at", ""),
                    "review_status": row.get("review_status") or "pending",
                    "reject_reason": row.get("reject_reason", ""),
                    "review_note": "",
                }
            )
    return rows


def write_review_csv(rows: list[dict[str, str]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "review_sheet.csv"
    with review_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return review_path


def write_review_html(rows: list[dict[str, str]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "review_gallery.html"
    rows_json = json.dumps(rows, ensure_ascii=False)
    labels_json = json.dumps(sorted({row["label"] for row in rows}), ensure_ascii=False)
    html_path.write_text(
        f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Review ảnh lá sầu riêng</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #173c27;
      background: #f4fbf6;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: grid;
      gap: 10px;
      padding: 12px 16px;
      background: #ffffff;
      border-bottom: 1px solid #d5eadc;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    select, input, button, textarea {{
      font: inherit;
      border: 1px solid #b9ddc6;
      border-radius: 8px;
      background: white;
      color: #173c27;
    }}
    select, input {{
      min-height: 38px;
      padding: 0 10px;
    }}
    button {{
      min-height: 38px;
      padding: 0 12px;
      cursor: pointer;
      font-weight: 650;
    }}
    button.primary {{ background: #176b3a; color: white; border-color: #176b3a; }}
    button.warn {{ background: #fff6e0; border-color: #e8c26c; }}
    button.danger {{ background: #fff0ed; border-color: #f0a399; }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      padding: 16px;
    }}
    .image-panel, .side-panel {{
      border: 1px solid #d5eadc;
      background: white;
      border-radius: 10px;
      overflow: hidden;
    }}
    .image-wrap {{
      min-height: 72vh;
      display: grid;
      place-items: center;
      background: #0d1f15;
    }}
    img {{
      max-width: 100%;
      max-height: 82vh;
      display: block;
      object-fit: contain;
    }}
    .side-panel {{
      padding: 14px;
      display: grid;
      gap: 12px;
      align-content: start;
    }}
    .meta {{
      display: grid;
      gap: 6px;
      font-size: 14px;
      word-break: break-word;
    }}
    .meta b {{ color: #0d4d29; }}
    .actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    textarea {{
      width: 100%;
      min-height: 74px;
      padding: 10px;
      resize: vertical;
    }}
    .progress {{
      height: 8px;
      border-radius: 99px;
      overflow: hidden;
      background: #e5f4ea;
    }}
    .bar {{
      height: 100%;
      width: 0%;
      background: #176b3a;
    }}
    .status-pill {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 99px;
      background: #eaf7ef;
      font-weight: 700;
    }}
    .status-invalid {{ background: #fff0ed; color: #a93020; }}
    .status-valid {{ background: #e7f8ee; color: #176b3a; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      .image-wrap {{ min-height: 45vh; }}
      img {{ max-height: 58vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Review ảnh lá sầu riêng</h1>
    <div class="toolbar">
      <button id="prev">Trước</button>
      <button id="next">Sau</button>
      <select id="label-filter"></select>
      <select id="status-filter">
        <option value="all">Tất cả trạng thái</option>
        <option value="pending">Pending</option>
        <option value="valid">Valid</option>
        <option value="invalid">Invalid</option>
      </select>
      <input id="search" placeholder="Tìm id/path...">
      <button id="export" class="primary">Export CSV</button>
    </div>
    <div class="progress"><div class="bar" id="bar"></div></div>
  </header>

  <main>
    <section class="image-panel">
      <div class="image-wrap">
        <img id="image" alt="Ảnh review">
      </div>
    </section>
    <aside class="side-panel">
      <div class="meta" id="meta"></div>
      <div class="actions">
        <button class="primary" data-status="valid" data-reason="">Valid</button>
        <button class="danger" data-status="invalid" data-reason="hand">Dính tay</button>
        <button class="danger" data-status="invalid" data-reason="blur">Mờ/rung</button>
        <button class="danger" data-status="invalid" data-reason="bad_label">Sai nhãn</button>
        <button class="danger" data-status="invalid" data-reason="too_small">Lá quá nhỏ</button>
        <button class="danger" data-status="invalid" data-reason="duplicate">Trùng ảnh</button>
        <button class="danger" data-status="invalid" data-reason="wrong_leaf">Sai loại lá</button>
        <button class="warn" data-status="pending" data-reason="">Pending</button>
      </div>
      <textarea id="note" placeholder="Ghi chú review..."></textarea>
      <p>
        Phím tắt: <b>V</b> valid, <b>H</b> dính tay, <b>B</b> mờ,
        <b>L</b> sai nhãn, <b>N</b> ảnh kế tiếp.
      </p>
    </aside>
  </main>

  <script>
    const rows = {rows_json};
    const labels = {labels_json};
    const storageKey = "durian-review-v1";
    const imageBase = "../supabase_dataset/";
    let state = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    let current = 0;

    for (const row of rows) {{
      if (!state[row.id]) {{
        state[row.id] = {{
          review_status: row.review_status || "pending",
          reject_reason: row.reject_reason || "",
          review_note: row.review_note || ""
        }};
      }}
    }}

    const image = document.getElementById("image");
    const meta = document.getElementById("meta");
    const note = document.getElementById("note");
    const labelFilter = document.getElementById("label-filter");
    const statusFilter = document.getElementById("status-filter");
    const search = document.getElementById("search");
    const bar = document.getElementById("bar");

    labelFilter.innerHTML = '<option value="all">Tất cả lớp</option>' +
      labels.map(label => `<option value="${{label}}">${{label}}</option>`).join("");

    function saveState() {{
      localStorage.setItem(storageKey, JSON.stringify(state));
    }}

    function filteredRows() {{
      const labelValue = labelFilter.value;
      const statusValue = statusFilter.value;
      const query = search.value.trim().toLowerCase();
      return rows.filter(row => {{
        const review = state[row.id] || {{}};
        if (labelValue !== "all" && row.label !== labelValue) return false;
        if (statusValue !== "all" && review.review_status !== statusValue) return false;
        if (query && !`${{row.id}} ${{row.image_path}} ${{row.label}}`.toLowerCase().includes(query)) return false;
        return true;
      }});
    }}

    function currentRows() {{
      const list = filteredRows();
      if (current >= list.length) current = Math.max(0, list.length - 1);
      return list;
    }}

    function render() {{
      const list = currentRows();
      const total = rows.length;
      const reviewed = rows.filter(row => (state[row.id] || {{}}).review_status !== "pending").length;
      bar.style.width = `${{Math.round((reviewed / Math.max(total, 1)) * 100)}}%`;

      if (!list.length) {{
        image.removeAttribute("src");
        meta.innerHTML = "Không có ảnh phù hợp bộ lọc.";
        note.value = "";
        return;
      }}

      const row = list[current];
      const review = state[row.id];
      image.src = imageBase + row.local_path;
      note.value = review.review_note || "";
      const statusClass = review.review_status === "valid"
        ? "status-valid"
        : review.review_status === "invalid"
          ? "status-invalid"
          : "";
      meta.innerHTML = `
        <div><b>Ảnh:</b> ${{current + 1}} / ${{list.length}} trong bộ lọc; đã review ${{reviewed}} / ${{total}}</div>
        <div><b>Trạng thái:</b> <span class="status-pill ${{statusClass}}">${{review.review_status || "pending"}}</span></div>
        <div><b>Lý do loại:</b> ${{review.reject_reason || "-"}}</div>
        <div><b>Lớp:</b> ${{row.label}}</div>
        <div><b>Split:</b> ${{row.split}}</div>
        <div><b>ID:</b> ${{row.id}}</div>
        <div><b>Storage:</b> ${{row.image_path}}</div>
        <div><b>Ngày:</b> ${{row.created_at || "-"}}</div>
      `;
    }}

    function mark(status, reason = "") {{
      const list = currentRows();
      if (!list.length) return;
      const row = list[current];
      state[row.id] = {{
        ...state[row.id],
        review_status: status,
        reject_reason: reason,
        review_note: note.value.trim()
      }};
      saveState();
      if (current < list.length - 1) current += 1;
      render();
    }}

    function toCsvValue(value) {{
      return `"${{String(value ?? "").replaceAll('"', '""')}}"`;
    }}

    function exportCsv() {{
      const header = ["id","label","disease","split","image_path","local_path","created_at","review_status","reject_reason","review_note"];
      const lines = [header.join(",")];
      for (const row of rows) {{
        const review = state[row.id] || {{}};
        lines.push([
          row.id, row.label, row.disease, row.split, row.image_path, row.local_path,
          row.created_at, review.review_status || "pending",
          review.reject_reason || "", review.review_note || ""
        ].map(toCsvValue).join(","));
      }}
      const blob = new Blob([lines.join("\\n")], {{ type: "text/csv;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "review_results.csv";
      link.click();
      URL.revokeObjectURL(url);
    }}

    document.getElementById("prev").addEventListener("click", () => {{
      current = Math.max(0, current - 1);
      render();
    }});
    document.getElementById("next").addEventListener("click", () => {{
      current = Math.min(currentRows().length - 1, current + 1);
      render();
    }});
    document.getElementById("export").addEventListener("click", exportCsv);
    document.querySelectorAll("[data-status]").forEach(button => {{
      button.addEventListener("click", () => mark(button.dataset.status, button.dataset.reason));
    }});
    note.addEventListener("change", () => {{
      const list = currentRows();
      if (!list.length) return;
      const row = list[current];
      state[row.id] = {{ ...state[row.id], review_note: note.value.trim() }};
      saveState();
    }});
    labelFilter.addEventListener("change", () => {{ current = 0; render(); }});
    statusFilter.addEventListener("change", () => {{ current = 0; render(); }});
    search.addEventListener("input", () => {{ current = 0; render(); }});
    document.addEventListener("keydown", event => {{
      if (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT") return;
      const key = event.key.toLowerCase();
      if (key === "v") mark("valid", "");
      if (key === "h") mark("invalid", "hand");
      if (key === "b") mark("invalid", "blur");
      if (key === "l") mark("invalid", "bad_label");
      if (key === "n") {{
        current = Math.min(currentRows().length - 1, current + 1);
        render();
      }}
    }});

    render();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create local image review CSV and HTML gallery.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_review_rows(args.dataset_dir)
    csv_path = write_review_csv(rows, args.output_dir)
    html_path = write_review_html(rows, args.output_dir)
    label_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    print(f"Review rows: {len(rows)}")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print(f"CSV: {csv_path}")
    print(f"HTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
