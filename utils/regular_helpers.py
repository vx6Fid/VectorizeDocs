def extract_page_content(page):
    elements, table_bboxes = [], []

    for table in page.find_tables():
        table_bboxes.append(table.bbox)
        table_text = "\n".join(" | ".join((cell or "") for cell in row) for row in table.extract())
        elements.append({"type": "table", "top": float(table.bbox[1]), "content": table_text})

    words = page.extract_words()
    grouped_lines = []

    for word in words:
        x0, x1, top, bottom = float(word["x0"]), float(word["x1"]), float(word["top"]), float(word["bottom"])
        if any(x0 >= bx0 and x1 <= bx1 and top >= by0 and bottom <= by1 for (bx0, by0, bx1, by1) in table_bboxes):
            continue
        for line in grouped_lines:
            if abs(line["top"] - top) <= 2:
                line["words"].append((x0, word["text"]))
                break
        else:
            grouped_lines.append({"top": top, "words": [(x0, word["text"])]})

    for line in grouped_lines:
        line["words"].sort()
        text = " ".join(word for _, word in line["words"])
        elements.append({"type": "text", "top": line["top"], "content": text})

    elements.sort(key=lambda e: e["top"])
    return elements

def elements_to_positions(elements):
    positions = []
    pos_counter = 1
    current = None
    for el in elements:
        if current is None:
            current = {"type": el["type"], "content": el["content"], "position": pos_counter}
        else:
            if el["type"] == "text" and current["type"] == "text":
                current["content"] += "\n" + el["content"]
            else:
                positions.append(current)
                pos_counter += 1
                current = {"type": el["type"], "content": el["content"], "position": pos_counter}
    if current:
        positions.append(current)
    return positions
