def split_text_to_subchunks(text, page_num, position_id, type_, chunk_size=300, overlap=40, is_scanned=False):
    sub_chunks = []
    start = 0
    sub_pos = 1
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            while end < text_len and text[end] not in [" ", "\n"]:
                end += 1
        sub_text = text[start:end].strip()
        if sub_text:
            sub_chunks.append({
                "page": page_num,
                "position": position_id,
                "sub_position": sub_pos,
                "type": type_,
                "is_scanned": is_scanned,
                "data": sub_text
            })
            sub_pos += 1
        start = max(end - overlap, end) if end - overlap < end else end
        if start <= end and end >= text_len:
            break
    return sub_chunks
