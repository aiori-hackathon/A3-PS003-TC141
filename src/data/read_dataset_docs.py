
from docx import Document

path = "data/raw/Traffic Flow Prediction Dataset.docx"

doc = Document(path)

for paragraph in doc.paragraphs:
    text = paragraph.text.strip()
    if text:
        print(text)

for table_number, table in enumerate(doc.tables, start=1):
    print(f"\n=== TABLE {table_number} ===")
    for row in table.rows:
        print(" | ".join(cell.text.strip() for cell in row.cells))
