#!/usr/bin/env python3
"""Split a PDF into chapters based on chapter headings."""

import fitz  # PyMuPDF
import re
import os
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """Remove/replace characters that are invalid in filenames."""
    # Replace problematic characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name[:100]  # Limit length


def find_chapters(doc) -> list[tuple[int, str]]:
    """Find chapter start pages and titles using text pattern matching.

    Returns list of (page_number, title) tuples where page_number is 0-indexed.
    """
    chapters = []

    for page_num in range(doc.page_count):
        text = doc[page_num].get_text()[:500]

        # Match "CHAPTER X" followed by title on next line
        match = re.search(r'^CHAPTER\s+(\d+)\s*\n\s*(.+)', text, re.MULTILINE)
        if match:
            chapter_num = match.group(1)
            chapter_title = match.group(2).strip()
            chapters.append((page_num, f"Chapter_{chapter_num}_{chapter_title}"))

    return chapters


def split_pdf_by_chapters(pdf_path: str, output_dir: str = "chapters"):
    """Split a PDF file into separate chapter files.

    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory to save chapter PDFs (created if doesn't exist)
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    print(f"Opened PDF: {total_pages} pages")

    # Try to use PDF bookmarks first
    toc = doc.get_toc()

    if toc:
        print(f"Found {len(toc)} bookmarks in PDF")
        # Filter to top-level entries (chapters)
        chapters = [(entry[2] - 1, entry[1]) for entry in toc if entry[0] == 1]
    else:
        print("No bookmarks found, detecting chapters by text...")
        chapters = find_chapters(doc)

    if not chapters:
        print("No chapters found in PDF!")
        return

    print(f"\nFound {len(chapters)} chapters:")
    for page, title in chapters:
        print(f"  Page {page + 1}: {title}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Split into chapters
    print(f"\nSplitting into {output_dir}/...")

    for i, (start_page, title) in enumerate(chapters):
        # End page is start of next chapter, or end of document
        if i + 1 < len(chapters):
            end_page = chapters[i + 1][0] - 1
        else:
            end_page = total_pages - 1

        # Create new PDF for this chapter
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)

        # Save with sanitized filename
        filename = f"{sanitize_filename(title)}.pdf"
        output_file = output_path / filename
        new_doc.save(str(output_file))
        new_doc.close()

        page_count = end_page - start_page + 1
        print(f"  Created: {filename} ({page_count} pages)")

    doc.close()
    print(f"\nDone! {len(chapters)} chapter PDFs saved to {output_dir}/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python split_pdf.py <pdf_path> [output_dir]")
        print("  pdf_path:   Path to the PDF file to split")
        print("  output_dir: Directory for output files (default: 'chapters')")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "chapters"

    split_pdf_by_chapters(pdf_path, output_dir)
