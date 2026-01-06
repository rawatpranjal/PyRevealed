#!/usr/bin/env python3
"""Transcribe PDF files to text using PyMuPDF."""

import fitz  # PyMuPDF
import os
from pathlib import Path


def transcribe_pdf(pdf_path: str, output_path: str = None) -> str:
    """Extract all text from a PDF file.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save text file (if None, returns text only)

    Returns:
        Extracted text content
    """
    doc = fitz.open(pdf_path)
    text_parts = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

    doc.close()

    full_text = "\n\n".join(text_parts)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

    return full_text


def transcribe_directory(input_dir: str, output_dir: str = None):
    """Transcribe all PDFs in a directory.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory for output text files (default: same as input with '_text' suffix)
    """
    input_path = Path(input_dir)

    if output_dir is None:
        output_dir = str(input_path) + "_text"

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    pdf_files = sorted(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files to transcribe\n")

    for pdf_file in pdf_files:
        output_file = output_path / (pdf_file.stem + ".txt")

        print(f"Transcribing: {pdf_file.name}")
        text = transcribe_pdf(str(pdf_file), str(output_file))

        # Count words and pages
        word_count = len(text.split())
        page_count = text.count("--- Page ")
        print(f"  -> {output_file.name} ({page_count} pages, {word_count:,} words)")

    print(f"\nDone! {len(pdf_files)} files transcribed to {output_dir}/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python transcribe_pdf.py <pdf_or_dir> [output_dir]")
        print("  pdf_or_dir: Path to a PDF file or directory of PDFs")
        print("  output_dir: Directory for output text files")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if os.path.isdir(input_path):
        transcribe_directory(input_path, output_dir)
    else:
        # Single file
        if output_dir:
            output_file = output_dir
        else:
            output_file = Path(input_path).stem + ".txt"
        transcribe_pdf(input_path, output_file)
        print(f"Transcribed: {input_path} -> {output_file}")
