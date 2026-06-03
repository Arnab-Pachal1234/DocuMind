from typing import List
from fastapi import UploadFile
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_pdf_text(pdf_files: List[UploadFile]) -> str:
    text = ""

    for pdf in pdf_files:
        try:
            pdf.file.seek(0)
            pdf_reader = PdfReader(pdf.file)

            for page in pdf_reader.pages:
                extracted = page.extract_text()

                if extracted:
                    text += extracted + "\n"

        except Exception as e:
            print(f"Error reading PDF {pdf.filename}: {e}")

    return text


def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_text(text)