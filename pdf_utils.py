from typing import List
from fastapi import UploadFile
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_pdf_text(pdf_files: List[UploadFile]) -> str:
    text = ""

    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf.file)

        for page in pdf_reader.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

    return text


def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_text(text)