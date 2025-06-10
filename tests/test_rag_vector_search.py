import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
os.environ["FAKE_EMBEDDINGS"] = "1"
import json
import subprocess

import rag_vector_search as rvs


def test_load_file_txt(tmp_path):
    path = tmp_path / 'a.txt'
    path.write_text('hello')
    assert rvs.load_file(str(path)) == 'hello'


def test_load_file_md(tmp_path):
    path = tmp_path / 'a.md'
    path.write_text('# title')
    assert rvs.load_file(str(path)) == '# title'


def test_load_file_pdf(tmp_path):
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / 'a.pdf'
    c = canvas.Canvas(str(pdf_path))
    c.drawString(10, 750, 'pdf text')
    c.save()
    text = rvs.load_file(str(pdf_path))
    assert 'pdf text' in text


def test_load_file_docx(tmp_path):
    from docx import Document

    path = tmp_path / 'a.docx'
    doc = Document()
    doc.add_paragraph('docx text')
    doc.save(str(path))
    text = rvs.load_file(str(path))
    assert 'docx text' in text


def test_build_query(tmp_path):
    txt = tmp_path / 'a.txt'
    txt.write_text('hello world. testing text.')
    rvs.build_index([str(txt)])
    results = rvs.query('testing', k=1)
    assert results
    assert results[0]['filename'] == txt.name


def test_cli(tmp_path):
    txt = tmp_path / 'b.txt'
    txt.write_text('some cli text about testing.')
    cmd = [sys.executable, 'rag_vector_search.py', str(txt)]
    env = dict(os.environ)
    env["FAKE_EMBEDDINGS"] = "1"
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    out, _ = proc.communicate(b'testing\n', timeout=10)
    lines = out.decode().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('['))
    json_data = "\n".join(lines[start:])
    data = json.loads(json_data)
    assert isinstance(data, list)
    assert data
    assert data[0]['filename'] == txt.name



