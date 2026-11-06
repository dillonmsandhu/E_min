import re

def clean_tex_header(header):
    h = header.replace("λ*", r"$\lambda^*$")
    h = h.replace("λ", r"$\lambda$")
    h = h.replace("∈", r"$\in$")
    return h

def clean_tex_cell(cell):
    # If it contains \pm
    if r"\pm" in cell:
        if r"\textbf{" in cell:
            inner = cell.replace(r"\textbf{", "").replace("}", "").strip()
            return f"\\textbf{{{inner.replace(r'\\pm', r'\\pm')}}}"
        return f"${cell}$"
    return cell
