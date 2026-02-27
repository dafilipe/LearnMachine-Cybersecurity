# latex_logger.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Optional, Any
import os
import datetime


def latex_escape(s: Any) -> str:
    """Escapa caracteres especiais do LaTeX."""
    if s is None:
        return ""
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def fmt_float(x: Any, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return latex_escape(x)


@dataclass
class LatexLogger:
    tex_path: str
    title: str = "Resultados"
    author: str = ""
    _is_open: bool = False

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.tex_path) or ".", exist_ok=True)
        with open(self.tex_path, "w", encoding="utf-8") as f:
            f.write(r"\documentclass[11pt]{article}" + "\n")
            f.write(r"\usepackage[a4paper,margin=2cm]{geometry}" + "\n")
            f.write(r"\usepackage{booktabs}" + "\n")
            f.write(r"\usepackage{longtable}" + "\n")
            f.write(r"\usepackage{array}" + "\n")
            f.write(r"\usepackage{hyperref}" + "\n")
            f.write(r"\usepackage{caption}" + "\n")
            f.write(r"\captionsetup{labelformat=empty}" + "\n")
            f.write("\n")
            f.write(r"\title{" + latex_escape(self.title) + "}\n")
            if self.author:
                f.write(r"\author{" + latex_escape(self.author) + "}\n")
            else:
                f.write(r"\author{}\n")
            f.write(r"\date{" + latex_escape(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "}\n")
            f.write("\n")
            f.write(r"\begin{document}" + "\n")
            f.write(r"\maketitle" + "\n")
            f.write("\n")
        self._is_open = True

    def close(self) -> None:
        if not self._is_open:
            return
        with open(self.tex_path, "a", encoding="utf-8") as f:
            f.write("\n" + r"\end{document}" + "\n")
        self._is_open = False

    def raw(self, latex: str) -> None:
        if not self._is_open:
            self.open()
        with open(self.tex_path, "a", encoding="utf-8") as f:
            f.write(latex)
            if not latex.endswith("\n"):
                f.write("\n")

    def section(self, title: str) -> None:
        self.raw(r"\section*{" + latex_escape(title) + "}")

    def subsection(self, title: str) -> None:
        self.raw(r"\subsection*{" + latex_escape(title) + "}")

    def paragraph(self, text: str) -> None:
        self.raw(latex_escape(text) + r"\par")

    def kv_table(self, items: Sequence[tuple[str, Any]], caption: Optional[str] = None) -> None:
        headers = ["Campo", "Valor"]
        rows = [(k, latex_escape(v)) for k, v in items]
        self.table(headers, rows, caption=caption, longtable=False)

    def table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[Any]],
        caption: Optional[str] = None,
        longtable: bool = True,
        align: Optional[str] = None,
    ) -> None:
        if not self._is_open:
            self.open()

        headers = [latex_escape(h) for h in headers]
        rows_esc: List[List[str]] = []
        for r in rows:
            rows_esc.append([latex_escape(x) for x in r])

        ncol = len(headers)
        if align is None:
            # default: left for first col, right for the rest
            align = "l" + "r" * (ncol - 1)

        env_begin = r"\begin{longtable}{" + align + "}" if longtable else r"\begin{tabular}{" + align + "}"
        env_end   = r"\end{longtable}" if longtable else r"\end{tabular}"

        with open(self.tex_path, "a", encoding="utf-8") as f:
            if caption:
                if longtable:
                    f.write(r"\begin{center}" + "\n")
                    f.write(r"\textbf{" + latex_escape(caption) + r"}\par" + "\n")
                else:
                    f.write(r"\begin{center}" + "\n")
                    f.write(r"\textbf{" + latex_escape(caption) + r"}\par" + "\n")

            f.write(env_begin + "\n")
            f.write(r"\toprule" + "\n")
            f.write(" & ".join(headers) + r" \\" + "\n")
            f.write(r"\midrule" + "\n")

            for r in rows_esc:
                if len(r) != ncol:
                    # garante colunas
                    r = (r + [""] * ncol)[:ncol]
                f.write(" & ".join(r) + r" \\" + "\n")

            f.write(r"\bottomrule" + "\n")
            f.write(env_end + "\n")

            if caption:
                f.write(r"\end{center}" + "\n")
            f.write("\n")
