#!/usr/bin/env python3

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "step_source_pdfs"

STEP_SOURCES = [
    ROOT / "step1_socioeco_harmonization_analysis.tex",
    ROOT / "step2_question_harmonization_descriptive_analysis.tex",
    ROOT / "step3_fNIRS_trigger_harmonization_quality_control.tex",
    ROOT / "step4_primary_ ROI_analysis.tex",
    ROOT / "step5_fixed_window_lock_ROI_analysis.tex",
    ROOT / "step6_ROI_analysis_question _covariates.text",
    ROOT / "step7_duration_aware.tex",
    ROOT / "step8_exploratory_back_ROI_item_pair.tex",
]


WRAPPER_TEMPLATE = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{hyperref}
\hypersetup{hidelinks}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
\begin{document}
\sloppy
%STEP_CONTENT%
\end{document}
"""


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_step_content(content: str, source_name: str) -> str:
    sanitized = content
    if source_name == "step7_duration_aware.tex" and sanitized.startswith("a\\section{"):
        sanitized = sanitized.replace("a\\section{", "\\section{", 1)
    unicode_ascii_map = {
        "│": "|",
        "└": "+",
        "├": "+",
        "─": "-",
    }
    for bad_char, replacement in unicode_ascii_map.items():
        sanitized = sanitized.replace(bad_char, replacement)
    return sanitized


def write_wrapper(source_path: Path) -> Path:
    content = source_path.read_text(encoding="utf-8")
    content = sanitize_step_content(content, source_path.name)
    wrapper_text = WRAPPER_TEMPLATE.replace("%STEP_CONTENT%", content)
    wrapper_path = OUTPUT_DIR / f"{source_path.stem}_standalone.tex"
    wrapper_path.write_text(wrapper_text, encoding="utf-8")
    return wrapper_path


def compile_wrapper(wrapper_path: Path) -> None:
    for _ in range(2):
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(OUTPUT_DIR),
                str(wrapper_path),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


def main() -> None:
    ensure_output_dir()
    built_pdfs: list[Path] = []
    for source_path in STEP_SOURCES:
        wrapper_path = write_wrapper(source_path)
        compile_wrapper(wrapper_path)
        built_pdfs.append(OUTPUT_DIR / f"{wrapper_path.stem}.pdf")

    print("Step source PDFs compiled successfully.")
    for pdf_path in built_pdfs:
        print(pdf_path)


if __name__ == "__main__":
    main()
