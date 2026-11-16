Compile from this directory:
  xelatex -interaction=nonstopmode -halt-on-error report.tex
  xelatex -interaction=nonstopmode -halt-on-error report.tex

Article prose and captions are preserved. Seven article images are replaced by updated figures. All 34 original image assets remain in figures/, including equation images. Source HTML, BeautifulSoup text extraction, Tesseract OCR and figure mapping are in source/.

The four plain-text web equations were typeset as LaTeX. The IL-6 source expression is flattened and ambiguous (dIL6dt = kprod ( 1+RTCEC50 + RTC ) - kclear IL6); the typeset fraction is a reconstruction consistent with the adjoining Hill-response explanation, not a recovered original fraction. The exact original is retained in source/article-extracted.txt and a TeX comment. Other image equations are preserved verbatim as images to avoid transcription loss.

Figure numbering and original captions are retained as printed text. Figure 18 uses the revised sensitivity visualization, which shows small changes rather than exact dose independence; original claims remain as requested. Tables and citations are linked to the Google document in the article appendix; they are not embedded in the article itself.

Equation revision: all dedicated equation images are now native numbered LaTeX. Cross-references use Equation X. The flattened RTC = Dc R TaKD expression was reconstructed as R_TC = D_c R T_a / K_D, consistent with the surrounding binding-affinity description; source HTML retains the exact ambiguous original. Inline temperature and rate subscripts are rendered in math mode.
