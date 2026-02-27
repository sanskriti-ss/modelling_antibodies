"""
Block Diagrams for Antibody Binding Models — using Graphviz
============================================================

Generates clean, auto-laid-out block-and-arrow diagrams using the
graphviz library (DOT language) instead of manual coordinate placement.

Covers:
  1. Bispecific antibody (Appendix 1)  — binding cycle
  2. Multivalent binding (Appendix 2)  — sequential ladder
  3. Anti-VEGF FcRn PK (Appendix 5)   — 2-compartment PK + equilibrium
"""

import graphviz
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════
#  Colour palette
# ═══════════════════════════════════════════════════
COLORS = {
    'drug':      '#2E86C1',   # blue
    'receptor':  '#C0392B',   # red
    'complex':   '#E67E22',   # orange
    'ternary':   '#27AE60',   # green
    'compartment': '#F5F5F5', # light grey
    'pk_node':   '#8E44AD',   # purple
    'edge':      '#333333',   # dark grey
    'fcrn':      '#D4AC0D',   # gold
}

NODE_STYLE = dict(
    style='filled,rounded',
    shape='box',
    fontname='Helvetica',
    fontsize='11',
    penwidth='1.5',
    margin='0.25,0.15',
)


# ═══════════════════════════════════════════════════
#  1.  Bispecific Antibody  (Appendix 1)
# ═══════════════════════════════════════════════════

def draw_bispecific_diagram(filename: str = 'diagram_bispecific'):
    """
    Diamond-shaped binding cycle:
        C (free drug) binds RE and/or RT, forming
        REC, RTC, and the ternary complex RETC.
    """
    dot = graphviz.Digraph(
        name='Bispecific',
        format='png',
        engine='neato',          # spring-based layout for the diamond
        graph_attr={
            'label': 'Bispecific Antibody Binding — Appendix 1',
            'labelloc': 't',
            'fontsize': '18',
            'fontname': 'Helvetica-Bold',
            'bgcolor': 'white',
            'dpi': '200',
            'size': '8,6',
            'pad': '0.5',
        },
    )

    # --- Nodes (placed in a diamond with neato pos) ---
    dot.node('C', 'C\n(Free Drug)',
             pos='4,2!', fillcolor=COLORS['drug'], fontcolor='white', **NODE_STYLE)

    dot.node('RE', 'RE\n(Free BCMA)',
             pos='2,4!', fillcolor=COLORS['receptor'], fontcolor='white', **NODE_STYLE)

    dot.node('RT', 'RT\n(Free CD3)',
             pos='6,4!', fillcolor=COLORS['receptor'], fontcolor='white', **NODE_STYLE)

    dot.node('REC', 'REC\n(BCMA·Drug)',
             pos='2,6!', fillcolor=COLORS['complex'], fontcolor='white', **NODE_STYLE)

    dot.node('RTC', 'RTC\n(CD3·Drug)',
             pos='6,6!', fillcolor=COLORS['complex'], fontcolor='white', **NODE_STYLE)

    dot.node('RETC', 'RETC\n(Ternary)',
             pos='4,8!', fillcolor=COLORS['ternary'], fontcolor='white', **NODE_STYLE)

    # --- Edges (equilibrium arrows) ---
    edge_kw = dict(color=COLORS['edge'], penwidth='1.8', fontname='Helvetica', fontsize='10')

    # C + RE ⇌ REC
    dot.edge('C', 'REC', label='  + RE\n  KD_E  ', dir='both', **edge_kw)
    # C + RT ⇌ RTC
    dot.edge('C', 'RTC', label='  + RT\n  KD_T  ', dir='both', **edge_kw)
    # REC + RT ⇌ RETC
    dot.edge('REC', 'RETC', label='  + RT\n  KD_T  ', dir='both', **edge_kw)
    # RTC + RE ⇌ RETC
    dot.edge('RTC', 'RETC', label='  + RE\n  KD_E  ', dir='both', **edge_kw)

    # Mass-balance dashed links
    dot.edge('RE', 'REC', style='dashed', color='#999999',
             label='  mass\n  balance', fontsize='8', fontcolor='#999999')
    dot.edge('RT', 'RTC', style='dashed', color='#999999',
             label='  mass\n  balance', fontsize='8', fontcolor='#999999')

    outpath = os.path.join(OUTPUT_DIR, filename)
    dot.render(outpath, cleanup=True)
    print(f"  ✓ Saved {outpath}.png")
    return dot


# ═══════════════════════════════════════════════════
#  2.  Multivalent Binding  (Appendix 2)
# ═══════════════════════════════════════════════════

def draw_multivalent_diagram(N: int = 6, filename: str | None = None):
    """
    Horizontal ladder:  C → R₁C → R₂C → … → RₙC
    with free receptor Rf feeding into every step.
    """
    if filename is None:
        filename = f'diagram_multivalent_N{N}'

    dot = graphviz.Digraph(
        name='Multivalent',
        format='png',
        engine='dot',            # hierarchical left-to-right
        graph_attr={
            'label': f'Multivalent Binding (N = {N}) — Appendix 2',
            'labelloc': 't',
            'fontsize': '18',
            'fontname': 'Helvetica-Bold',
            'bgcolor': 'white',
            'dpi': '200',
            'rankdir': 'LR',     # left → right
            'nodesep': '0.8',
            'ranksep': '1.0',
        },
    )

    # Free receptor at the top
    dot.node('Rf', 'Rf\n(Free Receptor)',
             fillcolor=COLORS['receptor'], fontcolor='white', **NODE_STYLE)

    # Free drug (leftmost)
    dot.node('C', 'C\n(Free Drug)',
             fillcolor=COLORS['drug'], fontcolor='white', **NODE_STYLE)

    # Intermediate complexes
    for j in range(1, N + 1):
        label = f'R{j}C'
        if j == N:
            label += '\n(Fully Bound)'
            fc = COLORS['ternary']
        else:
            fc = COLORS['complex']
        dot.node(f'R{j}C', label, fillcolor=fc, fontcolor='white', **NODE_STYLE)

    # Horizontal binding arrows:  C → R1C → R2C → ... → RNC
    edge_kw = dict(penwidth='1.8', fontname='Helvetica', fontsize='9')

    # Forward: (N-j+1)·kon·Rf   |   Reverse: j·koff
    prev = 'C'
    for j in range(1, N + 1):
        cur = f'R{j}C'
        fwd_label = f'{N - j + 1}·kon·Rf'
        rev_label = f'{j}·koff'
        # forward (blue)
        dot.edge(prev, cur, label=f' {fwd_label} ', color=COLORS['drug'],
                 fontcolor=COLORS['drug'], **edge_kw)
        # reverse (red)
        dot.edge(cur, prev, label=f' {rev_label} ', color=COLORS['receptor'],
                 fontcolor=COLORS['receptor'], **edge_kw)
        prev = cur

    # Rf feeds into each binding step (dashed grey)
    for j in range(1, N + 1):
        dot.edge('Rf', f'R{j}C', style='dashed', color='#AAAAAA',
                 arrowsize='0.6', penwidth='0.8')

    # Equation annotation
    dot.node('eq', (
        f'Eq B16:  (N - j + 1) * R_{{j-1}}C * Rf  -  j * KD * RjC  =  0\n'
        f'where KD = koff / kon'
    ), shape='note', style='filled', fillcolor='#E8F8E8',
       fontsize='10', fontname='Courier', margin='0.3,0.2')

    outpath = os.path.join(OUTPUT_DIR, filename)
    dot.render(outpath, cleanup=True)
    print(f"  ✓ Saved {outpath}.png")
    return dot


# ═══════════════════════════════════════════════════
#  3.  Anti-VEGF + FcRn PK  (Appendix 5)
# ═══════════════════════════════════════════════════

def draw_vegf_fcrn_diagram(filename: str = 'diagram_vegf_fcrn'):
    """
    2-compartment PK model with FcRn-mediated recycling:
      Central  ⇌  Peripheral
      Central → Endosome (FcRn binding at pH 6) → Recycled back
      Endosome: equilibrium at pH 6.0 and pH 7.4
      VEGF target binding in central compartment
    """
    dot = graphviz.Digraph(
        name='VEGF_FcRn',
        format='png',
        engine='dot',
        graph_attr={
            'label': 'Anti-VEGF + FcRn PK Model — Appendix 5',
            'labelloc': 't',
            'fontsize': '18',
            'fontname': 'Helvetica-Bold',
            'bgcolor': 'white',
            'dpi': '200',
            'compound': 'true',
            'nodesep': '0.7',
            'ranksep': '1.2',
        },
    )

    edge_kw = dict(penwidth='1.8', fontname='Helvetica', fontsize='9')

    # ── Central Compartment ──
    with dot.subgraph(name='cluster_central') as c:
        c.attr(label='Central Compartment (V₁)', style='rounded,filled',
               fillcolor='#EBF5FB', color='#2E86C1', fontsize='13',
               fontname='Helvetica-Bold', penwidth='2')

        c.node('Cf_c', 'Cf\n(Free Drug)', fillcolor=COLORS['drug'],
               fontcolor='white', **NODE_STYLE)
        c.node('VEGF', 'VEGF\n(Free Target)', fillcolor=COLORS['receptor'],
               fontcolor='white', **NODE_STYLE)
        c.node('VEGF_C', 'VEGF·C\n(Complex)', fillcolor=COLORS['complex'],
               fontcolor='white', **NODE_STYLE)

        c.edge('Cf_c', 'VEGF_C', label=' + VEGF\n KD_VEGF', dir='both',
               color=COLORS['edge'], **edge_kw)
        c.edge('VEGF', 'VEGF_C', style='dashed', color='#999999',
               arrowsize='0.6')

    # ── Peripheral Compartment ──
    with dot.subgraph(name='cluster_periph') as p:
        p.attr(label='Peripheral Compartment (V₂)', style='rounded,filled',
               fillcolor='#FDEDEC', color='#C0392B', fontsize='13',
               fontname='Helvetica-Bold', penwidth='2')

        p.node('Cp', 'Cp\n(Peripheral\nDrug)', fillcolor=COLORS['pk_node'],
               fontcolor='white', **NODE_STYLE)

    # ── Endosomal Compartment ──
    with dot.subgraph(name='cluster_endosome') as e:
        e.attr(label='Endosome (FcRn Recycling)', style='rounded,filled',
               fillcolor='#FEF9E7', color='#D4AC0D', fontsize='13',
               fontname='Helvetica-Bold', penwidth='2')

        e.node('Ce', 'Ce\n(Endosomal\nDrug)', fillcolor=COLORS['drug'],
               fontcolor='white', **NODE_STYLE)
        e.node('FcRn', 'FcRn\n(Free)', fillcolor=COLORS['fcrn'],
               fontcolor='white', **NODE_STYLE)
        e.node('FcRn_C_6', 'FcRn·C\n(pH 6.0)', fillcolor=COLORS['complex'],
               fontcolor='white', **NODE_STYLE)
        e.node('FcRn_C_7', 'FcRn·C\n(pH 7.4)', fillcolor=COLORS['ternary'],
               fontcolor='white', **NODE_STYLE)

        # Binding equilibria inside endosome
        e.edge('Ce', 'FcRn_C_6', label=' KD(pH 6)', dir='both',
               color=COLORS['edge'], **edge_kw)
        e.edge('FcRn_C_6', 'FcRn_C_7', label=' pH shift ', dir='both',
               style='dashed', color=COLORS['fcrn'], **edge_kw)
        e.edge('FcRn', 'FcRn_C_6', style='dashed', color='#999999',
               arrowsize='0.6')

    # ── Cross-compartment flows ──
    # Central ⇌ Peripheral  (Q/V₁, Q/V₂)
    dot.edge('Cf_c', 'Cp', label=' Q/V₁ ', color=COLORS['pk_node'], **edge_kw)
    dot.edge('Cp', 'Cf_c', label=' Q/V₂ ', color=COLORS['pk_node'], **edge_kw)

    # Central → Endosome (uptake: CL_up)
    dot.edge('Cf_c', 'Ce', label=' CL_up/V₁\n (uptake) ',
             color=COLORS['fcrn'], **edge_kw)

    # Endosome → Central (FcRn recycling of bound drug)
    dot.edge('FcRn_C_7', 'Cf_c', label=' FR·kRec\n (recycle) ',
             color=COLORS['ternary'], **edge_kw)

    # Elimination nodes
    dot.node('elim_endo', 'Lysosomal\nDegradation',
             shape='circle', style='filled', fillcolor='#F1948A',
             fontcolor='white', fontsize='9', fontname='Helvetica',
             width='1.0', fixedsize='true')
    dot.node('elim_cl', 'Clearance\n(CL)',
             shape='circle', style='filled', fillcolor='#F1948A',
             fontcolor='white', fontsize='9', fontname='Helvetica',
             width='1.0', fixedsize='true')

    dot.edge('Ce', 'elim_endo', label=' kDeg ', color='#C0392B', **edge_kw)
    dot.edge('Cf_c', 'elim_cl', label=' CL/V₁ ', color='#C0392B', **edge_kw)

    # IV dose input
    dot.node('dose', 'IV Dose', shape='invhouse', style='filled',
             fillcolor='#AED6F1', fontcolor='#154360',
             fontname='Helvetica-Bold', fontsize='11')
    dot.edge('dose', 'Cf_c', label=' bolus ', color=COLORS['drug'],
             style='bold', **edge_kw)

    # VEGF synthesis / elimination
    dot.node('vegf_syn', 'ksyn', shape='plaintext', fontsize='9',
             fontname='Helvetica')
    dot.edge('vegf_syn', 'VEGF', label=' synthesis ', color='#27AE60',
             style='dashed', **edge_kw)

    dot.node('elim_vegf', 'kint', shape='circle', style='filled',
             fillcolor='#F1948A', fontcolor='white', width='0.5',
             fixedsize='true', fontsize='8')
    dot.edge('VEGF_C', 'elim_vegf', label=' internalization ',
             color='#C0392B', style='dashed', **edge_kw)

    outpath = os.path.join(OUTPUT_DIR, filename)
    dot.render(outpath, cleanup=True)
    print(f"  ✓ Saved {outpath}.png")
    return dot


# ═══════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print("Generating diagrams with Graphviz…\n")
    draw_bispecific_diagram()
    draw_multivalent_diagram(N=6)
    draw_vegf_fcrn_diagram()
    print("\nDone.")
