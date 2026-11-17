"""Shared palette for reproducible research figures. Change colors here."""
import matplotlib.pyplot as plt
from cycler import cycler
PALETTE = {'BLUE': '#0072B2', 'TEAL': '#009E73', 'ORANGE': '#E69F00', 'RED': '#D55E00', 'PURPLE': '#8B5AA5', 'GOLD': '#B89A00', 'TEAL_LIGHT': '#E3F3ED', 'ORANGE_LIGHT': '#FFF0D5', 'RED_LIGHT': '#FBE8DE', 'PURPLE_LIGHT': '#EFE7F4', 'GOLD_LIGHT': '#F7F1DA', 'BLUE_LIGHT': '#E4EFF7', 'CYAN_LIGHT': '#E3F2F3', 'GRAY': '#707985', 'GRID': '#DCE1E6', 'INK': '#243342', 'GRAY_LIGHT': '#F1F3F5'}
def apply_theme():
    plt.rcParams.update({"axes.prop_cycle": cycler(color=[PALETTE[k] for k in ["BLUE", "TEAL", "ORANGE", "RED", "PURPLE", "GOLD"]]), "text.color": PALETTE["INK"], "axes.labelcolor": PALETTE["INK"], "xtick.color": PALETTE["INK"], "ytick.color": PALETTE["INK"], "axes.edgecolor": PALETTE["GRAY"], "grid.color": PALETTE["GRID"], "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})

def save_figure(figure, filename, **kwargs):
    """Export caption-ready figures; retain removed headings in sidecar metadata."""
    from pathlib import Path
    import json
    filename = Path(filename)
    headings = []
    for ax in figure.axes:
        for loc in ['left', 'center', 'right']:
            title = ax.get_title(loc=loc)
            if title:
                headings.append(title)
                ax.set_title('', loc=loc)
        for artist in list(ax.texts):
            if artist.get_transform() == ax.transAxes and artist.get_position()[1] > 1:
                headings.append(artist.get_text())
                artist.remove()
    if figure._suptitle is not None:
        headings.insert(0, figure._suptitle.get_text())
        figure._suptitle.remove()
        figure._suptitle = None
    for artist in list(figure.texts):
        if artist.get_position()[1] >= .85:
            headings.append(artist.get_text())
            artist.remove()
    filename.with_suffix('.captions.json').write_text(json.dumps(headings, indent=2))
    figure.savefig(filename.with_suffix('.png'), **kwargs)
    figure.savefig(filename.with_suffix('.svg'), **kwargs)
