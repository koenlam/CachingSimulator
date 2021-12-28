import matplotlib

COLUMNWIDTH = 345.0  # value given by Latex


# https://stackoverflow.com/questions/29187618/matplotlib-and-latex-beamer-correct-size
def get_figsize(columnwidth, wf=0.5, hf=(5.**0.5-1.0)/2.0):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                      Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                              using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf 
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]


def set_fig_fullwidth(scale=1.5):
    """ Setup figure size for full width figures
        Scaling is not required, e.g., in Latex import without [width=\textwidth] if scale=1.0
    """
    figsize = get_figsize(COLUMNWIDTH, wf=1.0*scale, hf=0.4*scale)
    params = {'backend': 'ps',
              'axes.labelsize': 10*scale,
              'legend.fontsize': 8*scale,
              'legend.labelspacing': 0.25,
              'xtick.labelsize': 8*scale,
              'ytick.labelsize': 8*scale,
              'figure.constrained_layout.use':True,
              'figure.figsize': figsize}
    matplotlib.rcParams.update(params)

def set_fig_halfwidth():
    """ Setup figure size for half width figures
        Scaling is required
    """
    figsize = get_figsize(COLUMNWIDTH, wf=1.0, hf=0.8)
    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'legend.fontsize': 16,
              'legend.labelspacing': 0.25,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.constrained_layout.use':True,
              'figure.figsize': figsize}
    matplotlib.rcParams.update(params)


def set_fig_onethirdwidth():
    """ Setup figure size for one third width figures
        Scaling is required
    """
    scale = 1.25
    figsize = get_figsize(COLUMNWIDTH, wf=0.8, hf=1.0)
    params = {'backend': 'ps',
              'axes.labelsize': 20*scale,
              'legend.fontsize': 16*scale,
              'legend.labelspacing': 0.25*scale,
              'xtick.labelsize': 16*scale,
              'ytick.labelsize': 16*scale,
              'figure.constrained_layout.use':True,
              'figure.figsize': figsize}
    matplotlib.rcParams.update(params)