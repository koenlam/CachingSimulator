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


def set_fig_fullwidth():
    """ Setup figure size for full width figures
        Scaling is not required, e.g., in Latex import without [width=\textwidth]
    """
    figsize = get_figsize(COLUMNWIDTH, wf=1.0, hf=0.4)
    params = {'backend': 'ps',
              'axes.labelsize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'figure.constrained_layout.use':True,
              'figure.figsize': figsize}
    matplotlib.rcParams.update(params)

def set_fig_halfwidth():
    """ Setup figure size for half width figures
        Scaling is required
    """
    figsize = get_figsize(COLUMNWIDTH, wf=1.0, hf=1.0)
    params = {'backend': 'ps',
              'axes.labelsize': 20,
              'legend.fontsize': 16,
              'legend.labelspacing': 0.25,
            #   'legend.loc': 'lower right',
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.constrained_layout.use':True,
              'figure.figsize': figsize}
    matplotlib.rcParams.update(params)