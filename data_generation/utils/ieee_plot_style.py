import matplotlib.pyplot as plt

def setup_ieee_style():
    """Configures matplotlib parameters for IEEE publication-style plots (without LaTeX)."""
    # Set font to Times New Roman if available, otherwise default serif
    plt.rc('font', family='serif', serif=['Times New Roman', 'DejaVu Serif', 'serif'])
    print(f"Attempting to use font: {plt.rcParams['font.serif']}")

    # Font sizes
    plt.rc('axes', titlesize=10)    # Title fontsize
    plt.rc('axes', labelsize=8)     # x and y labels fontsize
    plt.rc('xtick', labelsize=8)    # x tick labels fontsize
    plt.rc('ytick', labelsize=8)    # y tick labels fontsize
    plt.rc('legend', fontsize=8)    # legend fontsize
    plt.rc('figure', titlesize=12)  # Figure suptitle fontsize

    # Figure size (approx. single column width)
    # Adjust as needed, e.g., (3.5, 2.5) for single column, (7, 3) for double column
    plt.rc('figure', figsize=(3.5, 2.5))

    # Line widths and markers
    plt.rc('lines', linewidth=1.0)
    plt.rc('lines', markersize=4)

    # Grid
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle=':', linewidth=0.5)

    # Save settings
    plt.rc('savefig', dpi=300, bbox='tight', format='svg') # High DPI, tight bbox, SVG format
