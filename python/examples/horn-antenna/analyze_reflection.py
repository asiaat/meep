import numpy as np
import matplotlib.pyplot as plt
import argparse
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

def analyze_reflection(filename, savefig=True, dB=False, make_pdf=False):
    """
    Load reflection data from Meep output and plot it.
    Optionally create a PDF report.
    """
    # Load two-column data: frequency (GHz) and reflection values
    data = np.loadtxt(filename)
    freqs = data[:,0]
    refl  = data[:,1]

    if dB:
        refl = 20 * np.log10(np.abs(refl) + 1e-20)  # avoid log(0)
        ylabel = "Reflection (dB)"
    else:
        ylabel = "Reflection (arb. units)"

    # Plot reflection
    plt.figure(figsize=(8,5))
    plt.plot(freqs, refl, label="Reflection", lw=1.5)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel(ylabel)
    plt.title(f"Reflection vs Frequency\n{filename}")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()

    # Save / Show
    plotfile = None
    if savefig:
        plotfile = filename.rsplit(".",1)[0] + ("_dB.png" if dB else "_plot.png")
        plt.savefig(plotfile, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {plotfile}")
    plt.close()

    # PDF report
    if make_pdf:
        pdfname = filename.rsplit(".",1)[0] + "_report.pdf"
        doc = SimpleDocTemplate(pdfname, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(f"<b>Reflection Analysis Report</b>", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Analyzed file:</b> {filename}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Interpretation text
        interpretation = """
        The reflection coefficient (S11) describes how much of the input signal
        is reflected back from the antenna instead of being radiated.
        <br/><br/>
        - A value below -10 dB is generally considered acceptable for matching,
          meaning that less than 10% of the power is reflected.
        <br/>
        - The frequency range where reflection remains below -10 dB defines the
          antenna's effective operating bandwidth.
        <br/>
        - Sharp dips indicate resonances where the antenna is well matched.
        """
        story.append(Paragraph(interpretation, styles["BodyText"]))
        story.append(Spacer(1, 12))

        # Key stats
        min_refl = np.min(refl)
        freq_at_min = freqs[np.argmin(refl)]
        stats = f"""
        <b>Key Results:</b><br/>
        - Minimum reflection: {min_refl:.2f} {ylabel} at {freq_at_min:.2f} GHz<br/>
        - Frequency range analyzed: {freqs.min():.2f} – {freqs.max():.2f} GHz
        """
        story.append(Paragraph(stats, styles["BodyText"]))
        story.append(Spacer(1, 12))

        # Insert plot image
        if plotfile:
            story.append(Image(plotfile, width=400, height=300))

        doc.build(story)
        print(f"PDF report saved to {pdfname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Meep reflection data")
    parser.add_argument("filename", help="Input data file (two columns: freq, reflection)")
    parser.add_argument("--dB", action="store_true", help="Plot in dB scale (20*log10(|value|))")
    parser.add_argument("--no-save", action="store_true", help="Do not save the figure")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF report")
    args = parser.parse_args()

    analyze_reflection(args.filename, savefig=not args.no_save, dB=args.dB, make_pdf=args.pdf)
