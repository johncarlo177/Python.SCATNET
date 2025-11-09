import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
def main():
    fig, ax = plt.subplots(figsize=(10,5), dpi=150)
    ax.axis('off')

    def add_box(x, y, w, h, text):
        ax.add_patch(Rectangle((x, y), w, h, edgecolor='black', facecolor='#e8f0ff'))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center')

    def arrow(a,b,c,d):
        ax.add_patch(FancyArrowPatch((a,b),(c,d),arrowstyle='->',mutation_scale=12,linewidth=1.0,color='black'))
    add_box(0.5,0.6,1.4,0.4,'Input HSI Cube\n(Bands×H×W)')
    add_box(2.3,0.6,1.6,0.4,'3D Stem\n(3D Conv over Spectral–Spatial)')
    add_box(4.3,0.6,1.6,0.4,'fPCA Projection\n(Spectral Reduction)')
    add_box(6.3,0.6,1.8,0.4,'Adaptive Axial Attention\n(Long-range Spectral–Spatial)')
    add_box(8.5,0.6,1.3,0.4,'Global Avg Pool')
    add_box(10.1,0.6,1.2,0.4,'Classifier\n(Linear → Softmax)')
    for (a,b) in [(1.9,2.3),(3.9,4.3),(5.9,6.3),(8.1,8.5),(9.8,10.1)]: arrow(a,0.8,b,0.8)
    ax.text(4.3,0.35,'Class-Balanced Focal Loss + Mixup',fontsize=9,ha='left')
    ax.text(6.3,0.25,'Outputs attention maps used in Fig. A2',fontsize=9,ha='left')
    ax.set_xlim(0,11.8); ax.set_ylim(0,1.2); plt.tight_layout()
    plt.savefig('../outputs/Fig_1_SCATNet_Architecture.png', dpi=300, bbox_inches='tight')
if __name__ == '__main__': main()
