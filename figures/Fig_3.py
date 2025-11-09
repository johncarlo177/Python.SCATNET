import numpy as np, matplotlib.pyplot as plt
def main():
    classes=['Asphalt','Meadows','Gravel','Trees','Painted metal','Bare soil','Bitumen','Shadows','Bricks']
    sf=np.array([97.4,98.0,95.1,95.8,96.2,96.0,95.5,94.2,95.9]); sc=np.array([98.1,98.8,96.4,97.6,97.2,97.1,96.8,95.6,97.3])
    x=np.arange(len(classes)); w=0.38; plt.figure(figsize=(9,4.8),dpi=150)
    plt.bar(x-w/2,sf,width=w,label='SpectralFormer'); plt.bar(x+w/2,sc,width=w,label='SCATNet (Ours)')
    plt.xticks(x,classes,rotation=35,ha='right',fontsize=8); plt.ylabel('Per-class Accuracy (%)'); plt.title('Fig. 3. Per-class Accuracy on Pavia Centre (illustrative)')
    plt.ylim(90,100); plt.legend(frameon=False); plt.tight_layout(); plt.savefig('../outputs/Fig_3_PerClass_Accuracy_PC.png',dpi=300,bbox_inches='tight')
if __name__ == '__main__': main()
