import numpy as np, matplotlib.pyplot as plt
def main():
    labels=['SCATNet (full)','–fPCA','–Mix-up','–CB Loss']; oa=np.array([98.9,97.9,97.5,97.2]); aa=np.array([98.3,97.1,96.4,96.0]); flops=np.array([3.4,4.1,3.4,3.4])
    x=np.arange(len(labels)); bw=0.35; plt.figure(figsize=(8.5,4.8),dpi=150)
    plt.bar(x-bw/2,oa,width=bw,label='OA (%)'); plt.bar(x+bw/2,aa,width=bw,label='AA (%)')
    ax2=plt.twinx(); ax2.plot(x,flops,marker='o',linewidth=1.2,label='FLOPs (G)'); ax2.set_ylabel('FLOPs (G)')
    plt.xticks(x,labels,rotation=15,ha='right'); plt.ylabel('Accuracy (%)'); plt.title('Fig. 4. Ablation Study: Accuracy and Complexity')
    lines,labels1=plt.gca().get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels()
    plt.legend(lines+lines2,labels1+labels2,loc='lower left',frameon=False); plt.tight_layout(); plt.savefig('../outputs/Fig_4_Ablation_Accuracy_FLOPs.png',dpi=300,bbox_inches='tight')
if __name__ == '__main__': main()
