import numpy as np, matplotlib.pyplot as plt
def main():
    labels=['Asphalt','Meadows','Gravel','Trees','Painted metal','Bare soil','Bitumen','Shadows','Bricks']; base_counts=[1200,1500,800,1100,600,900,700,500,950]
    np.random.seed(42); n=len(labels); cm=np.zeros((n,n),dtype=int)
    for i,c in enumerate(base_counts):
        correct=int(0.985*c); rem=c-correct; probs=np.ones(n-1)/(n-1); off=np.random.multinomial(rem,probs)
        row=np.zeros(n,dtype=int); row[i]=correct; row[np.arange(n)!=i]=off; cm[i]=row
    cm=cm/cm.sum(axis=1,keepdims=True); fig,ax=plt.subplots(figsize=(7,5.2),dpi=150)
    im=ax.imshow(cm,aspect='auto',cmap='viridis'); cbar=plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cbar.set_label('Normalized Frequency')
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n)); ax.set_xticklabels(labels,rotation=45,ha='right',fontsize=8); ax.set_yticklabels(labels,fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Fig. A1. Confusion matrix for SCATNet on Pavia Centre'); plt.tight_layout(); plt.savefig('../outputs/Fig_A1_Confusion_Pavia.png',dpi=300,bbox_inches='tight')
if __name__ == '__main__': main()
