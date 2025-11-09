import numpy as np, matplotlib.pyplot as plt
def main():
    rng = np.random.default_rng(7)
    def smooth_labels(h,w,nc,seed):
        r = np.random.default_rng(seed); x = r.integers(0,nc,size=(h,w))
        for _ in range(30): x = (x+np.roll(x,1,0)+np.roll(x,-1,0)+np.roll(x,1,1)+np.roll(x,-1,1))//5
        return x
    datasets=[('Pavia Centre',110,72,9),('Indian Pines',145,145,16),('Houston 2013',116,635,15)]
    fig,axes=plt.subplots(len(datasets),3,figsize=(9,7),dpi=150); plt.subplots_adjust(hspace=0.35,wspace=0.08)
    for i,(name,H,W,C) in enumerate(datasets):
        gt=smooth_labels(H,W,min(C,12),10+i)
        sf=(gt+(rng.random((H,W))<0.03)*rng.integers(0,min(C,12),(H,W)))%min(C,12)
        scat=(gt+(rng.random((H,W))<0.015)*rng.integers(0,min(C,12),(H,W)))%min(C,12)
        axes[i,0].imshow(gt,aspect='auto'); axes[i,1].imshow(sf,aspect='auto'); axes[i,2].imshow(scat,aspect='auto')
        axes[i,0].set_ylabel(name,fontsize=9)
        for j,t in enumerate(['Ground Truth','SpectralFormer','SCATNet (Ours)']): axes[i,j].set_title(t,fontsize=9)
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Fig. 2. Qualitative Classification Maps (placeholders)',fontsize=11); plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig('../outputs/Fig_2_Qualitative_Maps.png',dpi=300,bbox_inches='tight')
if __name__ == '__main__': main()
